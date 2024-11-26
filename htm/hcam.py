from math import ceil
import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, repeat

# helpers

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def pad_to_multiple(t, multiple, dim = -2, value = 0.):
    seq_len = t.shape[dim]
    pad_to_len = ceil(seq_len / multiple) * multiple
    remainder = pad_to_len - seq_len

    if remainder == 0:
        return t

    zeroes = (0, 0) * (-dim - 1)
    padded_t = F.pad(t, (*zeroes, remainder, 0), value = value)
    return padded_t


class SinusoidalPosition(nn.Module):
    def __init__(
        self,
        dim,
        min_timescale = 2.,
        max_timescale = 1e4
    ):
        super().__init__()
        freqs = torch.arange(0, dim, min_timescale)
        inv_freqs = max_timescale ** (-freqs / dim)
        self.register_buffer('inv_freqs', inv_freqs)

    def forward(self, x):
        seq_len = x.shape[-2]
        seq = torch.arange(seq_len - 1, -1, -1.).to(x.device)
        sinusoidal_inp = rearrange(seq, 'n -> n ()') * rearrange(self.inv_freqs, 'd -> () d')
        pos_emb = torch.cat((sinusoidal_inp.sin(), sinusoidal_inp.cos()), dim = -1)
        return pos_emb


class PositionwiseFF(nn.Module):
    def __init__(self, args):
        super(PositionwiseFF, self).__init__()
        dim = args.hcam.dim
        hidden_dim = args.hcam.mlp.hidden_dim
        dropout = args.hcam.mlp.dropout

        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim), nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x = self.mlp(x)
        return x


class LocalAttention(nn.Module):
    def __init__(self, args):
        super().__init__()
        dim = args.hcam.dim
        heads = args.hcam.heads
        dim_head = args.hcam.dim_head
        self.dim = dim

        # attention
        self.scale = dim_head ** -0.5
        self.heads = heads
        inner_dim = dim_head * heads

        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias = False)
        self.to_out = nn.Linear(inner_dim, dim)


    def forward(self, x, mask = None):

        # multi head attention
        h = self.heads
        q, k, v = self.to_q(x), *self.to_kv(x).chunk(2, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b i (h d) -> (b h) i d', h=h), (q, k, v))
        q = q * self.scale
        sim = einsum('b i d, b j d -> b i j', q, k)

        attn = sim.softmax(dim = -1)
        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) i d -> b i (h d)', h = h)
        return self.to_out(out)


class LABlock(nn.Module):  
    def __init__(self, args):
        super().__init__()
        dim = args.hcam.dim
        self.norm = nn.LayerNorm(dim)
        self.attn = LocalAttention(args)
    
    def forward(self, x, **kwargs):
        norm_x = self.norm(x)
        out = self.attn(norm_x, **kwargs) + x
        return out


class Attention(nn.Module):
    def __init__(
        self,
        dim,
        dim_head = 64,
        heads = 8,
    ):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        inner_dim = dim_head * heads

        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias = False)
        self.to_out = nn.Linear(inner_dim, dim)

    def forward(
        self,
        x,
        mems,
        mask = None
    ):
        h = self.heads
        q, k, v = self.to_q(x), *self.to_kv(mems).chunk(2, dim = -1)

        q, k, v = map(lambda t: rearrange(t, 'b ... (h d) -> (b h) ... d', h = h), (q, k, v))
        q = q * self.scale

        sim = einsum('b m i d, b m i j d -> b m i j', q, k)

        if exists(mask):
            mask = repeat(mask, 'b ... -> (b h) ...', h = h)
            mask_value = -torch.finfo(sim.dtype).max
            sim = sim.masked_fill(~mask, mask_value)

        attn = sim.softmax(dim = -1)
        out = einsum('... i j, ... i j d -> ... i d', attn, v)
        out = rearrange(out, '(b h) ... d -> b ... (h d)', h = h)
        return self.to_out(out)


class HCAMAttention(nn.Module):
    def __init__(self, args):
        super().__init__()
        dim = args.hcam.dim
        heads = args.hcam.heads
        topk_mems = args.hcam.topk_mems
        mem_chunk_size = args.hcam.mem_chunk_size
        dim_head = args.hcam.dim_head
        add_pos_enc = args.hcam.add_pos_enc
        eps = args.hcam.eps

        self.dim = dim
        self.eps = eps
        self.scale = dim ** -0.5

        self.to_summary_queries = nn.Linear(dim, dim)
        self.to_summary_keys = nn.Linear(dim, dim)

        self.attn = Attention(dim=dim, heads=heads, dim_head=dim_head)

        self.topk_mems = topk_mems
        self.mem_chunk_size = mem_chunk_size
        self.pos_emb = SinusoidalPosition(dim=dim) if add_pos_enc else None

    def forward(
        self,
        queries,  
        memories,
        mask = None,
        chunk_attn_mask = None
    ):
        dim, query_len, mem_chunk_size, topk_mems, scale, eps = self.dim, queries.shape[1], self.mem_chunk_size, self.topk_mems, self.scale, self.eps

        # pad memories, and the memory mask, if needed
        # and then divide into chunks
        memories = pad_to_multiple(memories, mem_chunk_size, dim=-2, value=0.)
        memories = rearrange(memories, 'b (n c) d -> b n c d', c=mem_chunk_size)
        # print(memories[0, 0, :, -64:])
        # print(memories[0, 1, :, -64:])

        if exists(mask):
            mask = pad_to_multiple(mask, mem_chunk_size, dim=-1, value=False)
            mask = rearrange(mask, 'b (n c) -> b n c', c=mem_chunk_size)

        # summarize memories through mean-pool, accounting for mask
        if exists(mask):
            mean_mask = rearrange(mask, '... -> ... ()')
            memories = memories.masked_fill(~mean_mask, 0.)
            numer = memories.sum(dim = 2)
            denom = mean_mask.sum(dim = 2)
            summarized_memories = numer / (denom + eps)
        else:
            summarized_memories = memories.mean(dim=2)

        # derive queries and summarized memory keys
        summary_queries = self.to_summary_queries(queries)
        # summarized_memories = summarized_memories.detach()
        summarized_memories = summarized_memories

        # print(summarized_memories[0, 0, -64:])
        # print(summarized_memories[0, 1, -64:])

        # if 'sum' in self.space_emb_type:
        #     summarized_memories = summarized_memories + space_emb_mem
        # elif 'cat' in self.space_emb_type:
        #     summarized_memories = torch.cat((summarized_memories, space_emb_mem), dim=-1)

        summary_keys = self.to_summary_keys(summarized_memories)

        # do a single head attention over summary keys
        sim = einsum('b i d, b j d -> b i j', summary_queries, summary_keys) * scale
        mask_value = -torch.finfo(sim.dtype).max

        if exists(mask):
            chunk_mask = mask.any(dim=2)
            chunk_mask = rearrange(chunk_mask, 'b j -> b () j')
            sim = sim.masked_fill(~chunk_mask, mask_value)

        if exists(chunk_attn_mask):
            sim = sim.masked_fill(~chunk_attn_mask, mask_value)

        topk_logits, topk_indices = sim.topk(k=topk_mems, dim=-1)
        weights = topk_logits.softmax(dim=-1)
    
        # ready queries for in-memory attention
        queries = repeat(queries, 'b n d -> b k n d', k=topk_mems)

        # select the topk memories
        memories = repeat(memories, 'b m j d -> b m i j d', i=query_len)
        mem_topk_indices = repeat(topk_indices, 'b i m -> b m i j d', j=mem_chunk_size, d=dim)
        selected_memories = memories.gather(1, mem_topk_indices)
        
        # positional encoding on in-chunk (chunk size position embedding)
        if exists(self.pos_emb):
            pos_emb = self.pos_emb(memories)
            selected_memories = selected_memories + rearrange(pos_emb, 'n d -> () () () n d')

        # select the mask
        selected_mask = None
        if exists(mask):
            mask = repeat(mask, 'b m j -> b m i j', i = query_len)
            mask_topk_indices = repeat(topk_indices, 'b i m -> b m i j', j = mem_chunk_size)
            selected_mask = mask.gather(1, mask_topk_indices)

        # now do in-memory attention
        within_mem_output = self.attn(
            queries,
            # selected_memories.detach(),
            selected_memories,
            mask = selected_mask
        )

        # weight the in-memory attention outputs
        weighted_output = within_mem_output * rearrange(weights, 'b i m -> b m i ()')
        output = weighted_output.sum(dim=1)
        return output


class HCAMBlock(nn.Module):
    def __init__(self, args):
        super().__init__()
        dim = args.hcam.dim
        self.norm = nn.LayerNorm(dim)
        self.attn = HCAMAttention(args)
    
    def forward(self, queries, memories, mask, **kwargs):
        norm_queries = self.norm(queries)
        out = self.attn(norm_queries, memories, mask, **kwargs) + queries
        return out


class HCAMLayer(nn.Module):
    def __init__(self, args):
        super(HCAMLayer, self).__init__()
        self.hcamblock = HCAMBlock(args)
        self.lablock = LABlock(args)
        self.mlp = PositionwiseFF(args)
        self.norm = nn.LayerNorm(args.hcam.dim)
    
    def forward(self, x):
        q, m, mask = x
        o = self.lablock(q)
        o = self.hcamblock(o, m, mask)  # memory does not change
        norm_o = self.norm(o)
        o = self.mlp(norm_o) + o
        return (o, m, mask)