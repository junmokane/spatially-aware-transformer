import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import einsum
from einops import rearrange, repeat, reduce


def exists(val):
    return val is not None


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
        dim = args.tr.dim
        hidden_dim = args.tr.mlp.hidden_dim
        dropout = args.tr.mlp.dropout

        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim), nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x = self.mlp(x)
        return x
    

class TrLocalAttention(nn.Module):
    def __init__(self, args):
        super().__init__()
        dim = args.tr.dim
        heads = args.tr.heads
        dim_head = args.tr.dim_head
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


# Local Attention Block
class TrLABlock(nn.Module):  
    def __init__(self, args):
        super().__init__()
        dim = args.tr.dim
        self.norm = nn.LayerNorm(dim)
        self.attn = TrLocalAttention(args)
    
    def forward(self, x, **kwargs):
        norm_x = self.norm(x)
        out = self.attn(norm_x, **kwargs) + x
        return out


class TrLayer(nn.Module):
    def __init__(self, args):
        super(TrLayer, self).__init__()
        self.lablock = TrLABlock(args)
        self.mlp = PositionwiseFF(args)
        self.norm = nn.LayerNorm(args.tr.dim)
    
    def forward(self, x):
        o = self.lablock(x)
        norm_o = self.norm(o)
        o = self.mlp(norm_o) + o
        return o
