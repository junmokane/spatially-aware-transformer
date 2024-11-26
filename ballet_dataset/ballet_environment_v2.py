# Copyright 2021 DeepMind Technologies Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""A pycolab environment for going to the ballet.

A pycolab-based environment for testing memory for sequences of events. The
environment contains some number of "dancer" characters in (implicit) 3 x 3
squares within a larger 9 x 9 room. The agent starts in the center of the room.
At the beginning of an episode, the dancers each do a dance solo of a fixed
length, separated by empty time of a fixed length. The agent's actions do
nothing during the dances. After the last dance ends, the agent must go up to a
dancer, identified using language describing the dance. The agent is rewarded +1
for approaching the correct dancer, 0 otherwise.

The room is upsampled at a size of 9 pixels per square to render a view for the
agent, which is cropped in egocentric perspective, i.e. the agent is always in
the center of its view (see https://arxiv.org/abs/1910.00571).
"""
from absl import app
from absl import flags
from absl import logging

import dm_env

import numpy as np
from pycolab import cropping

from ballet_dataset import ballet_environment_core_v2 as ballet_core

FLAGS = flags.FLAGS

UPSAMPLE_SIZE = 9  # pixels per game square
SCROLL_CROP_SIZE = 33  # in game squares

DANCER_SHAPES = [
    "triangle", "empty_square", "plus", "inverse_plus", "ex", "inverse_ex",
    "circle", "empty_circle", "tee", "upside_down_tee",
    "h", "u", "upside_down_u", "vertical_stripes", "horizontal_stripes"
]

COLORS = {
    "red": np.array([255, 0, 0]),
    "green": np.array([0, 255, 0]),
    "blue": np.array([0, 0, 255]),
    "purple": np.array([128, 0, 128]),
    "orange": np.array([255, 165, 0]),
    "yellow": np.array([255, 255, 0]),
    "brown": np.array([128, 64, 0]),
    "pink": np.array([255, 64, 255]),
    "cyan": np.array([0, 255, 255]),
    "dark_green": np.array([0, 100, 0]),
    "dark_red": np.array([100, 0, 0]),
    "dark_blue": np.array([0, 0, 100]),
    "olive": np.array([100, 100, 0]),
    "teal": np.array([0, 100, 100]),
    "lavender": np.array([215, 200, 255]),
    "peach": np.array([255, 210, 170]),
    "rose": np.array([255, 205, 230]),
    "light_green": np.array([200, 255, 200]),
    "light_yellow": np.array([255, 255, 200]),
}


def _generate_template(object_name):
  """Generates a template object image, given a name with color and shape."""
  object_color, object_type = object_name.split()
  template = np.zeros((UPSAMPLE_SIZE, UPSAMPLE_SIZE))
  half = UPSAMPLE_SIZE // 2
  if object_type == "triangle":
    for i in range(UPSAMPLE_SIZE):
      for j in range(UPSAMPLE_SIZE):
        if (j <= half and i >= 2 * (half - j)) or (j > half and i >= 2 *
                                                   (j - half)):
          template[i, j] = 1.
  elif object_type == "square":
    template[:, :] = 1.
  elif object_type == "empty_square":
    template[:2, :] = 1.
    template[-2:, :] = 1.
    template[:, :2] = 1.
    template[:, -2:] = 1.
  elif object_type == "plus":
    template[:, half - 1:half + 2] = 1.
    template[half - 1:half + 2, :] = 1.
  elif object_type == "inverse_plus":
    template[:, :] = 1.
    template[:, half - 1:half + 2] = 0.
    template[half - 1:half + 2, :] = 0.
  elif object_type == "ex":
    for i in range(UPSAMPLE_SIZE):
      for j in range(UPSAMPLE_SIZE):
        if abs(i - j) <= 1 or abs(UPSAMPLE_SIZE - 1 - j - i) <= 1:
          template[i, j] = 1.
  elif object_type == "inverse_ex":
    for i in range(UPSAMPLE_SIZE):
      for j in range(UPSAMPLE_SIZE):
        if not (abs(i - j) <= 1 or abs(UPSAMPLE_SIZE - 1 - j - i) <= 1):
          template[i, j] = 1.
  elif object_type == "circle":
    for i in range(UPSAMPLE_SIZE):
      for j in range(UPSAMPLE_SIZE):
        if (i - half)**2 + (j - half)**2 <= half**2:
          template[i, j] = 1.
  elif object_type == "empty_circle":
    for i in range(UPSAMPLE_SIZE):
      for j in range(UPSAMPLE_SIZE):
        if abs((i - half)**2 + (j - half)**2 - half**2) < 6:
          template[i, j] = 1.
  elif object_type == "tee":
    template[:, half - 1:half + 2] = 1.
    template[:3, :] = 1.
  elif object_type == "upside_down_tee":
    template[:, half - 1:half + 2] = 1.
    template[-3:, :] = 1.
  elif object_type == "h":
    template[:, :3] = 1.
    template[:, -3:] = 1.
    template[half - 1:half + 2, :] = 1.
  elif object_type == "u":
    template[:, :3] = 1.
    template[:, -3:] = 1.
    template[-3:, :] = 1.
  elif object_type == "upside_down_u":
    template[:, :3] = 1.
    template[:, -3:] = 1.
    template[:3, :] = 1.
  elif object_type == "vertical_stripes":
    for j in range(half + UPSAMPLE_SIZE % 2):
      template[:, 2*j] = 1.
  elif object_type == "horizontal_stripes":
    for i in range(half + UPSAMPLE_SIZE % 2):
      template[2*i, :] = 1.
  else:
    raise ValueError("Unknown object: {}".format(object_type))

  if object_color not in COLORS:
    raise ValueError("Unknown color: {}".format(object_color))

  template = np.tensordot(template, COLORS[object_color], axes=0)

  return template


# Agent and wall templates
_CHAR_TO_TEMPLATE_BASE = {
    ballet_core.AGENT_CHAR:
        np.tensordot(
            np.ones([UPSAMPLE_SIZE, UPSAMPLE_SIZE]),
            np.array([255, 255, 255]),
            axes=0),
    ballet_core.WALL_CHAR:
        np.tensordot(
            np.ones([UPSAMPLE_SIZE, UPSAMPLE_SIZE]),
            np.array([40, 40, 40]),
            axes=0),
}


def get_scrolling_cropper(rows=9, cols=9, crop_pad_char=" "):
  return cropping.ScrollingCropper(rows=rows, cols=cols,
                                   to_track=[ballet_core.AGENT_CHAR],
                                   pad_char=crop_pad_char,
                                   scroll_margins=(None, None))


class BalletEnvironment(dm_env.Environment):
  """A Python environment API for pycolab ballet tasks."""

  def __init__(self, num_dancers, dance_delay, max_steps, rng=None):
    """Construct a BalletEnvironment that wraps pycolab games for agent use.

    This class inherits from dm_env and has all the expected methods and specs.

    Args:
      num_dancers: The number of dancers to use, between 1 and 8 (inclusive).
      dance_delay: How long to delay between the dances.
      max_steps: The maximum number of steps to allow in an episode, after which
          it will terminate.
      rng: An optional numpy Random Generator, to set a fixed seed use e.g.
          `rng=np.random.default_rng(seed=...)`
    """
    self._num_dancers = num_dancers
    self._dance_delay = dance_delay
    self._max_steps = max_steps

    # internal state
    if rng is None:
      rng = np.random.default_rng()
    self._rng = rng
    self._current_game = None       # Current pycolab game instance.
    self._state = None              # Current game step state.
    self._game_over = None          # Whether the game has ended.
    self._char_to_template = None   # Mapping of chars to sprite images.

    # rendering tools
    self._cropper = get_scrolling_cropper(SCROLL_CROP_SIZE, SCROLL_CROP_SIZE,
                                          " ")

  def _game_factory(self):
    """Samples dancers and positions, returns a pycolab core game engine."""
    target_dancer_index = self._rng.integers(self._num_dancers)
    motions = list(ballet_core.DANCE_SEQUENCES.keys())
    positions = ballet_core.DANCER_POSITIONS.copy()
    colors = list(COLORS.keys())
    shapes = DANCER_SHAPES.copy()
    dancer_chars = ballet_core.POSSIBLE_DANCER_CHARS.copy()
    
    '''
    fix positions and randomize the rest 
    '''
    self._rng.shuffle(motions)
    motions = motions + motions[2:5] + motions[0:2] + motions[0:1] + motions[2:3] + motions[3:4] + motions[5:6] + motions[6:7]
    self._rng.shuffle(dancer_chars)
    dancer_chars = dancer_chars[:self._num_dancers]
    self._rng.shuffle(colors)
    colors = colors[:self._num_dancers]
    self._rng.shuffle(shapes)
    shapes = shapes + shapes[:self._num_dancers - len(shapes)]
    '''
    random permute on positions and the rest (positions not shuffled yet)
    '''
    order = np.arange(self._num_dancers)
    self._rng.shuffle(order[4:])
    self._rng.shuffle(order[:8])  # to make most visited room is guaranteed in memory

    positions = [positions[i] for i in order]
    motions = [motions[i] for i in order]
    dancer_chars = [dancer_chars[i] for i in order]
    colors = [colors[i] for i in order]
    shapes = [shapes[i] for i in order]
    '''
    make space sequence
    '''
    spaces = []
    space_idx = np.arange(8)
    self._rng.shuffle(space_idx)
    for i, pos in enumerate(positions):
      y, x = pos 
      spaces.append(space_idx[(y - 2) // 4])
    spaces = np.array(spaces)
    
    '''
    make label
    '''
    max_room = 8
    cap = 9
    seq_len = len(motions)
    task_label = np.zeros((4)).astype(np.uint8)
    task_label[0] = np.random.randint(seq_len - cap, seq_len)  # Temporal FIFO
    task_label[1] = np.random.randint(0, cap)  # Temporal LIFO
    
    label_cand = []
    for i, pos in enumerate(positions):
      y, x = pos 
      if y == 2:
        label_cand.append(i)
    task_label[2] = np.random.choice(label_cand)  # Spatial LFU

    label_cand = []
    for i in range(max_room):
      label_cand.append((spaces == i).nonzero()[0][-1])
    task_label[3] = np.random.choice(label_cand)  # Spatial MFU

    '''
    make heuristics result
    '''
    task_heur = np.zeros((4, cap)).astype(np.uint8)
    task_heur[0, :] = np.arange(seq_len - cap, seq_len)  # Temporal FIFO
    task_heur[1, :] = np.arange(0, cap)  # Temporal LIFO
  
    def sfu(arr, opt):  # space least frequently used
      count = np.zeros(max_room).astype(np.int32)
      out = {}

      for i in range(len(arr)):
          
          sp = arr[i]
          if sp not in out.keys():
              out[sp] = []

          if i < cap:
              count[sp] += 1
              out[sp].append(i)
          else:
              
              if opt == 'slfu':
                  count_ = np.where(count == 0, 1000, count)
                  rem_sp = np.argmin(count_)
              elif opt == 'smfu':
                  count_ = np.where(count == 0, -1000, count)
                  rem_sp = np.argmax(count_)
              else:
                  exit()
              
              count[rem_sp] -= 1
              out[rem_sp].pop(0)
              out[sp].append(i)
              count[sp] += 1

      time_idx = []
      for k, v in out.items():
        if len(v) > 0:
          time_idx += v

      return out, count, time_idx

    # Spatial LFU
    out1, count1, time_idx = sfu(spaces, 'slfu')
    task_heur[2, :] = time_idx
    # Spatial MFU
    out2, count2, time_idx = sfu(spaces, 'smfu')
    task_heur[3, :] = time_idx
    
    assert task_label[0] in task_heur[0]
    assert task_label[1] in task_heur[1]
    assert task_label[2] in task_heur[2], f"{spaces} and {out1} and {count1}"
    assert task_label[3] in task_heur[3], f"{spaces} and {out2} and {count2}"

    '''
    add them to simulator!
    '''
    dancers_and_properties = []

    for dancer_i in range(self._num_dancers):
      if dancer_i == target_dancer_index:
        value = 1.
      else:
        value = 0.
      dancers_and_properties.append(
          (ballet_core.POSSIBLE_DANCER_CHARS[dancer_i],
           positions[dancer_i],
           motions[dancer_i],
           shapes[dancer_i],
           colors[dancer_i],
           value))

    self.info_dict = {
      'positions': positions,
      'spaces': spaces,
      'motions': motions,
      'task_heur': task_heur,
      'task_label': task_label,
    }
    
    logging.info("Making level with dancers_and_properties: %s",
                 dancers_and_properties)

    return ballet_core.make_game(
        dancers_and_properties=dancers_and_properties,
        dance_delay=self._dance_delay)

  def _render_observation(self, observation):
    """Renders from raw pycolab image observation to agent-usable ones."""
    observation = self._cropper.crop(observation)
    obs_rows, obs_cols = observation.board.shape
    image = np.zeros([obs_rows * UPSAMPLE_SIZE, obs_cols * UPSAMPLE_SIZE, 3],
                     dtype=np.float32)
    for i in range(obs_rows):
      for j in range(obs_cols):
        this_char = chr(observation.board[i, j])
        if this_char != ballet_core.FLOOR_CHAR:
          image[
              i * UPSAMPLE_SIZE:(i + 1) * UPSAMPLE_SIZE, j *
              UPSAMPLE_SIZE:(j + 1) * UPSAMPLE_SIZE] = self._char_to_template[
                  this_char]
    image /= 255.
    language = np.array(self._current_game.the_plot["instruction_string"])
    full_observation = (image, language)
    return full_observation

  def reset(self):
    """Start a new episode."""
    # Build a new game and retrieve its first set of state/reward/discount.
    self._current_game = self._game_factory()
    # set up rendering, cropping, and state for current game
    self._char_to_template = {
        k: _generate_template(v) for k, v in self._current_game.the_plot[
            "char_to_color_shape"]}
    self._char_to_template.update(_CHAR_TO_TEMPLATE_BASE)
    self._cropper.set_engine(self._current_game)
    self._state = dm_env.StepType.FIRST
    # let's go!
    observation, _, _ = self._current_game.its_showtime()
    observation = self._render_observation(observation)
    return dm_env.TimeStep(
        step_type=self._state,
        reward=None,
        discount=None,
        observation=observation)

  def step(self, action):
    """Apply action, step the world forward, and return observations."""
    # If needed, reset and start new episode.
    if self._state == dm_env.StepType.LAST:
      self._clear_state()
    if self._current_game is None:
      return self.reset()

    # Execute the action in pycolab.
    observation, reward, discount = self._current_game.play(action)

    self._game_over = self._is_game_over()
    reward = reward if reward is not None else 0.
    observation = self._render_observation(observation)

    # Check the current status of the game.
    if self._game_over:
      self._state = dm_env.StepType.LAST
    else:
      self._state = dm_env.StepType.MID

    return dm_env.TimeStep(
        step_type=self._state,
        reward=reward,
        discount=discount,
        observation=observation)

  @property
  def observation_spec(self):
    image_shape = (SCROLL_CROP_SIZE * UPSAMPLE_SIZE,
                   SCROLL_CROP_SIZE * UPSAMPLE_SIZE,
                   3)
    return (
        # vision
        dm_env.specs.Array(
            shape=image_shape, dtype=np.float32, name="image"),
        # language
        dm_env.specs.Array(
            shape=[], dtype=str, name="language"),
        )

  @property
  def action_spec(self):
    return dm_env.specs.BoundedArray(
        shape=[], dtype="int32",
        minimum=0, maximum=7,
        name="grid_actions")

  def _is_game_over(self):
    """Returns whether it is game over, either from the engine or timeout."""
    return (self._current_game.game_over or
            (self._current_game.the_plot.frame >= self._max_steps))

  def _clear_state(self):
    """Clear all the internal information about the game."""
    self._state = None
    self._current_game = None
    self._char_to_template = None
    self._game_over = None


def simple_builder(level_name):
  """Simplifies building from fixed defs.

  Args:
    level_name: '{num_dancers}_delay{delay_length}', where each variable is an
        integer. The levels used in the paper were:
        ['2_delay16', '4_delay16', '8_delay16',
         '2_delay48', '4_delay48', '8_delay48']

  Returns:
    A BalletEnvironment with the requested settings.
  """
  num_dancers, dance_delay = level_name.split("_")
  num_dancers = int(num_dancers)
  dance_delay = int(dance_delay[5:])
  max_steps = 320 if dance_delay == 16 else 1024
  level_args = dict(
      num_dancers=num_dancers,
      dance_delay=dance_delay,
      max_steps=max_steps)
  return BalletEnvironment(**level_args)


def main(argv):
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")
  env = simple_builder("4_delay16")
  for _ in range(3):
    obs = env.reset().observation
    for _ in range(300):
      obs = env.step(0).observation
    print(obs)

if __name__ == "__main__":
  app.run(main)