import random
import numpy as np
from tqdm import tqdm


def encode_state(state, base):
    unique_id = 0
    m = 1
    for i, value in enumerate(state):
        unique_id += value * m
        m *= base
    return unique_id


def encode_state_bases(state, bases):
    unique_id = 0
    m = 1
    for i, value in enumerate(state):
        unique_id += value * m
        m *= bases[i]
    return unique_id


def next_phase(phase):
  return (phase + 1) % 3


def generate_discuss_matrix(n, s_alive, agent, ind):
  discuss_actions = np.zeros((n, n), dtype=int)

  for i in range(n):
    if i == agent:
      discuss_actions[i][ind] = 1
    elif s_alive[i] == 1:
      choices = [j for j in range(n) if i != j and s_alive[j] == 0]
      j = np.random.choice(choices)
      discuss_actions[i][j] = 1

  return discuss_actions


def generate_discuss_matrix_test(n, s_alive, to_accuse, to_accuse_mafia, mafia_player):
  discuss_actions = np.zeros((n, n), dtype=int)

  for i in range(n):
    if s_alive[i] == 1:
      if i == mafia_player:
        discuss_actions[i][to_accuse_mafia] = 1
      else:
        discuss_actions[i][to_accuse] = 1

  return discuss_actions


# most accused / most accusing / random
def generate_vote_matrix(n, s_alive, agent, val):
  vote_actions = np.full(n, -1, dtype=int)

  for i in range(n):
    if i == agent:
      vote_actions[i] = val
    elif s_alive[i] == 1:
      vote_actions[i] = random.randint(0, 2)

  return vote_actions


def is_end_state(s_alive, agent, mafia_ind):
  if s_alive[mafia_ind] == 0:   # cilivian win
    return 1
  if sum(s_alive) == 2:   # mafia win
    return 0
  if s_alive[agent] == 0:   # agent dead
    return 2
  return -1


def is_end_state_test(s_alive, mafia_ind):
  if s_alive[mafia_ind] == 0:   # cilivian win
    return 1
  if sum(s_alive) == 2:   # mafia win
    return 0
  return -1


def get_reward(s_alive, agent, mafia_ind):
  if agent == mafia_ind:
    if s_alive[mafia_ind] == 0:
      return -10
    if sum(s_alive) == 2:
      return 10
  else:
    if s_alive[mafia_ind] == 0:
      return 10
    if sum(s_alive) == 2:
      return -10
    if s_alive[agent] == 0:
      return -10
  return 0