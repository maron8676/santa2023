import pickle
import time
from ast import literal_eval
from collections import deque
from sys import stdin

import pandas as pd
import tqdm
from sympy.combinatorics import Permutation

readline = stdin.readline


def li():
    return list(map(int, readline().split()))


def count_pair(state, r, f):
    count = 0
    for i in range(r + 1):
        for j in range(f * 2):
            if state[i * f * 2 + j] == state[i * f * 2 + (j + 1) % (f * 2)]:
                count += 1

    return count


def count_correct(state, correct):
    count = 0
    return sum([state[i] == correct[i] for i in range(len(state))])


row = 3
column = 33
target_type = f'globe_{row}/{column}'
selected_types = [target_type]

puzzle_info = pd.read_csv("puzzle_info.csv", index_col='puzzle_type')

base_moves = []
for i in range(column):
    for j in range((row + 1) // 2):
        base_moves.append(f"f{i}.-r{j}.f{i}.r{j}.-r{row - j}.f{i}.r{row - j}.f{i}")
        base_moves.append(f"-r{j}.f{i}.r{j}.-r{row - j}.f{i}.r{row - j}")

# globeの状態数
allowed_moves = literal_eval(puzzle_info.loc[target_type, 'allowed_moves'])
allowed_moves = {k: Permutation(v) for k, v in allowed_moves.items()}
key_list = list(allowed_moves.keys())
for key in key_list:
    if key[0] == "r":
        allowed_moves["-" + key] = allowed_moves[key] ** (-1)
for moves in base_moves:
    move_list = moves.split('.')
    pos = list(range((row + 1) * column * 2))
    for move in move_list:
        pos = allowed_moves[move](pos)
    allowed_moves[moves] = Permutation(pos)

initial_state = list(range((row + 1) * column * 2))

# solve_dict = {';'.join(list(map(str, initial_state))): []}
solve_dict = {';'.join(list(map(str, initial_state))): []}
solve_dict_rev = dict()

queue = deque([(initial_state, [])])

before_time = time.time()
while len(queue) > 0:
    state = queue.popleft()
    for m in base_moves:
        p = allowed_moves[m]

        new_state = p(state[0])
        new_state_str = ';'.join(list(map(str, new_state)))
        if new_state_str not in solve_dict:
            operation = []
            operation.extend(state[1])
            operation.append(m)
            if len(operation) < 3:
                queue.append((new_state, operation))
            solve_dict[new_state_str] = operation
            solve_dict_rev['.'.join(operation)] = new_state

    now = time.time()
    if now - before_time > 1:
        print(len(state[1]), len(queue), len(solve_dict))
        before_time = now

print(len(solve_dict))

with open(f'globe-{row}-{column}-af_rotate.pkl', mode='wb') as f:
    pickle.dump(solve_dict, f)
with open(f'globe-{row}-{column}-af_rotate_rev.pkl', mode='wb') as f:
    pickle.dump(solve_dict_rev, f)
