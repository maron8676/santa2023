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


puzzle_info = pd.read_csv("puzzle_info.csv", index_col='puzzle_type')
base_moves = ["r0", "-r0", "r1", "-r1", "f0", "f1", "f2", "f3", "f4", "f5", "f6", "f7", "f8", "f9", "f10", "f11", "f12",
              "f13", "f14", "f15"]
# base_moves = ["f0", "f1", "f2", "f3", "f4", "f5", "f6", "f7", "f8", "f9", "f10", "f11", "f12",
#               "f13", "f14", "f15"]
# base_moves = ["f0", "f2", "f4", "f6", "f8", "f10", "f12", "f14", 'r0.r0.r1.r1']

globe_1_8type = 'globe_1/8'
selected_types = [globe_1_8type]

# globe_1/8の状態数
allowed_moves = literal_eval(puzzle_info.loc[globe_1_8type, 'allowed_moves'])
allowed_moves = {k: Permutation(v) for k, v in allowed_moves.items()}
key_list = list(allowed_moves.keys())
for key in key_list:
    if key[0] == "r":
        allowed_moves["-" + key] = allowed_moves[key] ** (-1)
allowed_moves['f0.f4.f0'] = Permutation(
    [8, 9, 10, 11, 4, 5, 6, 7, 0, 1, 2, 3, 12, 13, 14, 15, 24, 25, 26, 27, 20, 21, 22, 23, 16, 17, 18, 19, 28, 29, 30,
     31])
allowed_moves['f0.f2'] = Permutation(
    [23, 22, 25, 24, 0, 1, 2, 3, 4, 5, 10, 11, 12, 13, 14, 15, 7, 6, 9, 8, 16, 17, 18, 19, 20, 21, 26, 27, 28, 29, 30,
     31])
allowed_moves['r1.r1'] = Permutation(
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 16,
     17])
allowed_moves['-r1.-r1'] = Permutation(
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 30, 31, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28,
     29])
allowed_moves['r0.r1'] = Permutation(
    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
     16])
allowed_moves['r0.r0.r1.r1'] = Permutation(
    [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0, 1, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 16,
     17])

# initial_state = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27,
#                  28, 29, 30, 31]
initial_state = 'A;A;C;C;E;E;G;G;I;I;K;K;M;M;O;O;B;B;D;D;F;F;H;H;J;J;L;L;N;N;P;P'.split(';')

# solve_dict = {';'.join(list(map(str, initial_state))): []}
solve_dict = {';'.join(initial_state): []}
solve_dict_rev = dict()

queue = deque([(initial_state, [])])

before_time = time.time()
while len(queue) > 0:
    state = queue.popleft()
    for m in base_moves:
        p = allowed_moves[m]

        new_state = p(state[0])
        new_state_list = [new_state]
        for i in range(15):
            new_state_list.append(allowed_moves['r0.r1'](new_state_list[-1]))
        new_state_str_list = list(map(lambda x: ';'.join(x), new_state_list))
        if not any([new_state_str in solve_dict for new_state_str in new_state_str_list]):
            operation = []
            operation.extend(state[1])
            operation.append(m)
            if len(operation) < 8:
                queue.append((new_state, operation))
            solve_dict[new_state_str_list[0]] = operation
            solve_dict_rev['.'.join(operation)] = new_state

    now = time.time()
    if now - before_time > 1:
        print(len(state[1]), len(queue), len(solve_dict))
        before_time = now

print(len(solve_dict))

with open('globe-1-8-af_rotate.pkl', mode='wb') as f:
    pickle.dump(solve_dict, f)
with open('globe-1-8-af_rotate_rev.pkl', mode='wb') as f:
    pickle.dump(solve_dict_rev, f)
