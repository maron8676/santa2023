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


puzzle_info = pd.read_csv("puzzle_info.csv", index_col='puzzle_type')
base_moves = ["r0", "-r0", "r1", "-r1", "f0", "f1", "f2", "f3", "f4", "f5", "f6", "f7", "f8", "f9", "f10", "f11", "f12",
              "f13", "f14", "f15"]

globe_1_8type = 'globe_1/8'
selected_types = [globe_1_8type]

# globe_1/8の状態数
allowed_moves = literal_eval(puzzle_info.loc[globe_1_8type, 'allowed_moves'])
allowed_moves = {k: Permutation(v) for k, v in allowed_moves.items()}
key_list = list(allowed_moves.keys())
for key in key_list:
    if key[0] == "r":
        allowed_moves["-" + key] = allowed_moves[key] ** (-1)

initial_state = 'A;A;C;C;E;E;G;G;I;I;K;K;M;M;O;O;B;B;D;D;F;F;H;H;J;J;L;L;N;N;P;P'
solve_dict = {initial_state: []}

queue = deque([(initial_state.split(';'), [])])

before_time = time.time()
while len(queue) > 0:
    state = queue.popleft()
    for m in base_moves:
        p = allowed_moves[m]

        new_state = p(state[0])
        new_state_str = ';'.join(new_state)
        if new_state_str not in solve_dict:
            operation = []
            operation.extend(state[1])
            operation.append(m)
            if len(operation) < 6:
                queue.append((new_state, operation))
            solve_dict[new_state_str] = operation

    now = time.time()
    if now - before_time > 1:
        print(len(state[1]), len(queue), len(solve_dict))
        before_time = now

print(len(solve_dict))

with open('globe-1-8-6.pkl', mode='wb') as f:
    pickle.dump(solve_dict, f)
