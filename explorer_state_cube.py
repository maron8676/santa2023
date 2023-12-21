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

base_moves = ["f0", "-f0", "f1", "-f1", "r0", "-r0", "r1", "-r1", "d0", "-d0", "d1", "-d1"]

# cube_2/2/2の状態数
allowed_moves = literal_eval(puzzle_info.loc['cube_2/2/2', 'allowed_moves'])
allowed_moves = {k: Permutation(v) for k, v in allowed_moves.items()}

initial_state = 'AAAABBBBCCCCDDDDEEEEFFFF'
solve_dict = {initial_state: []}

queue = deque([(initial_state, [])])

before_time = time.time()
while len(queue) > 0:
    state = queue.popleft()
    for m in base_moves:
        power = 1
        if m[0] == "-":
            m = m[1:]
            power = -1
        p = allowed_moves[m]

        new_state = ''.join((p ** power)(state[0]))
        if new_state not in solve_dict:
            operation = []
            operation.extend(state[1])
            if power == -1:
                m = "-" + m
            operation.append(m)
            queue.append((new_state, operation))
            solve_dict[new_state] = operation

    now = time.time()
    if now - before_time > 1:
        print(len(state[1]), len(queue), len(solve_dict))
        before_time = now

print(len(solve_dict))

with open('cube-222.pkl', mode='wb') as f:
    pickle.dump(solve_dict, f)
