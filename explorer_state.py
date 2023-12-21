import pickle
from ast import literal_eval
from collections import deque
from sys import stdin

import pandas as pd
from sympy.combinatorics import Permutation

readline = stdin.readline


def li():
    return list(map(int, readline().split()))


puzzle_info = pd.read_csv("puzzle_info.csv", index_col='puzzle_type')

base_moves = ["l", "-l", "r", "-r"]

# wreath_6/6の状態数
allowed_moves = literal_eval(puzzle_info.loc['wreath_6/6', 'allowed_moves'])
allowed_moves = {k: Permutation(v) for k, v in allowed_moves.items()}

initial_state = 'CACAAABBBB'
# state_set = set()
# state_set.add(initial_state)
# new_state_set = set()
# new_state_set.update(state_set)
solve_dict = {initial_state: []}

queue = deque([(initial_state, [])])

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
    # print(len(queue), len(solve_dict))

# print(solve_dict)
print(len(solve_dict))

# for i in range(1000):
#     for state in state_set:
#         for m in base_moves:
#             power = 1
#             if m[0] == "-":
#                 m = m[1:]
#                 power = -1
#             p = allowed_moves[m]
#
#             new_state = (p ** power)(state)
#             new_state_set.add(''.join(new_state))
#
#     state_set.update(new_state_set)
#     print(i, len(state_set))
#     if len(state_set) == 3150:
#         break
# print()

# wreath_7/7の状態数
allowed_moves = literal_eval(puzzle_info.loc['wreath_7/7', 'allowed_moves'])
allowed_moves = {k: Permutation(v) for k, v in allowed_moves.items()}

initial_state = 'CACAAAABBBBB'
# state_set = set()
# state_set.add(initial_state)
# new_state_set = set()
# new_state_set.update(state_set)

queue = deque([(initial_state, [])])

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
    # print(len(queue), len(solve_dict))

# print(solve_dict)
print(len(solve_dict))

with open('wreath67.pkl', mode='wb') as f:
    pickle.dump(solve_dict, f)

# for i in range(1000):
#     for state in state_set:
#         for m in base_moves:
#             power = 1
#             if m[0] == "-":
#                 m = m[1:]
#                 power = -1
#             p = allowed_moves[m]
#
#             new_state = (p ** power)(state)
#             new_state_set.add(''.join(new_state))
#
#     state_set.update(new_state_set)
#     print(i, len(state_set))
#     if len(state_set) == 16632:
#         break
# print()

# wreath_12/12の状態数
allowed_moves = literal_eval(puzzle_info.loc['wreath_12/12', 'allowed_moves'])
allowed_moves = {k: Permutation(v) for k, v in allowed_moves.items()}

initial_state = 'CAACAAAAAAAABBBBBBBBBB'
state_set = set()
state_set.add(initial_state)
# stack = [initial_state]

new_state_set = set()
new_state_set.update(state_set)

# while len(stack) > 0:
#     state = stack.pop()
#     for m in base_moves:
#         power = 1
#         if m[0] == "-":
#             m = m[1:]
#             power = -1
#         p = allowed_moves[m]
#
#         new_state = ''.join((p ** power)(state))
#         if new_state not in state_set:
#             stack.append(new_state)
#             state_set.add(new_state)
#     print(len(stack), len(state_set))

# for i in range(100):
#     for state in state_set:
#         for m in base_moves:
#             power = 1
#             if m[0] == "-":
#                 m = m[1:]
#                 power = -1
#             p = allowed_moves[m]
#
#             new_state = (p ** power)(state)
#             new_state_set.add(''.join(new_state))
#
#     state_set.update(new_state_set)
#     print(i, len(state_set))
#     if len(state_set) == 42678636: #24
#         break

# print(len(state_set))
