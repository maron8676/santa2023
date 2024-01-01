from collections import deque
import pickle
from ast import literal_eval
from dataclasses import dataclass
from itertools import chain, product
from typing import Dict, List

import pandas as pd
import tqdm
from sympy.combinatorics import Permutation

puzzle_info = pd.read_csv("puzzle_info.csv", index_col='puzzle_type')
puzzles = pd.read_csv("puzzles.csv")
sample_submission = pd.read_csv("sample_submission.csv", index_col='id')

cube_3type = 'cube_3/3/3'
selected_types = [cube_3type]
subset = puzzles[puzzles['puzzle_type'].isin(selected_types)]

allowed_moves = literal_eval(puzzle_info.loc[cube_3type, 'allowed_moves'])
allowed_moves = {k: Permutation(v) for k, v in allowed_moves.items()}
key_list = list(allowed_moves.keys())
for key in key_list:
    allowed_moves["-" + key] = allowed_moves[key] ** (-1)

print(f'cube_3/3/3 num: {len(subset)}')
cube_3a_solution = 'A;A;A;A;A;A;A;A;A;B;B;B;B;B;B;B;B;B;C;C;C;C;C;C;C;C;C;D;D;D;D;D;D;D;D;D;E;E;E;E;E;E;E;E;E;F;F;F;F;F;F;F;F;F'
cube_3a_solution_list = cube_3a_solution.split(';')
# base_moves = ["f0", "-f0", "f1", "-f1", "f2", "-f2", "r0", "-r0", "r1", "-r1", "r2", "-r2", "d0", "-d0", "d1", "-d1",
#               "d2", "-d2"]
base_moves = ["f0", "-f0", "f2", "-f2", "r0", "-r0", "r2", "-r2", "d0", "-d0", "d2", "-d2"]
center_moves = ["f1", "-f1", "r1", "-r1", "d1", "-d1"]

with open('cube-333-8.pkl', mode='rb') as f:
    solve8 = pickle.load(f)


class Rubiks3a:
    def __init__(self, initial_state):
        self.solution_state = cube_3a_solution.split(';')
        self.state = initial_state

    def apply_move(self, move):
        p = allowed_moves[move]
        next_state = p(self.state)
        return Rubiks3a(next_state)

    def count_solved(self):
        """
        揃っている個数をカウント
        """
        return sum([self.state[i] == cube_3a_solution_list[i] for i in range(9 * 6)])

    def is_solved(self):
        return self.count_solved() == 9 * 6


class Search:
    def __init__(self):
        self.current_solution = []

    def depth_limited_search(self, rubiks: Rubiks3a, depth):
        if depth == 0 and rubiks.is_solved():
            return rubiks.state

        if ''.join(rubiks.state) in solve8:
            return rubiks.state

        if depth == 0:
            return False

        # 枝刈り
        if prune(depth, rubiks):
            return False

        prev_moves = self.current_solution[-5:]
        for move in base_moves:
            if not is_move_available(prev_moves, move):
                continue
            self.current_solution.append(move)
            if self.depth_limited_search(rubiks.apply_move(move), depth - 1):
                return True
            self.current_solution.pop()

    def start_search(self, state, max_length=30):
        for depth in range(0, max_length):
            print(f"start searching length {depth}")
            solve_state = self.depth_limited_search(state, depth)
            if solve_state:
                return self.current_solution, solve_state
        return None, None


def is_move_available(prev_moves: List[str], move: str):
    if len(prev_moves) == 0:
        return True
    # 同じ方向に３回しない
    if len(prev_moves) >= 2 and prev_moves[-2] == move and prev_moves[-1] == move:
        return False

    # 直前が逆向きではない
    last_move = prev_moves[-1]
    if move[0] == '-' and last_move == move[1:]:
        return False
    if move[0] != '-' and last_move[1:] == move:
        return False

    # 同じ面は手前からやる（後で考える）
    plane = move[0] if move[0] != '-' else move[1]
    plane_num = move[1] if move[0] != '-' else move[2]
    for i in range(len(prev_moves) - 1, -1, -1):
        prev_move = prev_moves[i]
        prev_plane = prev_move[0] if prev_move[0] != '-' else prev_move[1]
        prev_plane_num = prev_move[1] if prev_move[0] != '-' else prev_move[2]
        if plane != prev_plane:
            break

        if plane_num < prev_plane_num:
            return False

    return True


def prune(depth, rubiks: Rubiks3a):
    """
    それ以上探索を進めても無意味ならTrueを返す
    """
    if depth <= 8 and ''.join(rubiks.state) not in solve8:
        return True

    return False


for index, row in subset.iterrows():
    if row.solution_state != cube_3a_solution:
        continue

    print(index)
    # 中央を全探索
    initial_state = row.initial_state.split(';')
    solve_dict = {''.join(initial_state): []}

    queue = deque([(initial_state, [])])
    center_ans = None
    while len(queue) > 0:
        state = queue.popleft()
        center = sum([state[0][i] == cube_3a_solution_list[i] for i in range(4, 54, 9)])
        if center == 6:
            center_ans = state[1]
            break

        for m in center_moves:
            p = allowed_moves[m]

            new_state = ''.join(p(state[0]))
            if new_state not in solve_dict:
                operation = []
                operation.extend(state[1])
                operation.append(m)
                if len(operation) < 8:
                    queue.append((new_state, operation))
                solve_dict[new_state] = operation
    print(center_ans)

    rubiks3a = Rubiks3a(row.initial_state.split(';'))
    for p in center_ans:
        rubiks3a.apply_move(p)
    search = Search()
    moves, state = search.start_search(rubiks3a, max_length=20)

    if moves is not None:
        print("found")
        center_ans.extend(moves)
        center_ans.extend(solve8[''.join(state)])
        sample_submission.loc[row.id]['moves'] = '.'.join(center_ans)

sample_submission.to_csv('rubiks.csv')
