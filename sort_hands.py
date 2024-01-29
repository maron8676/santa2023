import collections
from ast import literal_eval

import pandas as pd
from sympy.combinatorics import Permutation

from puzzle import Cube, Wreath, Globe

puzzle_info = pd.read_csv("puzzle_info.csv", index_col='puzzle_type')
puzzles = pd.read_csv("puzzles.csv")
submission = pd.read_csv("submission.csv", index_col='id')

cube_prefix = "cube_"
wreath_prefix = "wreath_"
globe_prefix = "globe_"


def reverse_moves(moves):
    res = moves[::-1]
    for i in range(len(res)):
        move = res[i]
        if move[0] == "-":
            res[i] = move[1:]
        else:
            res[i] = "-" + move
    return res


def reverse_move(move):
    if move[0] == "-":
        res = move[1:]
    else:
        res = "-" + move
    return res


def get_plane_num(m):
    if m[0] == "-":
        return int(m[2:])
    else:
        return int(m[1:])


def get_plane(m):
    if m[0] == "-":
        return m[1]
    else:
        return m[0]


if __name__ == "__main__":
    for p_number in range(398):
        print(p_number)
        puzzle = puzzles.loc[p_number]
        puzzle_type = puzzle["puzzle_type"]
        initial_state = puzzle["initial_state"].split(";")
        solution_state = puzzle["solution_state"].split(";")
        num_wildcards = puzzle["num_wildcards"]
        puzzle_state = None
        allowed_moves = literal_eval(puzzle_info.loc[puzzle_type, 'allowed_moves'])
        allowed_moves = {k: Permutation(v) for k, v in allowed_moves.items()}
        key_list = list(allowed_moves.keys())
        for key in key_list:
            allowed_moves["-" + key] = allowed_moves[key] ** (-1)

        if puzzle_type.startswith(cube_prefix):
            f, r, d = puzzle_type[len(cube_prefix):].split("/")
            puzzle_state = Cube(int(f), list(range(len(solution_state))), initial_state, allowed_moves, 0)
        if puzzle_type.startswith(wreath_prefix):
            continue
            # left, right = puzzle_type[len(wreath_prefix):].split("/")
            # puzzle_state = Wreath(int(left), int(right), list(range(len(solution_state))), initial_state, allowed_moves,
            #                       0)
        if puzzle_type.startswith(globe_prefix):
            continue
            # row, column = puzzle_type[len(globe_prefix):].split("/")
            # puzzle_state = Globe(int(row), int(column), list(range(len(solution_state))), initial_state, allowed_moves,
            #                      0)

        moves = submission.loc[p_number, 'moves'].split('.')
        new_moves = []
        index = 0
        stack = []
        for move in moves:
            if len(stack) == 0:
                stack.append(move)
                continue

            if get_plane(stack[-1]) == get_plane(move):
                stack.append(move)
                continue

            stack.sort(key=lambda x: get_plane_num(x))
            new_moves.extend(stack)
            stack.clear()

            stack.append(move)

        if len(stack) > 0:
            new_moves.extend(stack)
            stack.clear()

        submission.loc[p_number, 'moves'] = '.'.join(new_moves)
        # rev_moves = reverse_moves(moves)

    submission.to_csv("submission.csv")
