import sys
from ast import literal_eval
from collections import deque
from puzzle import Wreath, Globe

import pandas as pd
from sympy.combinatorics import Permutation


def print_globe(i, j, state):
    for row in range(i + 1):
        temp = []
        for index in range(row * j * 2, (row + 1) * j * 2):
            temp.append(f"{state[index]:3}")
        print(';'.join(temp))
    print()


puzzle_info = pd.read_csv("puzzle_info.csv", index_col='puzzle_type')
puzzles = pd.read_csv("puzzles.csv")
submission = pd.read_csv("submission.csv", index_col='id')

cube_prefix = "cube_"
wreath_prefix = "wreath_"
globe_prefix = "globe_"
wreath_cross_dict = {
    6: {
        "cross": [0, 2],
        "left": list(range(1, 2)),
        "right": list(range(9, 7, -1))
    },
    7: {
        "cross": [0, 2],
        "left": list(range(1, 2)),
        "right": list(range(11, 9, -1))
    },
    12: {
        "cross": [0, 3],
        "left": list(range(1, 3)),
        "right": list(range(21, 18, -1))
    },
    21: {
        "cross": [0, 6],
        "left": list(range(1, 6)),
        "right": list(range(39, 33, -1))
    },
    33: {
        "cross": [0, 9],
        "left": list(range(1, 9)),
        "right": list(range(63, 54, -1))
    },
    100: {
        "cross": [0, 25],
        "left": list(range(1, 25)),
        "right": list(range(197, 172, -1))
    }
}

if __name__ == "__main__":
    p_number = int(input("Puzzle No."))
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

    if puzzle_type.startswith(wreath_prefix):
        left, right = puzzle_type[len(wreath_prefix):].split("/")
        puzzle_state = Wreath(int(left), int(right), initial_state, solution_state, allowed_moves, num_wildcards)
    if puzzle_type.startswith(globe_prefix):
        row, column = puzzle_type[len(globe_prefix):].split("/")
        puzzle_state = Globe(int(row), int(column), initial_state, solution_state, allowed_moves, num_wildcards)

        puzzle_state.print()
        puzzle_state.solve()
        puzzle_state.print()
        print(".".join(puzzle_state.move_list))
