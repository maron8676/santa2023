from ast import literal_eval

import pandas as pd
from sympy.combinatorics import Permutation

from puzzle import Wreath, Globe

puzzle_info = pd.read_csv("puzzle_info.csv", index_col='puzzle_type')
puzzles = pd.read_csv("puzzles.csv")
submission = pd.read_csv("submission.csv", index_col='id')

cube_prefix = "cube_"
wreath_prefix = "wreath_"
globe_prefix = "globe_"

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

    while puzzle_state is not None and not puzzle_state.is_solved():
        puzzle_state.print()
        move_key = input("move: ")
        if move_key == "q":
            break

        puzzle_state.move(move_key)

    if puzzle_state.is_solved():
        print("solved!")
        print()
    puzzle_state.print()
    puzzle_state.parse_command("s")
