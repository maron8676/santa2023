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

    if puzzle_type.startswith(cube_prefix):
        f, r, d = puzzle_type[len(cube_prefix):].split("/")
        puzzle_state = Cube(int(f), list(range(len(solution_state))), initial_state, allowed_moves, 0)
    if puzzle_type.startswith(wreath_prefix):
        left, right = puzzle_type[len(wreath_prefix):].split("/")
        puzzle_state = Wreath(int(left), int(right), list(range(len(solution_state))), initial_state, allowed_moves, 0)
    if puzzle_type.startswith(globe_prefix):
        row, column = puzzle_type[len(globe_prefix):].split("/")
        puzzle_state = Globe(int(row), int(column), list(range(len(solution_state))), initial_state, allowed_moves, 0)

    moves = submission.loc[p_number, 'moves'].split('.')
    rev_moves = reverse_moves(moves)

    state_list = [puzzle_state.state]

    for move in rev_moves:
        puzzle_state.move(move)
        state_list.append(puzzle_state.state)

    print(state_list[-1])
    state_len = len(state_list)
    print(state_len)
    perm_dict = dict()
    for i in range(state_len):
        num_dict = dict()
        for index, num in enumerate(state_list[i]):
            num_dict[num] = index
        for j in range(i + 2, state_len):
            perm = [0] * len(puzzle_state.state)
            for index, num in enumerate(state_list[j]):
                perm[index] = num_dict[num]

            perm_dict[(i, j)] = perm

    print(len(perm_dict))
    hist = [0] * (len(solution_state) + 1)
    comp = list(range(len(solution_state)))
    for key in perm_dict:
        perm = perm_dict[key]
        diff = 0
        for i in range(len(perm)):
            if perm[i] != comp[i]:
                diff += 1
        hist[diff] += 1

    for i, value in enumerate(hist):
        print(i, value)

    # perm_set = set()
    # perm_dict2 = dict()
    # same_dict = dict()
    # for key in perm_dict:
    #     perm = perm_dict[key]
    #     perm_str = ';'.join(list(map(str, perm)))
    #     if perm_str not in perm_set:
    #         perm_set.add(perm_str)
    #         perm_dict2[perm_str] = key
    #     else:
    #         now_key = perm_dict2[perm_str]
    #         if key[0] <= now_key[0] and now_key[1] <= key[1]:
    #             perm_dict2[perm_str] = key
    #         same_dict[perm_str]
    # print(len(perm_set))
