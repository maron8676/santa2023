import pickle
from ast import literal_eval
from dataclasses import dataclass
from itertools import chain, product
from typing import Dict, List

import pandas as pd
import tqdm
from sympy.combinatorics import Permutation


class ParticipantVisibleError(Exception):
    pass


puzzle_info = pd.read_csv("puzzle_info.csv", index_col='puzzle_type')
puzzles = pd.read_csv("puzzles.csv")
sample_submission = pd.read_csv("sample_submission.csv", index_col='id')


def brute_force(puzzle_id, row, moves_list, allowed_moves):
    initial_state = row.initial_state.split(';')
    solution_state = row.solution_state.split(';')
    for moves in tqdm.tqdm(moves_list):
        temp_state = initial_state
        for m in moves.split("."):
            power = 1
            if m[0] == "-":
                m = m[1:]
                power = -1
            try:
                p = allowed_moves[m]
            except KeyError:
                raise ParticipantVisibleError(f"{m} is not an allowed move for {puzzle_id}.")

            temp_state = (p ** power)(temp_state)

        # Check that submitted moves solve puzzle
        num_wrong_facelets = sum(not (s == t) for s, t in zip(solution_state, temp_state))
        if row.num_wildcards >= num_wrong_facelets:
            return moves
    raise ParticipantVisibleError(f"Submitted moves do not solve {row.id}.")


def generate_moves(charset, maxlength):
    return list('.'.join(candidate)
                for candidate in chain.from_iterable(product(charset, repeat=i)
                                                     for i in range(1, maxlength + 1)))


def reverse_moves(moves):
    res = moves[::-1]
    for i in range(len(res)):
        move = res[i]
        if move[0] == "-":
            res[i] = move[1:]
        else:
            res[i] = "-" + move
    return res


with open('wreath6-7-12.pkl', mode='rb') as f:
    wreath6712 = pickle.load(f)
# print(wreath6712)
print(wreath6712['BAACCBBAAB'])

selected_types = ['wreath_6/6', 'wreath_7/7', 'wreath_12/12']
subset = puzzles[puzzles['puzzle_type'].isin(selected_types)]
print(f'brute_force num: {len(subset)}')

wreath_items = ['A', 'B', 'C']
# index_in_subset = 0
for index, row in subset.iterrows():
    # print(index_in_subset, len(subset))
    # index_in_subset += 1

    best_ans = []
    initial_state_list = row.initial_state.split(';')

    if row.num_wildcards == 2:
        for i in range(len(initial_state_list)):
            for r1 in wreath_items:
                temp1 = initial_state_list[i]
                initial_state_list[i] = r1
                for j in range(i + 1, len(initial_state_list)):
                    for r2 in wreath_items:
                        temp2 = initial_state_list[j]
                        initial_state_list[j] = r2
                        initial_state = ''.join(initial_state_list)
                        if initial_state in wreath6712:
                            ans = wreath6712[initial_state]
                            if len(best_ans) == 0 or len(ans) < len(best_ans):
                                best_ans = ans
                        initial_state_list[j] = temp2
                initial_state_list[i] = temp1
    else:
        best_ans = wreath6712[''.join(initial_state_list)]
    result = reverse_moves(best_ans)
    # print(initial_state, result)
    # result = brute_force(row.id, row, all_moves, allowed_moves)
    sample_submission.loc[row.id]['moves'] = '.'.join(result)

with open('cube-222.pkl', mode='rb') as f:
    cube222 = pickle.load(f)
print(cube222['DEDAEBABCACADCDFFFEEBFBC'])

selected_types = ['cube_2/2/2']
subset = puzzles[puzzles['puzzle_type'].isin(selected_types)]
print(f'brute_force num: {len(subset)}')

for index, row in subset.iterrows():
    solution_state = ''.join(row.solution_state.split(';'))
    if solution_state != 'AAAABBBBCCCCDDDDEEEEFFFF':
        continue

    initial_state = ''.join(row.initial_state.split(';'))
    result = cube222[initial_state]
    result = reverse_moves(result)
    sample_submission.loc[row.id]['moves'] = '.'.join(result)

sample_submission.to_csv('submission.csv')

"""Evaluation metric for Santa 2023."""


def score(
        solution: pd.DataFrame,
        submission: pd.DataFrame,
        series_id_column_name: str,
        moves_column_name: str,
        puzzle_info_path: str,
) -> float:
    """Santa 2023 evaluation metric.

    Parameters
    ----------
    solution : pd.DataFrame

    submission : pd.DataFrame

    series_id_column_name : str

    moves_column_name : str

    Returns
    -------
    total_num_moves : int
    """
    if list(submission.columns) != [series_id_column_name, moves_column_name]:
        raise ParticipantVisibleError(
            f"Submission must have columns {series_id_column_name} and {moves_column_name}."
        )

    puzzle_info = pd.read_csv(puzzle_info_path, index_col='puzzle_type')
    total_num_moves = 0
    for sol, sub in tqdm.tqdm(iterable=zip(solution.itertuples(), submission.itertuples()), total=submission.shape[0]):
        puzzle_id = getattr(sol, series_id_column_name)
        assert puzzle_id == getattr(sub, series_id_column_name)
        allowed_moves = literal_eval(puzzle_info.loc[sol.puzzle_type, 'allowed_moves'])
        allowed_moves = {k: Permutation(v) for k, v in allowed_moves.items()}
        puzzle = Puzzle(
            puzzle_id=puzzle_id,
            allowed_moves=allowed_moves,
            solution_state=sol.solution_state.split(';'),
            initial_state=sol.initial_state.split(';'),
            num_wildcards=sol.num_wildcards,
        )

        # Score submission row
        total_num_moves += score_puzzle(puzzle_id, puzzle, getattr(sub, moves_column_name))

    return total_num_moves


@dataclass
class Puzzle:
    """A permutation puzzle."""

    puzzle_id: str
    allowed_moves: Dict[str, List[int]]
    solution_state: List[str]
    initial_state: List[str]
    num_wildcards: int


def score_puzzle(puzzle_id, puzzle, sub_solution):
    """Score the solution to a permutation puzzle."""
    # Apply submitted sequence of moves to the initial state, from left to right
    moves = sub_solution.split('.')
    state = puzzle.initial_state
    for m in moves:
        power = 1
        if m[0] == "-":
            m = m[1:]
            power = -1
        try:
            p = puzzle.allowed_moves[m]
        except KeyError:
            raise ParticipantVisibleError(f"{m} is not an allowed move for {puzzle_id}.")
        state = (p ** power)(state)

    # Check that submitted moves solve puzzle
    num_wrong_facelets = sum(not (s == t) for s, t in zip(puzzle.solution_state, state))
    if num_wrong_facelets > puzzle.num_wildcards:
        raise ParticipantVisibleError(f"Submitted moves do not solve {puzzle_id}.")

    # The score for this instance is the total number of moves needed to solve the puzzle
    return len(moves)


scoring = True
if scoring:
    my_submission = pd.read_csv('submission.csv')

    total_score = score(puzzles,
                        my_submission,
                        'id',
                        'moves',
                        'puzzle_info.csv')

    print(f'Leaderboard: {total_score}')

my_submission = pd.read_csv('submission.csv')
sample_submission = pd.read_csv("sample_submission.csv", index_col='id')
print(my_submission.head())

selected_types = ['cube_2/2/2', 'wreath_6/6', 'wreath_7/7', 'wreath_12/12']
subset = puzzles[puzzles['puzzle_type'].isin(selected_types)]
print(subset.head())
for sub in subset.itertuples():
    id = getattr(sub, 'id')
    moves = getattr(my_submission.loc[id], 'moves').split('.')
    sample_moves = getattr(sample_submission.loc[id], 'moves').split('.')
    print(id, len(moves), len(sample_moves))

print(puzzles.groupby('puzzle_type').count())

# リースの状態数　実はそんなになかったりする？
# 実験してみる
