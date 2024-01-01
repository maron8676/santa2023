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


def reverse_moves(moves):
    res = moves[::-1]
    for i in range(len(res)):
        move = res[i]
        if move[0] == "-":
            res[i] = move[1:]
        else:
            res[i] = "-" + move
    return res


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


puzzle_info = pd.read_csv("puzzle_info.csv", index_col='puzzle_type')
puzzles = pd.read_csv("puzzles.csv")
sample_submission = pd.read_csv("submission.csv", index_col='id')

puzzle_type = 'globe_1/8'
selected_types = [puzzle_type]
subset = puzzles[puzzles['puzzle_type'].isin(selected_types)]

allowed_moves = literal_eval(puzzle_info.loc[puzzle_type, 'allowed_moves'])
allowed_moves = {k: Permutation(v) for k, v in allowed_moves.items()}
key_list = list(allowed_moves.keys())
for key in key_list:
    allowed_moves["-" + key] = allowed_moves[key] ** (-1)

with open('globe-1-8-6.pkl', mode='rb') as f:
    solve6 = pickle.load(f)

for index, row in subset.iterrows():
    moves = sample_submission.loc[index]['moves'].split('.')
    rev_moves = reverse_moves(moves)
    print(index)
    if row.solution_state != 'A;A;C;C;E;E;G;G;I;I;K;K;M;M;O;O;B;B;D;D;F;F;H;H;J;J;L;L;N;N;P;P':
        print("skip")
        continue

    last_state_index = -1
    last_state = None
    last_moves = None
    state = row.solution_state.split(';')
    for index, move in enumerate(rev_moves):
        state = allowed_moves[move](state)
        if ';'.join(state) in solve6:
            last_state_index = len(moves) - 1 - index
            last_state = state
            last_moves = solve6[';'.join(state)]
    new_moves = []
    new_moves.extend(moves[:last_state_index])
    new_moves.extend(reverse_moves(last_moves))
    print(len(moves), len(new_moves))

    state = row.initial_state.split(';')
    for move in new_moves:
        state = allowed_moves[move](state)
    print(state)
    print()
    sample_submission.loc[row.id]['moves'] = '.'.join(new_moves)

sample_submission.to_csv('submission2.csv')

my_submission = pd.read_csv('submission2.csv')
total_score = score(puzzles, my_submission, 'id', 'moves', 'puzzle_info.csv')
print(f'Leaderboard: {total_score}')
