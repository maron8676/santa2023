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
sample_submission = pd.read_csv("submission2.csv", index_col='id')

selected_types = ['cube_3/3/3']
subset = puzzles[puzzles['puzzle_type'].isin(selected_types)]

allowed_moves = literal_eval(puzzle_info.loc['cube_3/3/3', 'allowed_moves'])
allowed_moves = {k: Permutation(v) for k, v in allowed_moves.items()}
key_list = list(allowed_moves.keys())
for key in key_list:
    allowed_moves["-" + key] = allowed_moves[key] ** (-1)

with open('cube-333-6-af.pkl', mode='rb') as f:
    solve6 = pickle.load(f)
with open('cube-333-6-dis-af.pkl', mode='rb') as f:
    dis_solved = pickle.load(f)

for index, row in subset.iterrows():
    moves = sample_submission.loc[index]['moves'].split('.')
    rev_moves = reverse_moves(moves)
    print(index)
    if row.solution_state != 'A;A;A;A;A;A;A;A;A;B;B;B;B;B;B;B;B;B;C;C;C;C;C;C;C;C;C;D;D;D;D;D;D;D;D;D;E;E;E;E;E;E;E;E;E;F;F;F;F;F;F;F;F;F':
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

    state = row.initial_state.split(';')
    stop_index = len(new_moves) - 1
    for index, move in enumerate(new_moves):
        state = allowed_moves[move](state)
        if ';'.join(state) in dis_solved[2] and (
                row.num_wildcards == '2' or row.num_wildcards == '4' or row.num_wildcards == '6'):
            stop_index = index
            break
        if ';'.join(state) in dis_solved[4] and (row.num_wildcards == '4' or row.num_wildcards == '6'):
            stop_index = index
            break
        if ';'.join(state) in dis_solved[6] and (row.num_wildcards == '6'):
            stop_index = index
            break
    print(state)
    print(len(moves), len(new_moves), stop_index + 1)
    print()
    sample_submission.loc[row.id]['moves'] = '.'.join(new_moves[:stop_index + 1])

with open('cube-333-6-full.pkl', mode='rb') as f:
    solve6 = pickle.load(f)
with open('cube-333-6-dis-full.pkl', mode='rb') as f:
    dis_solved = pickle.load(f)
for index, row in subset.iterrows():
    moves = sample_submission.loc[index]['moves'].split('.')
    rev_moves = reverse_moves(moves)
    print(index)
    if row.solution_state != 'N0;N1;N2;N3;N4;N5;N6;N7;N8;N9;N10;N11;N12;N13;N14;N15;N16;N17;N18;N19;N20;N21;N22;N23;N24;N25;N26;N27;N28;N29;N30;N31;N32;N33;N34;N35;N36;N37;N38;N39;N40;N41;N42;N43;N44;N45;N46;N47;N48;N49;N50;N51;N52;N53':
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
    stop_index = len(new_moves) - 1
    for index, move in enumerate(new_moves):
        state = allowed_moves[move](state)
        if ';'.join(state) in dis_solved[2] and (
                row.num_wildcards == '2' or row.num_wildcards == '4' or row.num_wildcards == '6'):
            stop_index = index
            break
        if ';'.join(state) in dis_solved[4] and (row.num_wildcards == '4' or row.num_wildcards == '6'):
            stop_index = index
            break
        if ';'.join(state) in dis_solved[6] and (row.num_wildcards == '6'):
            stop_index = index
            break
    print(state)
    print(len(moves), len(new_moves), stop_index + 1)
    print()
    sample_submission.loc[row.id]['moves'] = '.'.join(new_moves[:stop_index + 1])

with open('cube-333-6-another.pkl', mode='rb') as f:
    solve6 = pickle.load(f)
with open('cube-333-6-dis-another.pkl', mode='rb') as f:
    dis_solved = pickle.load(f)
for index, row in subset.iterrows():
    moves = sample_submission.loc[index]['moves'].split('.')
    rev_moves = reverse_moves(moves)
    print(index)
    if row.solution_state != 'A;B;A;B;A;B;A;B;A;B;C;B;C;B;C;B;C;B;C;D;C;D;C;D;C;D;C;D;E;D;E;D;E;D;E;D;E;F;E;F;E;F;E;F;E;F;A;F;A;F;A;F;A;F':
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
    stop_index = len(new_moves) - 1
    for index, move in enumerate(new_moves):
        state = allowed_moves[move](state)
        if ';'.join(state) in dis_solved[2] and (
                row.num_wildcards == '2' or row.num_wildcards == '4' or row.num_wildcards == '6'):
            stop_index = index
            break
        if ';'.join(state) in dis_solved[4] and (row.num_wildcards == '4' or row.num_wildcards == '6'):
            stop_index = index
            break
        if ';'.join(state) in dis_solved[6] and (row.num_wildcards == '6'):
            stop_index = index
            break
    print(state)
    print(len(moves), len(new_moves), stop_index + 1)
    print()
    sample_submission.loc[row.id]['moves'] = '.'.join(new_moves[:stop_index + 1])

sample_submission.to_csv('submission.csv')

my_submission = pd.read_csv('submission.csv')
total_score = score(puzzles, my_submission, 'id', 'moves', 'puzzle_info.csv')
print(f'Leaderboard: {total_score}')
