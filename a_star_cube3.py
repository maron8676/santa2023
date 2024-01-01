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


import heapq
import time


def heuristic(state, goal_state):
    """
    Heuristic function estimating the cost from the current state to the goal state.
    Here, we use the number of mismatched colors between the current state and the goal state.
    """
    return sum(s != g for s, g in zip(state, goal_state))


def a_star_search_with_timeout(initial_state, goal_state, timeout=300):
    """
    A* search algorithm with a timeout feature.

    :param initial_state: The starting state of the puzzle.
    :param goal_state: The target state to reach.
    :param timeout: The maximum time (in seconds) allowed for the search.
    :return: The shortest sequence of moves to solve the puzzle, or None if unsolved within timeout.
    """
    start_time = time.time()
    open_set = []  # Priority queue for states to explore
    heapq.heappush(open_set, (0, initial_state, []))  # Each entry: (priority, state, path taken)

    closed_set = set()  # Set to keep track of already explored states

    before_time = time.time()
    while open_set:
        if time.time() - start_time > timeout:
            return None  # Timeout check

        priority, current_state, path = heapq.heappop(open_set)
        now = time.time()
        if now - before_time > 1:
            print(priority, len(open_set), len(closed_set))
            before_time = now


        if current_state == goal_state:
            return path  # Goal state reached

        state_tuple = tuple(current_state)
        if state_tuple in closed_set:
            continue  # Skip already explored states

        closed_set.add(state_tuple)

        for move in allowed_moves:
            new_state = allowed_moves[move](current_state)
            new_state_tuple = tuple(new_state)
            if new_state_tuple not in closed_set:
                priority = len(path) + 1 + heuristic(new_state, goal_state)
                heapq.heappush(open_set, (priority, new_state, path + [move]))


# Modifying the A* search algorithm to improve efficiency
def improved_heuristic_with_wildcards(state, goal_state, num_wildcards):
    """
    Improved heuristic function considering wildcards.
    """
    mismatches = sum(s != g for s, g in zip(state, goal_state))
    return max(0, mismatches - num_wildcards)


def improved_a_star_search_with_wildcards(initial_state, goal_state, allowed_moves, num_wildcards, max_depth=30,
                                          timeout=100):
    """
    Improved A* search algorithm with wildcards, depth limit, and timeout.

    :param initial_state: List representing the initial state of the puzzle.
    :param goal_state: List representing the goal state of the puzzle.
    :param allowed_moves: Dictionary of allowed moves and their corresponding permutations.
    :param num_wildcards: Number of wildcards allowed for the puzzle.
    :param max_depth: Maximum depth to search to limit the search space.
    :param timeout: Time limit in seconds for the search.
    :return: Shortest sequence of moves to solve the puzzle, or None if no solution is found.
    """
    start_time = time.time()
    open_set = []
    heapq.heappush(open_set, (0, initial_state, [], num_wildcards))  # (priority, state, path, remaining wildcards)
    closed_set = set()

    while open_set:
        if time.time() - start_time > timeout:
            return None  # Timeout

        _, current_state, path, remaining_wildcards = heapq.heappop(open_set)

        if len(path) > max_depth:  # Depth limit
            continue

        if current_state == goal_state or improved_heuristic_with_wildcards(current_state, goal_state,
                                                                            remaining_wildcards) == 0:
            return path

        closed_set.add((tuple(current_state), remaining_wildcards))

        for move in allowed_moves:
            new_state = allowed_moves[move](current_state)
            new_state_tuple = tuple(new_state)
            if new_state_tuple not in closed_set:
                priority = len(path) + 1 + heuristic(new_state, goal_state)
                heapq.heappush(open_set, (priority, new_state, path + [move]))
        for move in allowed_moves:
            new_state = allowed_moves[move](current_state)
            if (tuple(new_state), remaining_wildcards) not in closed_set:
                priority = len(path) + 1 + improved_heuristic_with_wildcards(new_state, goal_state,
                                                                             remaining_wildcards)
                heapq.heappush(open_set, (priority, new_state, path + [move], remaining_wildcards))

    return None  # No solution found


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

# Testing the A* search algorithm with an example
test_initial_state = puzzles.loc[30]['initial_state'].split(';')
test_goal_state = puzzles.loc[30]['solution_state'].split(';')

# Running the A* search to find a solution
a_star_solution = a_star_search_with_timeout(test_initial_state, test_goal_state)
print(a_star_solution)

# with open('cube-333-6-af.pkl', mode='rb') as f:
#     solve6 = pickle.load(f)
#
# for index, row in subset.iterrows():
#     moves = sample_submission.loc[index]['moves'].split('.')
#     rev_moves = reverse_moves(moves)
#     print(index)
#     if row.solution_state != 'A;A;A;A;A;A;A;A;A;B;B;B;B;B;B;B;B;B;C;C;C;C;C;C;C;C;C;D;D;D;D;D;D;D;D;D;E;E;E;E;E;E;E;E;E;F;F;F;F;F;F;F;F;F':
#         print("skip")
#         continue
#
#     last_state_index = -1
#     last_state = None
#     last_moves = None
#     state = row.solution_state.split(';')
#     for index, move in enumerate(rev_moves):
#         state = allowed_moves[move](state)
#         if ';'.join(state) in solve6:
#             last_state_index = len(moves) - 1 - index
#             last_state = state
#             last_moves = solve6[';'.join(state)]
#     new_moves = []
#     new_moves.extend(moves[:last_state_index])
#     new_moves.extend(reverse_moves(last_moves))
#     print(len(moves), len(new_moves))
#
#     state = row.initial_state.split(';')
#     for move in new_moves:
#         state = allowed_moves[move](state)
#     print(state)
#     print()
#     sample_submission.loc[row.id]['moves'] = '.'.join(new_moves)
