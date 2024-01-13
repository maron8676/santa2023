import pickle
import time
from ast import literal_eval
from collections import deque
from sys import stdin

import pandas as pd
import tqdm
from sympy.combinatorics import Permutation, PermutationGroup

puzzle_info = pd.read_csv("puzzle_info.csv", index_col='puzzle_type')
puzzles = pd.read_csv("puzzles.csv")
sample_submission = pd.read_csv("submission_public.csv", index_col='id')

size = 19
allowed_moves = literal_eval(puzzle_info.loc[f'cube_{size}/{size}/{size}', 'allowed_moves'])
allowed_moves = {k: Permutation(v) for k, v in allowed_moves.items()}
key_list = list(allowed_moves.keys())
for key in key_list:
    allowed_moves["-" + key] = allowed_moves[key] ** (-1)

values = list(allowed_moves.values())
G = PermutationGroup(*values)

print(G)
G.schreier_sims()
with open(f"cube{size}_schreier_sim.pkl", mode="wb") as f:
    pickle.dump(G, f)
