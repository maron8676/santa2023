from collections import defaultdict, deque
import pandas as pd
from sys import stdin

readline = stdin.readline


def li():
    return list(map(int, readline().split()))


my_submission = pd.read_csv('submission.csv')
submission_public = pd.read_csv("submission_public.csv", index_col='id')
sample_submission = pd.read_csv("sample_submission.csv", index_col='id')
print(my_submission.head())

for sub in my_submission.itertuples():
    id = getattr(sub, 'id')
    moves = getattr(my_submission.loc[id], 'moves').split('.')
    sample_moves = getattr(sample_submission.loc[id], 'moves').split('.')
    public_moves = getattr(submission_public.loc[id], 'moves').split('.')
    print(id, len(moves), len(public_moves), len(sample_moves))
