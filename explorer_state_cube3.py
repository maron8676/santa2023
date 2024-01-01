import pickle
import time
from ast import literal_eval
from collections import deque
from sys import stdin

import pandas as pd
import tqdm
from sympy.combinatorics import Permutation

readline = stdin.readline


def li():
    return list(map(int, readline().split()))


puzzle_info = pd.read_csv("puzzle_info.csv", index_col='puzzle_type')
base_moves = ["f0", "-f0", "f1", "-f1", "f2", "-f2", "r0", "-r0", "r1", "-r1", "r2", "-r2", "d0", "-d0", "d1", "-d1",
              "d2", "-d2"]
# base_moves = ["f0", "-f0", "f2", "-f2", "r0", "-r0", "r2", "-r2", "d0", "-d0", "d2", "-d2"]

cube_3type = 'cube_3/3/3'
selected_types = [cube_3type]

# cube_3/3/3の状態数
allowed_moves = literal_eval(puzzle_info.loc[cube_3type, 'allowed_moves'])
allowed_moves = {k: Permutation(v) for k, v in allowed_moves.items()}
key_list = list(allowed_moves.keys())
for key in key_list:
    allowed_moves["-" + key] = allowed_moves[key] ** (-1)

# initial_state = 'N0;N1;N2;N3;N4;N5;N6;N7;N8;N9;N10;N11;N12;N13;N14;N15;N16;N17;N18;N19;N20;N21;N22;N23;N24;N25;N26;N27;N28;N29;N30;N31;N32;N33;N34;N35;N36;N37;N38;N39;N40;N41;N42;N43;N44;N45;N46;N47;N48;N49;N50;N51;N52;N53'
initial_state = 'A;A;A;A;A;A;A;A;A;B;B;B;B;B;B;B;B;B;C;C;C;C;C;C;C;C;C;D;D;D;D;D;D;D;D;D;E;E;E;E;E;E;E;E;E;F;F;F;F;F;F;F;F;F'
solve_dict = {initial_state: []}

queue = deque([(initial_state.split(';'), [])])

before_time = time.time()
while len(queue) > 0:
    state = queue.popleft()
    for m in base_moves:
        p = allowed_moves[m]

        new_state = p(state[0])
        new_state_str = ';'.join(new_state)
        if new_state_str not in solve_dict:
            operation = []
            operation.extend(state[1])
            operation.append(m)
            if len(operation) < 6:
                queue.append((new_state, operation))
            solve_dict[new_state_str] = operation

    now = time.time()
    if now - before_time > 1:
        print(len(state[1]), len(queue), len(solve_dict))
        before_time = now

print(len(solve_dict))

with open('cube-333-6-af.pkl', mode='wb') as f:
    pickle.dump(solve_dict, f)
