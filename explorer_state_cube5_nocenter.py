import pickle
import time
from ast import literal_eval
from collections import deque, defaultdict
from sys import stdin

import pandas as pd
import tqdm
from sympy.combinatorics import Permutation

readline = stdin.readline


def li():
    return list(map(int, readline().split()))


def reverse_move(move):
    if move[0] == "-":
        res = move[1:]
    else:
        res = "-" + move
    return res


def get_plane_num(m):
    if m[0] == "-":
        return int(m[2:])
    else:
        return int(m[1:])


def get_plane(m):
    if m[0] == "-":
        return m[1]
    else:
        return m[0]


def is_same_plane(m1, m2):
    return get_plane(m1) == get_plane(m2)


puzzle_info = pd.read_csv("puzzle_info.csv", index_col='puzzle_type')
# base_moves = ["f0", "-f0", "f1", "-f1", "f2", "-f2", "r0", "-r0", "r1", "-r1", "r2", "-r2", "d0", "-d0", "d1", "-d1",
#               "d2", "-d2"]
base_moves = ["f0", "-f0", "f1", "-f1", "f3", "-f3", "f4", "-f4",
              "r0", "-r0", "r1", "-r1", "r3", "-r3", "r4", "-r4",
              "d0", "-d0", "d1", "-d1", "d3", "-d3", "d4", "-d4"]

cube_3type = 'cube_5/5/5'
selected_types = [cube_3type]

# cube_3/3/3の状態数
allowed_moves = literal_eval(puzzle_info.loc[cube_3type, 'allowed_moves'])
allowed_moves = {k: Permutation(v) for k, v in allowed_moves.items()}
key_list = list(allowed_moves.keys())
for key in key_list:
    allowed_moves["-" + key] = allowed_moves[key] ** (-1)

initial_state = 'N0;N1;N2;N3;N4;N5;N6;N7;N8;N9;N10;N11;N12;N13;N14;N15;N16;N17;N18;N19;N20;N21;N22;N23;N24;N25;N26;N27;N28;N29;N30;N31;N32;N33;N34;N35;N36;N37;N38;N39;N40;N41;N42;N43;N44;N45;N46;N47;N48;N49;N50;N51;N52;N53;N54;N55;N56;N57;N58;N59;N60;N61;N62;N63;N64;N65;N66;N67;N68;N69;N70;N71;N72;N73;N74;N75;N76;N77;N78;N79;N80;N81;N82;N83;N84;N85;N86;N87;N88;N89;N90;N91;N92;N93;N94;N95;N96;N97;N98;N99;N100;N101;N102;N103;N104;N105;N106;N107;N108;N109;N110;N111;N112;N113;N114;N115;N116;N117;N118;N119;N120;N121;N122;N123;N124;N125;N126;N127;N128;N129;N130;N131;N132;N133;N134;N135;N136;N137;N138;N139;N140;N141;N142;N143;N144;N145;N146;N147;N148;N149'
solve_dict = {initial_state: []}

queue = deque([(initial_state.split(';'), [])])

duplicate_list = []
before_time = time.time()
while len(queue) > 0:
    state = queue.popleft()
    for m in base_moves:
        if len(state[1]) > 0 and state[1][-1] == reverse_move(m):
            continue
        if len(state[1]) > 1 and state[1][-1] == m and state[1][-2] == m:
            continue
        if len(state[1]) > 0 and is_same_plane(state[1][-1], m) and get_plane_num(state[1][-1]) > get_plane_num(m):
            continue
        p = allowed_moves[m]

        new_state = p(state[0])
        new_state_str = ';'.join(new_state)
        if new_state_str not in solve_dict:
            operation = []
            operation.extend(state[1])
            operation.append(m)
            if len(operation) < 7:
                queue.append((new_state, operation))
            solve_dict[new_state_str] = operation
        else:
            duplicate_state = solve_dict[new_state_str]
            # if (len(duplicate_state) >= 2 and duplicate_state[-1] == reverse_move(m)
            #         and duplicate_state[-2] == reverse_move(state[1][-1])):
            #     pass
            if len(duplicate_state) != len(state[1]) + 1:
                # print("duplicate")
                # print(duplicate_state)
                # print(state[1], m)
                duplicate_list.append([".".join(duplicate_state), ".".join(state[1]) + f".{m}"])

    now = time.time()
    if now - before_time > 1:
        print(len(state[1]), len(queue), len(solve_dict))
        before_time = now

print(len(solve_dict))
with open("duplicate_cube5_nocenter.txt", mode="w") as f:
    for line in duplicate_list:
        f.write(f"{line[0]},{line[1]}\n")
