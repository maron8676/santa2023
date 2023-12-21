import json
from collections import defaultdict, deque
from sys import stdin

readline = stdin.readline


def li():
    return list(map(int, readline().split()))


line_list = []
puzzle_info_list = []
with open('puzzle_info.csv', mode='r', encoding='utf8') as f:
    for line in f:
        line_list.append(line.strip())

line_list.pop(0)
for line in line_list:
    columns = line.split(',', maxsplit=1)
    puzzle_info_list.append({
        'puzzle_type': columns[0],
        'allowed_moves': json.loads(columns[1][1:-1].replace("'", '"'))
    })

for puzzle_info in puzzle_info_list:
    print(puzzle_info['puzzle_type'], puzzle_info['allowed_moves'].keys())

# cube　ルービックキューブ
# print(puzzle_info_list[0]['allowed_moves']['f0'])
# for i in range(len(puzzle_info_list[0]['allowed_moves']['f0'])):
#     print(i, puzzle_info_list[0]['allowed_moves']['f0'][i])

# wreath　２つのリング
# print(puzzle_info_list[11]['allowed_moves']['l'])
# for i in range(len(puzzle_info_list[11]['allowed_moves']['l'])):
#     print(i, puzzle_info_list[11]['allowed_moves']['l'][i])
# print()
# for i in range(len(puzzle_info_list[11]['allowed_moves']['r'])):
#     print(i, puzzle_info_list[11]['allowed_moves']['r'][i])

# globe　球型のパズル
print(puzzle_info_list[17]['allowed_moves']['r0'])
for i in range(len(puzzle_info_list[17]['allowed_moves']['r0'])):
    print(i, puzzle_info_list[17]['allowed_moves']['r0'][i])
print()
for i in range(len(puzzle_info_list[17]['allowed_moves']['f0'])):
    print(i, puzzle_info_list[17]['allowed_moves']['f0'][i])

