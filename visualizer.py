from ast import literal_eval
from collections import deque

import pandas as pd
from sympy.combinatorics import Permutation


def print_wreath(left, right, state):
    cross = wreath_cross_dict[left]["cross"]
    left_cross = wreath_cross_dict[left]["left"]
    right_cross = wreath_cross_dict[left]["right"]

    left_points = [i for i in range(cross[1] + 1, left)]
    left_half_list = [left_points[:len(left_points) // 2], left_points[len(left_points) // 2:]]
    right_points = [i for i in range(left, right_cross[-1])]
    right_half_list = [right_points[:len(right_points) // 2], right_points[len(right_points) // 2:]]

    # leftの後半, rightの前半
    line = []
    for index in left_half_list[1]:
        line.append(f"{state[index]:4}")
    line.append(" " * 4)
    line.append(f"{state[cross[0]]:4}")
    line.append(" " * 4)
    for index in right_half_list[0]:
        line.append(f"{state[index]:4}")
    print(''.join(line))

    # cross
    for i in range(max(len(left_cross), len(right_cross))):
        line = []
        line.append(" " * 4 * len(left_half_list[1]))
        if i < len(right_cross):
            line.append(f"{state[right_cross[i]]:4}")
        else:
            line.append(" " * 4)
        line.append(" " * 4)
        if i < len(left_cross):
            line.append(f"{state[left_cross[i]]:4}")
        else:
            line.append(" " * 4)
        print(''.join(line))

    # leftの前半（逆順）, rightの後半（逆順）
    line = []
    if len(left_points) % 2 == 1:
        line.append(" " * 4)
    for index in left_half_list[0][::-1]:
        line.append(f"{state[index]:4}")
    line.append(" " * 4)
    line.append(f"{state[cross[1]]:4}")
    line.append(" " * 4)
    for index in right_half_list[1][::-1]:
        line.append(f"{state[index]:4}")
    print(''.join(line))

    print()
    pass


def print_globe(i, j, state):
    for row in range(i + 1):
        temp = []
        for index in range(row * j * 2, (row + 1) * j * 2):
            temp.append(f"{state[index]:3}")
        print(';'.join(temp))
    print()


puzzle_info = pd.read_csv("puzzle_info.csv", index_col='puzzle_type')
puzzles = pd.read_csv("puzzles.csv")
submission = pd.read_csv("submission.csv", index_col='id')

cube_prefix = "cube_"
wreath_prefix = "wreath_"
globe_prefix = "globe_"
wreath_cross_dict = {
    6: {
        "cross": [0, 2],
        "left": list(range(1, 2)),
        "right": list(range(9, 7, -1))
    },
    7: {
        "cross": [0, 2],
        "left": list(range(1, 2)),
        "right": list(range(11, 9, -1))
    },
    12: {
        "cross": [0, 3],
        "left": list(range(1, 3)),
        "right": list(range(21, 18, -1))
    },
    21: {
        "cross": [0, 6],
        "left": list(range(1, 6)),
        "right": list(range(39, 33, -1))
    },
    33: {
        "cross": [0, 9],
        "left": list(range(1, 9)),
        "right": list(range(63, 54, -1))
    },
    100: {
        "cross": [0, 25],
        "left": list(range(1, 25)),
        "right": list(range(197, 172, -1))
    }
}


class Puzzle:
    def __init__(self, initial_state, solution_state, allowed_moves, num_wildcards):
        self.initial_state = initial_state
        self.state = initial_state
        self.solution_state = solution_state
        self.allowed_moves = allowed_moves
        self.num_wildcards = num_wildcards

        self.move_list = []
        self.history_queue = deque([(self.state, self.move_list)])
        self.history_index = 0
        self.history_limit = 100

    def is_solved(self):
        diff = 0
        for i in range(len(self.state)):
            if self.state[i] != self.solution_state[i]:
                diff += 1
        return diff <= self.num_wildcards

    def move(self, move_key):
        raise NotImplementedError

    def parse_command(self, move_key):
        # 元に戻す
        if move_key == "u":
            if self.history_index <= 0:
                return True

            self.history_index -= 1
            history = self.history_queue[self.history_index]
            self.state = history[0]
            self.move_list = history[1]
            return True

        # やり直す
        if move_key == "re":
            if self.history_index + 1 >= len(self.history_queue):
                return True

            self.history_index += 1
            history = self.history_queue[self.history_index]
            self.state = history[0]
            self.move_list = history[1]
            return True

        # 現状の移動リストを表示
        if move_key == "s":
            print(".".join(self.move_list))
            return True

        # 初期化
        if move_key == "i":
            self.move_list = []
            self.state = self.initial_state
            self.history_queue = deque([(self.state, self.move_list)])
            self.history_index = 0
            return True

        return False

    def move_from_key_list(self, move_key_input_list):
        # 履歴管理のため、意図的に新しいリストオブジェクトを作る
        self.move_list = self.move_list + move_key_input_list
        for move_key_input in move_key_input_list:
            self.state = self.allowed_moves[move_key_input](self.state)

        # 新しい側の履歴を削除してから追加
        while self.history_index + 1 < len(self.history_queue):
            self.history_queue.pop()
        self.history_queue.append((self.state, self.move_list))
        self.history_index += 1

        # 上限超えたら古い順から削除
        while len(self.history_queue) > self.history_limit:
            self.history_queue.popleft()
            self.history_index -= 1

    def print(self):
        raise NotImplementedError


class Wreath(Puzzle):
    def __init__(self, left, right, initial_state, solution_state, allowed_moves, num_wildcards):
        super().__init__(initial_state, solution_state, allowed_moves, num_wildcards)
        self.left = left
        self.right = right
        # self.sugar_dict = self.create_sugar_list()

    def __str__(self):
        return f"wreath_{self.left}/{self.right} {self.state}"

    def move(self, move_key):
        if self.parse_command(move_key):
            return

        move_key_input_list = move_key.split(".")
        if any([move_key_input not in self.allowed_moves for move_key_input in move_key_input_list]):
            print(f"not allowed: {move_key}")
            return

        # 移動処理
        self.move_from_key_list(move_key_input_list)

    def print(self):
        print(f"move_num: {len(self.move_list)}")
        print_wreath(self.left, self.right, self.state)


class Globe(Puzzle):
    def __init__(self, row, column, initial_state, solution_state, allowed_moves, num_wildcards):
        super().__init__(initial_state, solution_state, allowed_moves, num_wildcards)
        self.row = row
        self.column = column
        self.sugar_dict = self.create_sugar_list()

    def __str__(self):
        return f"globe_{self.row}/{self.column} {self.state}"

    def move(self, move_key):
        if self.parse_command(move_key):
            return

        if move_key in self.sugar_dict:
            move_key = self.sugar_dict[move_key]

        move_key_input_list = move_key.split(".")
        if any([move_key_input not in self.allowed_moves for move_key_input in move_key_input_list]):
            print(f"not allowed: {move_key}")
            return

        # 移動処理
        self.move_from_key_list(move_key_input_list)

    def create_sugar_list(self):
        sugar_dict = dict()
        for i in range((self.row + 1) // 2):
            for j in range(self.column):
                sugar_dict[f"swap/{i}/{j}"] = f"f{j}.-r{i}.f{j}.r{i}.-r{self.row - i}.f{j}.r{self.row - i}.f{j}"
                sugar_dict[f"swapd/{i}/{j}"] = f"-r{i}.f{j}.r{i}.-r{self.row - i}.f{j}.r{self.row - i}"
        for i in range((self.row + 1) // 2):
            sugar_dict[f"flip1"] = f"-r{i}.f0.r{self.row - i}.f0.-r{self.row - i}"

        return sugar_dict

    def print(self):
        print(f"move_num: {len(self.move_list)}")
        print_globe(self.row, self.column, self.state)


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

    if puzzle_type.startswith(wreath_prefix):
        left, right = puzzle_type[len(wreath_prefix):].split("/")
        puzzle_state = Wreath(int(left), int(right), initial_state, solution_state, allowed_moves, num_wildcards)
    if puzzle_type.startswith(globe_prefix):
        row, column = puzzle_type[len(globe_prefix):].split("/")
        puzzle_state = Globe(int(row), int(column), initial_state, solution_state, allowed_moves, num_wildcards)

    while puzzle_state is not None and not puzzle_state.is_solved():
        puzzle_state.print()
        move_key = input("move: ")
        if move_key == "q":
            break

        puzzle_state.move(move_key)

    if puzzle_state.is_solved():
        print("solved!")
        print()
    puzzle_state.print()
    puzzle_state.parse_command("s")
