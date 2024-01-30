import itertools
import pickle
import time
from collections import defaultdict, deque
from sys import stdin
from sympy.combinatorics import Permutation

import tqdm

readline = stdin.readline


def li():
    return list(map(int, readline().split()))


class Puzzle:
    def __init__(self, initial_state, solution_state, allowed_moves, num_wildcards):
        self.initial_state = initial_state
        self.state = initial_state
        self.solution_state = solution_state
        self.allowed_moves = allowed_moves
        self.num_wildcards = num_wildcards
        self.save_slot = dict()
        self.solve_dict_rev = None

        self.history_limit = 1000
        self.initialize()

    def is_solved(self):
        diff = 0
        for i in range(len(self.state)):
            if self.state[i] != self.solution_state[i]:
                diff += 1
        return diff <= self.num_wildcards

    def move(self, move_key):
        if self.parse_command(move_key):
            return

        move_key_input_list = move_key.split(".")
        if any([move_key_input not in self.allowed_moves for move_key_input in move_key_input_list]):
            print(f"not allowed: {move_key}")
            return

        # 移動処理
        self.move_from_key_list(move_key_input_list)

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
            self.initialize()
            return True

        return False

    def initialize(self):
        self.move_list = []
        self.state = self.initial_state
        self.history_queue = deque([(self.state, self.move_list)])
        self.history_index = 0

    def save(self, key):
        self.save_slot[key] = (self.move_list, self.state)

    def load(self, key):
        loaded = self.save_slot[key]
        self.move_list = loaded[0]
        self.state = loaded[1]
        self.history_queue = deque([(self.state, self.move_list)])
        self.history_index = 0

    def move_from_key_list(self, move_key_input_list, verbose=False):
        # 履歴管理のため、意図的に新しいリストオブジェクトを作る
        self.move_list = self.move_list + move_key_input_list
        for move_key_input in move_key_input_list:
            self.state = self.allowed_moves[move_key_input](self.state)
            if verbose:
                self.print()
                time.sleep(0.1)

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


class Cube(Puzzle):
    def __init__(self, size, initial_state, solution_state, allowed_moves, num_wildcards):
        super().__init__(initial_state, solution_state, allowed_moves, num_wildcards)
        self.size = size

    def __str__(self):
        return f"cube_{self.size}/{self.size}/{self.size} {self.state}"

    def print(self):
        print(f"move_num: {len(self.move_list)}")
        print_cube(self.size, self.state)

    def move(self, move_key):
        if self.parse_command(move_key):
            return

        move_key_input_list = move_key.split(".")
        if any([move_key_input not in self.allowed_moves for move_key_input in move_key_input_list]):
            print(f"not allowed: {move_key}")
            return

        # 移動処理
        self.move_from_key_list(move_key_input_list)


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
    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcd"

    def __init__(self, row, column, initial_state, solution_state, allowed_moves, num_wildcards):
        super().__init__(initial_state, solution_state, allowed_moves, num_wildcards)
        self.row = row
        self.column = column
        self.sugar_dict = self.create_sugar_list()
        self.has_each_color = self.solution_state[0] != "A"
        self.same_num = 1
        for index in range(1, self.column):
            if self.solution_state[index] == "A":
                self.same_num += 1
        self.unsolved_pair_list = [[] for _ in range((self.row + 1) // 2)]
        self.flip_list = [[] for _ in range((self.row + 1) // 2)]
        # print(self.has_each_color, self.solution_state[0])

    def __str__(self):
        return f"globe_{self.row}/{self.column} {self.state}"

    def solve(self):
        # 1. 上下の組を作る
        #   1.1. 反対側に置いてあるものを探す
        #   1.2. 移動幅は1, 3, 5, ..., n - 1　これを偶数個組み合わせる
        #   1.3. フリップが偶数でなければやり直す
        for i in range((self.row + 1) // 2):
            self.save(i)
            self.solve_row_pair(i)
            self.update_flip()

            retry = 0
            all_odd = all([not self.is_even_flip(j) for j in range((self.row + 1) // 2)])
            if all_odd:
                self.move("f0")
                self.update_flip()
            all_even = all([self.is_even_flip(j) for j in range((self.row + 1) // 2)])
            while not self.is_even_flip(i) and self.column % 2 == 0 or not all_even and self.column % 2 == 1:
                print("odd")
                self.print()
                print(self.flip_list)

                self.load(i)
                retry += 1
                for _ in range(retry):
                    self.move(f"r{i}")
                self.solve_row_pair(i)
                self.update_flip()

                all_odd = all([not self.is_even_flip(j) for j in range((self.row + 1) // 2)])
                if all_odd:
                    self.move("f0")
                    self.update_flip()
                all_even = all([self.is_even_flip(j) for j in range((self.row + 1) // 2)])
            print(self.flip_list)

        # 2. 必要なら中央を解く　書き出し必要
        if self.row % 2 == 0:
            index = self.row // 2
            first = self.solution_state[index * self.column * 2]
            found = -1
            for cell in range(index * self.column * 2, (index + 1) * self.column * 2):
                before = cell - 1 if cell > index * self.column * 2 else cell + self.column * 2 - 1
                if self.state[cell] == first and self.state[before] != first:
                    found = cell
                    break
            diff = found - index * self.column * 2
            if diff <= self.column:
                for _ in range(diff):
                    self.move(f"r{index}")
            else:
                diff = (index + 1) * self.column * 2 - found
                for _ in range(diff):
                    self.move(f"-r{index}")

        # 3. 上下のflipを直す
        #   3.1. 反対側まで移動してフリップする
        for i in range((self.row + 1) // 2):
            self.solve_row_flip(i)

        # 4. 左右の順番入れ替える
        for i in range((self.row + 1) // 2):
            self.fix_order(i)

    def solve_row_pair(self, i):
        before_state = self.state
        self.update_unsolved_pair(i)
        self.update_flip()

        while len(self.unsolved_pair_list[i]) > 0:
            self.print()
            print(len(self.unsolved_pair_list[i]))
            print(self.unsolved_pair_list)
            print(self.get_flip_list(i))
            for (index1, index2) in itertools.product(range(len(self.unsolved_pair_list[i])), repeat=2):
                if index1 == index2:
                    continue
                p1 = self.unsolved_pair_list[i][index1]
                p2 = self.unsolved_pair_list[i][index2]
                if self.column % 2 == 0:
                    # 同じ側で距離が偶数
                    if self.is_pair(p1[0], p2[0]) and (p1[0] - p2[0]) % 2 == 0:
                        self.create_pair_same_row(i, p1, p2)
                        break
                    # 違う側で距離が奇数
                    if self.is_pair(p1[0], p2[1]) and (p1[0] - p2[0]) % 2 == 1:
                        self.create_pair_opposite_row(i, p1, p2)
                        break
                else:
                    # 同じ側で距離が奇数
                    if self.is_pair(p1[0], p2[0]) and (p1[0] - p2[0]) % 2 == 1:
                        self.create_pair_same_row_odd(i, p1, p2)
                        break
                    # 違う側で距離が奇数
                    if self.is_pair(p1[0], p2[1]) and (p1[0] - p2[0]) % 2 == 1:
                        self.create_pair_opposite_row_odd(i, p1, p2)
                        break
            self.update_unsolved_pair(i)
            if before_state == self.state:
                column_list = [pair[0] for pair in self.unsolved_pair_list[i]]
                column_list.sort()
                for index in range(len(column_list)):
                    if (column_list[(index + 1) % len(column_list)] - column_list[index]) % (self.column * 2) != 1:
                        move_key = f"swap/{i}/{(column_list[index] + 1) % (self.column * 2)}"
                        print(move_key)
                        self.move(move_key)
                        break

            self.update_unsolved_pair(i)
            before_state = self.state

            time.sleep(0.1)

    def solve_row_flip(self, i):
        before_state = self.state
        self.update_flip()

        while len(self.flip_list[i]) > 0:
            self.print()
            print(self.flip_list[i])
            for (index1, index2) in itertools.product(range(len(self.flip_list[i])), repeat=2):
                if index1 == index2:
                    continue
                p1 = self.flip_list[i][index1]
                p2 = self.flip_list[i][index2]

                if (p1[0] - p2[0]) % self.column == 0:
                    move_key = f"swapd/{i}/{(p1[0] + 1) % (self.column)}"
                    print(move_key)
                    self.move(move_key)
                    break

            if before_state == self.state:
                nearest = -1
                nearest_pair = None
                for (index1, index2) in itertools.product(range(len(self.flip_list[i])), repeat=2):
                    if index1 == index2:
                        continue
                    p1 = self.flip_list[i][index1]
                    p2 = self.flip_list[i][index2]

                    dis = min((p1[0] - p2[0]) % (self.column * 2), (p2[0] - p1[0]) % (self.column * 2))
                    if dis > nearest:
                        nearest = dis
                        nearest_pair = (index1, index2)

                p1 = self.flip_list[i][nearest_pair[0]]
                p2 = self.flip_list[i][nearest_pair[1]]
                print(nearest, p1, p2, nearest_pair)
                if nearest > self.column // 2:
                    if (p1[0] - p2[0]) % (self.column * 2) < (p2[0] - p1[0]) % (self.column * 2):
                        move_key = f"swap/{i}/{(p1[0] + 1) % (self.column * 2)}/{self.column - nearest}"
                        print(move_key)
                        self.move(move_key)
                    else:
                        move_key = f"swap/{i}/{(p2[0] + 1) % (self.column * 2)}/{self.column - nearest}"
                        print(move_key)
                        self.move(move_key)
                else:
                    if (p1[0] - p2[0]) % (self.column * 2) < (p2[0] - p1[0]) % (self.column * 2):
                        move_key = f"swap/{i}/{(p1[0] + 1) % (self.column * 2)}/{nearest}"
                        print(move_key)
                        self.move(move_key)
                    else:
                        move_key = f"swap/{i}/{(p2[0] + 1) % (self.column * 2)}/{nearest}"
                        print(move_key)
                        self.move(move_key)

            before_state = self.state
            self.update_flip()

            time.sleep(0.1)

    def fix_order(self, i):
        move_list = []
        for index in range(self.column * 2):
            for k in range(1, min(self.column + 1, 10)):
                move_list.append(f"swap/{i}/{index}/{k}")
        while self.calc_order_loss(i) > 0:
            best_key = None
            best_loss = self.calc_order_loss(i)
            print(best_loss)
            self.save(f"order{i}")
            for num in range(1, 3):
                for keys in itertools.product(move_list, repeat=num):
                    for key in keys:
                        self.move(key)
                    loss = self.calc_order_loss(i)
                    if loss < best_loss:
                        best_loss = loss
                        best_key = keys
                    # print(keys, loss)
                    self.load(f"order{i}")

            print(best_key)
            for key in best_key:
                self.move(key)
            self.print()

    def calc_order_loss(self, i):
        loss = 0
        for index in range(i * self.column * 2, (i + 1) * self.column * 2):
            state_i = self.encode(self.state[index])
            # print(loss, state_i, index1)
            if self.has_each_color:
                # 正しい位置はindex1
                index1 = index
                diff = abs(state_i - index1)
                if diff > 0:
                    loss += abs(state_i - index1) ** 2
            else:
                # 正しい位置はstate_iからstate_i+self.same_num
                index1 = index % (self.column * 2)
                if self.row % 2 == 1:
                    state_i = state_i // 2 * self.same_num
                else:
                    state_i = state_i // 3 * self.same_num
                if state_i <= index1 < state_i + self.same_num:
                    continue
                loss_list = [abs(state_i - index1), abs(index1 - state_i),
                             abs((state_i + self.same_num - 1) - index1),
                             abs(index1 - (state_i + self.same_num - 1))]
                if min(loss_list) > 0:
                    loss += min(loss_list) + 1

        return loss

    def create_pair_same_row(self, i, p1, p2):
        print((p1[0], p2[0]))
        if p1[0] > p2[0]:
            if p1[0] - p2[0] > self.column:
                s1 = p1[0]
                s2 = p2[0]
            else:
                s1 = p2[0]
                s2 = p1[0]
        else:
            if p2[0] - p1[0] > self.column:
                s1 = p2[0]
                s2 = p1[0]
            else:
                s1 = p1[0]
                s2 = p2[0]
        diff = (s2 - s1) % self.column
        if 0 < diff <= self.column // 2:
            move_key = f"f{s2 % (self.column * 2)}.f{(s1 + (self.column + diff) // 2) % (self.column * 2)}"
            print(move_key)
            self.move(move_key)
        else:
            if diff != 0:
                move_key = f"f{(s1 + self.column // 2) % (self.column * 2)}.f{(s1 + self.column - diff // 2) % (self.column * 2)}"
                print(move_key)
                self.move(move_key)
        move_key = f"get_pair/{i}/{(s1 + 1) % (self.column * 2)}"
        print(move_key)
        self.move(move_key)

    def create_pair_opposite_row(self, i, p1, p2):
        print((p1[0], p2[1]))
        diff = (p2[0] - p1[0]) % (self.column * 2)
        if diff < self.column:
            move_key = f"f{(p1[0] + (diff - 1) // 2 + 1) % (self.column * 2)}"
            print(move_key)
            self.move(move_key)
        else:
            move_key = f"f{(p1[0] + self.column - (self.column * 2 - diff - 1) // 2) % (self.column * 2)}"
            print(move_key)
            self.move(move_key)
        move_key = f"get_pair/{i}/{(p1[0] + 1) % (self.column)}"
        print(move_key)
        self.move(move_key)

    def create_pair_same_row_odd(self, i, p1, p2):
        print((p1[0], p2[0]))
        if p1[0] > p2[0]:
            if p1[0] - p2[0] > self.column:
                s1 = p1[0]
                s2 = p2[0] + self.column * 2
            else:
                s1 = p2[0]
                s2 = p1[0]
        else:
            if p2[0] - p1[0] > self.column:
                s1 = p2[0]
                s2 = p1[0] + self.column * 2
            else:
                s1 = p1[0]
                s2 = p2[0]
        diff = (s2 - s1) % self.column
        if diff != 1:
            move_key = f"f{((s1 + self.column + s2) // 2 - self.column // 2 - 1) % (self.column * 2)}.f{(s1 + self.column - 1 - self.column // 2) % (self.column * 2)}"
            print(move_key)
            self.move(move_key)
        else:
            move_key = f"f{s2 % (self.column * 2)}.f{(s1 + self.column - 1 - self.column // 2) % (self.column * 2)}.f{(s1 + self.column - 2 - self.column // 2) % (self.column * 2)}.f{(s1 + self.column - 1 - self.column // 2) % (self.column * 2)}"
            print(move_key)
            self.move(move_key)
        move_key = f"get_pair/{i}/{(s1 + 1) % (self.column * 2)}"
        print(move_key)
        self.move(move_key)

    def create_pair_opposite_row_odd(self, i, p1, p2):
        print((p1[0], p2[1]))
        if p1[0] > p2[0]:
            if p1[0] - p2[0] > self.column:
                s1 = p1[0]
                s2 = p2[0] + self.column * 2
            else:
                s1 = p2[0]
                s2 = p1[0]
        else:
            if p2[0] - p1[0] > self.column:
                s1 = p2[0]
                s2 = p1[0] + self.column * 2
            else:
                s1 = p1[0]
                s2 = p2[0]
        diff = (s2 - s1) % self.column
        if diff != 0:
            move_key = f"f{((s1 + self.column + s2) // 2 - self.column // 2) % (self.column * 2)}"
            print(move_key)
            self.move(move_key)
        else:
            move_key = f"f{(s2 - self.column // 2 - 1) % (self.column * 2)}.f{(s2 - self.column // 2 - 2) % (self.column * 2)}.f{(s2 - self.column // 2 - 1) % (self.column * 2)}"
            print(move_key)
            self.move(move_key)
        move_key = f"get_pair/{i}/{(s1 + 1) % (self.column * 2)}"
        print(move_key)
        self.move(move_key)

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
                sugar_dict[f"swapd/{i}/{j}"] = f"-r{i}.f{j}.r{i}.-r{self.row - i}.f{j}.r{self.row - i}"
            for j in range(self.column * 2):
                sugar_dict[f"swap/{i}/{j}"] = f"f{j}.-r{i}.f{j}.r{i}.-r{self.row - i}.f{j}.r{self.row - i}.f{j}"
                for k in range(1, self.column + 1):
                    sugar_dict[
                        f"swap/{i}/{j}/{k}"] = f"f{j}.{'.'.join([f'-r{i}' for _ in range(k)])}.f{j}.{'.'.join([f'r{i}' for _ in range(k)])}.{'.'.join([f'-r{self.row - i}' for _ in range(k)])}.f{j}.{'.'.join([f'r{self.row - i}' for _ in range(k)])}.f{j}"
                sugar_dict[f"get_pair/{i}/{j}"] = f"-r{self.row - i}.f{j}.r{self.row - i}"
        for i in range((self.row + 1) // 2):
            sugar_dict[f"flip/{i}"] = f"-r{i}.f0.r{self.row - i}.f0.-r{self.row - i}"

        return sugar_dict

    def encode(self, cell):
        if self.has_each_color:
            return int(cell[1:])
        else:
            return self.alphabet.index(cell)

    def update_unsolved_pair(self, i):
        self.unsolved_pair_list[i].clear()
        for j in range(self.column * 2):
            upper_i = i * self.column * 2 + j
            lower_i = (self.row - i) * self.column * 2 + j
            if not self.is_pair(upper_i, lower_i):
                self.unsolved_pair_list[i].append((upper_i, lower_i))

    def update_flip(self):
        for i in range((self.row + 1) // 2):
            self.flip_list[i] = self.get_flip_list(i)

    def is_pair(self, i, j):
        cell_1 = self.encode(self.state[i])
        cell_2 = self.encode(self.state[j])
        if self.has_each_color and abs(cell_1 - cell_2) != abs(
                cell_1 // (self.column * 2) - cell_2 // (self.column * 2)) * self.column * 2:
            return False
        elif not self.has_each_color and self.row % 2 == 1 and (cell_1 // 2 != cell_2 // 2 or cell_1 == cell_2):
            return False
        elif not self.has_each_color and self.row % 2 == 0 and (cell_1 // 3 != cell_2 // 3 or cell_1 == cell_2):
            return False
        return True

    def is_even_flip(self, i):
        return len(self.flip_list[i]) % 2 == 0

    def get_flip_list(self, i):
        flip_list = []
        for j in range(self.column * 2):
            upper = self.encode(self.state[i * self.column * 2 + j])
            lower = self.encode(self.state[(self.row - i) * self.column * 2 + j])

            if lower < upper:
                flip_list.append((i * self.column * 2 + j, (self.row - i) * self.column * 2 + j))
        return flip_list

    def print(self):
        print(f"move_num: {len(self.move_list)}")
        print_globe(self.row, self.column, self.state)


def print_cube(size, state):
    cell_len = 2
    for i in range(size):
        padding = " " * (cell_len * size)
        print(padding, *state[i * size:(i + 1) * size])
    for i in range(size):
        row = []
        for j in range(3, 7):
            plane = j % 4 + 1
            row.append(' '.join(state[i * size + size ** 2 * plane:(i + 1) * size + size ** 2 * plane]))
        print('  '.join(row))
    for i in range(size):
        padding = " " * (cell_len * size)
        print(padding, *state[i * size + size ** 2 * 5:(i + 1) * size + size ** 2 * 5])


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
        line.append(f"{state[index]:2}")
    line.append(" " * 2)
    line.append(f"{state[cross[0]]:2}")
    line.append(" " * 2)
    for index in right_half_list[0]:
        line.append(f"{state[index]:2}")
    print(''.join(line))

    # cross
    for i in range(max(len(left_cross), len(right_cross))):
        line = []
        line.append(" " * 2 * len(left_half_list[1]))
        if i < len(right_cross):
            line.append(f"{state[right_cross[i]]:2}")
        else:
            line.append(" " * 2)
        line.append(" " * 2)
        if i < len(left_cross):
            line.append(f"{state[left_cross[i]]:2}")
        else:
            line.append(" " * 2)
        print(''.join(line))

    # leftの前半（逆順）, rightの後半（逆順）
    line = []
    if len(left_points) % 2 == 1:
        line.append(" " * 2)
    for index in left_half_list[0][::-1]:
        line.append(f"{state[index]:2}")
    line.append(" " * 2)
    line.append(f"{state[cross[1]]:2}")
    line.append(" " * 2)
    for index in right_half_list[1][::-1]:
        line.append(f"{state[index]:2}")
    print(''.join(line))

    print()
    pass


def print_globe(i, j, state):
    for row in range(i + 1):
        temp = []
        for index in range(row * j * 2, (row + 1) * j * 2):
            temp.append(f"{state[index]:2}")
        print(';'.join(temp))
    print()


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
