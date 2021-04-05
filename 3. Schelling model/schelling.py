import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import os

def plot_want_to_move(schellings, n):
    plt.figure(figsize=(14, 11))
    plt.title('Number of households that want to move', fontsize=20)

    for instance in schellings:
        r = instance.r
        want_to_move = instance.want_to_move
        plt.plot(np.linspace(1, 100, n), want_to_move, label=f'{int(r / 0.125)}/8 surrounds')
        plt.legend(fontsize=16)

        plt.tick_params(axis='both', labelsize=20)
        plt.xlabel('Number of iterations', fontsize=20)
        plt.ylabel('Number of people wants to move', fontsize=20)


class Schelling:

    def __init__(self, num_of_players: int, n: int, r: float):
        self.c = num_of_players
        self.r = r
        self.n = n
        self.playing_field_all, self.playing_field = self._create_playing_field()
        self.want_to_move = []

    def _create_playing_field(self):
        agents_zero = round(self.n ** 2 / 2)
        agents_one = self.n ** 2 - agents_zero

        arr = np.hstack((np.zeros(agents_zero, dtype=np.int16), np.ones(agents_one, dtype=np.int16)))
        np.random.shuffle(arr)

        type1 = arr.reshape(self.n, self.n)
        type2 = self._inverse_transform(type1)
        playing_field = np.array([type1, type2])

        return playing_field, type2

    @staticmethod
    def _inverse_transform(type_matrix):
        new_matrix = np.where((type_matrix == 0) | (type_matrix == 1), type_matrix ^ 1, type_matrix)
        return new_matrix

    def _find_neighbours(self):
        kernel = np.pad(np.zeros(1)[:, np.newaxis], 1, constant_values=1)
        neighbours_3d = np.zeros((self.c, self.n, self.n))

        for current_c in range(self.c):
            neighbours_3d[current_c] = signal.convolve2d(self.playing_field_all[current_c], kernel, mode='same',
                                                         boundary='wrap')

        neighbours_3d *= self.playing_field_all
        neighbours = np.sum(neighbours_3d, axis=0)
        need_to_move = neighbours < int(kernel.sum() * self.r)

        self.want_to_move.append(need_to_move.sum().sum())

        return need_to_move

    def _shuffle_agents(self, need_to_move):
        move_agents = self.playing_field[need_to_move]
        np.random.shuffle(move_agents)
        self.playing_field[need_to_move] = move_agents

    def _update_playing_field(self):
        type1 = self._inverse_transform(self.playing_field)
        self.playing_field_all = np.array([type1, self.playing_field])

    def play(self):
        agents_need_to_move = self._find_neighbours()
        self._shuffle_agents(agents_need_to_move)
        self._update_playing_field()

    def save_image(self, num_of_iteration=None, save_img=False):
        plt.figure(figsize=(14, 11))
        plt.title(f'{self.r * 100}% of agents surrounds')
        plt.imshow(self.playing_field, cmap=plt.cm.gray)

        if save_img:
            path = '/content/drive/My Drive/Schelling model'
            folder = os.path.join(path, f'{self.r}_percent_of_agents_folder')
            fname = os.path.join(folder, f'{num_of_iteration}.png')
            plt.savefig(fname)
            plt.close()