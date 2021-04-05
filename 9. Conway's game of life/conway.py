import os
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

from scipy.sparse import rand

class Conway:

    def __init__(self, field):
        self.alive_cells = field
        self.field_3d = self.__create_field_3d()
        self.alive_cells_array = []

    def __create_field_3d(self):
        alive_cells = self.alive_cells
        dead_cells = abs(self.alive_cells - 1)
        field_3d = np.array([alive_cells, dead_cells])
        return field_3d

    def _step(self):
        kernel = np.pad(np.zeros(1)[:, np.newaxis], 1, constant_values=1)

        rows, cols = self.alive_cells.shape
        field_3d_current = np.zeros((2, rows, cols))

        for current_c in range(2):
            field_3d_current[current_c] = signal.convolve2d(self.field_3d[current_c], kernel, mode='same', boundary='wrap')

        field_3d_current *= self.field_3d

        alive_new = self._update_alive(field_3d_current[0])
        dead_new = self._update_dead(field_3d_current[1])

        self.alive_cells = alive_new + dead_new
        self.field_3d = np.array([self.alive_cells, abs(self.alive_cells - 1)])

    def _update_alive(self, alive):
        alive_update = np.zeros_like(alive)
        alive_update[(alive == 2) | (alive == 3)] = 1

        return alive_update

    def _update_dead(self, dead):
        dead_update = np.zeros_like(dead)
        dead_update[dead == 5] = 1

        return dead_update

    def simulate(self, num_of_iteration, save_img=False, name_folder=None):
        # plt.figure(figsize=(14,11))
        # plt.title(f'{self.r * 100}% of agents surrounds')
        # plt.imshow(self.alive_cells, cmap=plt.cm.gray)
        self.alive_cells_array.append(self.alive_cells.sum())
        self._step()

        if save_img:
            path = "/content/drive/MyDrive/Sk/HPPL/9. Conway's game of life"
            path = os.path.join(path, name_folder)
            if not os.path.exists(path):
                os.makedirs(path)

            fname = os.path.join(path, f'{num_of_iteration}.png')
            plt.savefig(fname)
            plt.close()

class ConwayParallel:

    def __init__(self, comm, field):
        self.comm = comm
        self.size = comm.Get_size()
        self.rank = comm.Get_rank()
        self.root = 0

        self.left_neighbour = (self.rank - 1) % self.size
        self.right_neighbour = (self.rank + 1) % self.size

        self.alive_cells = field
        self.field_3d = self.__create_field_3d()
        self.alive_cells_array = []

    def __create_field_3d(self):
        alive_cells = self.alive_cells
        dead_cells = abs(self.alive_cells - 1)
        field_3d = np.array([alive_cells, dead_cells])
        return field_3d

    def _step(self, col_from_left, col_from_right):

        grid_with_neighbours = np.concatenate([np.expand_dims(col_from_left, 1),
                                               self.alive_cells,
                                               np.expand_dims(col_from_right, 1)], axis=1)

        grid_with_neighbours = np.concatenate([grid_with_neighbours[-2:-1, :],
                                               grid_with_neighbours,
                                               grid_with_neighbours[0:1, :]], axis=0).astype(int)

        field_3d = np.array([grid_with_neighbours, abs(grid_with_neighbours - 1)])

        kernel = np.pad(np.zeros(1)[:, np.newaxis], 1, constant_values=1)

        rows, cols = self.alive_cells.shape
        field_3d_current = np.zeros((2, rows, cols))

        for current_c in range(2):
            field_3d_current[current_c] = signal.convolve2d(field_3d[current_c], kernel, mode='same', boundary='wrap')

        field_3d_current *= field_3d

        alive_new = self._update_alive(field_3d_current[0])
        dead_new = self._update_dead(field_3d_current[1])

        self.alive_cells = alive_new + dead_new
        self.field_3d = np.array([self.alive_cells, abs(self.alive_cells - 1)])

    def _update_alive(self, alive):
        alive_update = np.zeros_like(alive)
        alive_update[(alive == 2) | (alive == 3)] = 1

        return alive_update

    def _update_dead(self, dead):
        dead_update = np.zeros_like(dead)
        dead_update[dead == 5] = 1

        return dead_update

    def simulate(self, num_of_iteration=None, save_img=False, name_folder=None):
        # plt.figure(figsize=(14,11))
        # plt.title(f'{self.r * 100}% of agents surrounds')
        # plt.imshow(self.alive_cells, cmap=plt.cm.gray)

        left_col = np.ascontiguousarray(self.alive_cells[:, 0])
        right_col = np.ascontiguousarray(self.alive_cells[:, -1])

        col_from_left = np.empty(left_col.shape, dtype=int)
        col_from_right = np.empty(right_col.shape, dtype=int)

        self.comm.Isend([left_col, MPI.INT], dest=self.left_neighbour)
        self.comm.Irecv([col_from_right, MPI.INT], source=self.right_neighbour).wait()
        self.comm.Isend([right_col, MPI.INT], dest=self.right_neighbour)
        self.comm.Irecv([col_from_left, MPI.INT], source=self.left_neighbour).wait()

        self.alive_cells_array.append(self.alive_cells.sum())
        self._step(col_from_left, col_from_right)

        if save_img:
            grid_buffer = None if self.root != self.rank else np.empty((self.size, 100, 100), dtype=np.int)
            self.comm.Gather(self.alive_cells, grid_buffer, root=self.root)

            path = "/content/drive/MyDrive/Sk/HPPL/9. Conway's game of life"
            path = os.path.join(path, name_folder)
            if not os.path.exists(path):
                os.makedirs(path)

            fname = os.path.join(path, f'{num_of_iteration}.png')
            plt.savefig(fname)
            plt.close()

# if __name__ == "__main__":
#     comm = MPI.COMM_WORLD

#     X = rand(100, 100, density=0.2, format='csr')
#     X.data[:] = 1

#     X = X.todense()

#     c = ConwayParallel(comm, X)
#     # c.simulate()

#     print(c.size, c.rank)

#     for i in tqdm_notebook(range(500)):
#         c.simulate(i, save_img=False)

