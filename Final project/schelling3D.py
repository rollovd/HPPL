import numpy as np
from scipy import signal

import glob
import os
from tqdm import tqdm_notebook
import imageio
import cv2

import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import seaborn as sns

params = {'legend.fontsize': 'x-large',
          'figure.figsize': (15, 5),
          'axes.labelsize': 'x-large',
          'axes.titlesize': 'x-large',
          'xtick.labelsize': 'x-large',
          'ytick.labelsize': 'x-large'
          }
pylab.rcParams.update(params)


class Schelling3D:

    def __init__(self, num_of_players: int, n: int, r: float, save_images: bool = False):
        self.c = num_of_players
        self.r = r
        self.n = n
        self.playing_field_all, self.playing_field = self._create_playing_field()
        self.want_to_move = []

        if save_images:
            path = '/content/drive/MyDrive/Sk/HPPL/Project'
            self.folder = os.path.join(path, f'{self.r}_{self.n}_percent_of_agents_folder')

            os.makedirs(self.folder, exist_ok=True)

    def _create_playing_field(self):
        agents_zero = round(self.n ** 3 / 2)
        agents_one = self.n ** 3 - agents_zero

        arr = np.hstack((np.zeros(agents_zero, dtype=np.int16), np.ones(agents_one, dtype=np.int16)))
        np.random.shuffle(arr)

        type1 = arr.reshape(self.n, self.n, self.n)
        type2 = abs(type1 - 1)
        playing_field = np.array([type1, type2])

        return playing_field, type2

    def _find_neighbours(self):
        kernel = np.ones((3, 3, 3))
        kernel[1][1][1] = 0

        neighbours_3d = np.zeros((self.c, self.n, self.n, self.n))

        for current_c in range(self.c):
            neighbours_3d[current_c] = signal.convolve(self.playing_field_all[current_c], kernel, mode='same',
                                                       method='direct')

        neighbours_3d *= self.playing_field_all
        neighbours = np.sum(neighbours_3d, axis=0)
        need_to_move = neighbours < int(kernel.sum() * self.r)

        self.want_to_move.append(need_to_move.sum())

        return need_to_move

    def _shuffle_agents(self, need_to_move):
        move_agents = self.playing_field[need_to_move]
        np.random.shuffle(move_agents)
        self.playing_field[need_to_move] = move_agents

    def _update_playing_field(self):
        type1 = abs(self.playing_field - 1)
        self.playing_field_all = np.array([type1, self.playing_field])

    def play(self):
        agents_need_to_move = self._find_neighbours()
        shuffle_field = self._shuffle_agents(agents_need_to_move)
        self._update_playing_field()

    def save_image(self, num_of_iteration=None, save_img=False):
        plt.figure(figsize=(8, 5))
        plt.title(f'{self.r * 100}% of agents surrounds')
        plt.imshow(self.playing_field, cmap=plt.cm.gray)

        if save_img:
            path = '/content/drive/My Drive/Schelling model'
            folder = os.path.join(path, f'{self.r}_{self.n}_percent_of_agents_folder')
            fname = os.path.join(folder, f'{num_of_iteration}.png')
            plt.savefig(fname)
            plt.close()

    def draw_projections(self, total_num_of_iterations=None, num_of_iteration=None, save_img=False):

        data = self.playing_field

        f, (ax1, ax2, ax3, axcb) = plt.subplots(1, 4,
                                                gridspec_kw={'width_ratios': [1, 1, 1, 0.08]})

        f.suptitle(f'Iteration {num_of_iteration}/{total_num_of_iterations}. Total number of agents {self.n ** 3}',
                   fontsize=16)

        f.set_figheight(7)
        f.set_figwidth(25)

        ax1.title.set_text('$X$')
        ax2.title.set_text('$Y$')
        ax3.title.set_text('$Z$')

        ax1.get_shared_y_axes().join(ax2, ax3)
        g1 = sns.heatmap(data.sum(axis=0), cmap="YlGnBu", cbar=False, ax=ax1)
        g1.set_ylabel('')
        g1.set_xlabel('')

        g2 = sns.heatmap(data.sum(axis=1), cmap="YlGnBu", cbar=False, ax=ax2)
        g2.set_ylabel('')
        g2.set_xlabel('')
        g2.set_yticks([])

        g3 = sns.heatmap(data.sum(axis=2), cmap="YlGnBu", ax=ax3, cbar_ax=axcb)
        g3.set_ylabel('')
        g3.set_xlabel('')
        g3.set_yticks([])

        # may be needed to rotate the ticklabels correctly:
        for ax in [g1, g2, g3]:
            tl = ax.get_xticklabels()
            ax.set_xticklabels(tl, rotation=90)
            tly = ax.get_yticklabels()
            ax.set_yticklabels(tly, rotation=0)

        if save_img:
            fold = os.path.join(self.folder, 'projections')

            os.makedirs(fold, exist_ok=True)

            fname = os.path.join(fold, f'{num_of_iteration}.png')
            plt.savefig(fname)
            plt.close()

    def plot_scatter(self, total_num_of_iterations=None, num_of_iteration=None, save_img=False):
        z, x, y = schelling3d.playing_field.nonzero()

        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')

        ax.set_title(f'Iteration {num_of_iteration}/{total_num_of_iterations}. Total number of agents {self.n ** 3}',
                     fontsize=16)
        ax.scatter(x, y, -z, zdir='z', c='red')

        ax.set_xlabel('$X$', fontsize=18)
        ax.set_ylabel('$Y$', fontsize=18)
        ax.set_zlabel(r'$Z$', fontsize=14)

        if save_img:
            fold = os.path.join(self.folder, f'3d')

            os.makedirs(fold, exist_ok=True)

            fname = os.path.join(fold, f'{num_of_iteration}.png')
            plt.savefig(fname)
            plt.close()

    def save_all_images(self, total_num_of_iterations=None, num_of_iteration=None, save_img=False):
        self.plot_scatter(total_num_of_iterations=total_num_of_iterations, num_of_iteration=num_of_iteration,
                          save_img=True)
        self.draw_projections(total_num_of_iterations=total_num_of_iterations, num_of_iteration=num_of_iteration,
                              save_img=True)

    def create_gifs(self):

        def create_gif(filenames, end_file):
            images = []
            for filename in tqdm_notebook(filenames):
                images.append(imageio.imread(filename))

            imageio.mimsave(end_file, images)

        three_dim_folder = os.path.join(self.folder, '3d/*.png')
        projection_folder = os.path.join(self.folder, 'projections/*.png')

        end_three = os.path.join(self.folder, f'{self.r}_{self.n}_agents_3d.gif')
        end_projection = os.path.join(self.folder, f'{self.r}_{self.n}_agents_projection.gif')

        filenames_3d = glob.glob(three_dim_folder)
        filenames_proj = glob.glob(projection_folder)

        create_gif(filenames_3d, end_three)
        create_gif(filenames_proj, end_projection)

        # Create reader object for the gif
        gif1 = imageio.get_reader(end_three)
        gif2 = imageio.get_reader(end_projection)

        # If they don't have the same number of frame take the shorter
        number_of_frames = min(gif1.get_length(), gif2.get_length())

        # Create writer object
        new_gif = imageio.get_writer(os.path.join(self.folder, f'combine{self.r}_{self.n}.gif'))

        for frame_number in tqdm_notebook(range(number_of_frames)):
            img1 = gif1.get_next_data()
            img2 = gif2.get_next_data()

            img1 = cv2.resize(img1, (864, 504))
            # here is the magic
            new_image = np.hstack((img1, img2))
            new_gif.append_data(new_image)

        gif1.close()
        gif2.close()
        new_gif.close()