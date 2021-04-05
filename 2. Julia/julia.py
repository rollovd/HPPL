import numpy as np
import matplotlib.pyplot as plt

def julia_set(c, x_min=-2, x_max=2, y_min=-2, y_max=2,
              pixels_per_unit=200, num_iterations=1000, limit_value=2):
    num_of_units = x_max - x_min
    points = pixels_per_unit * num_of_units

    x_axis = np.linspace(x_min, x_max, points)
    y_axis = np.linspace(y_max, y_min, points)

    complex_points_matrix = np.tile(x_axis, (points, 1)) + np.transpose([1j * y_axis] * points)
    mask = np.full((points, points), True, dtype=bool)
    julia = np.zeros((points, points))

    for _ in range(num_iterations):
        complex_points_matrix[mask] = complex_points_matrix[mask] ** 2 + c
        julia[mask] += 1

        mask_check = abs(complex_points_matrix) > limit_value
        mask[mask_check] = False

    return julia, c


def plot_julia(julia, c, index=None, title=True, save_img=False, color=True):
    plt.figure(figsize=(14, 11))
    plt.imshow(julia, cmap=None if color else plt.cm.gray, extent=(-2, 2, -2, 2))

    plt.tick_params(axis='both', labelsize=20)

    if title:
        plt.title(f'Real part = {c.real}, imaginary part = {c.imag}', fontsize=16)

    plt.xlabel('Re(Z)', fontsize=20)
    plt.ylabel('Im(Z)', fontsize=20)

    if save_img:
        fname = f'/content/drive/My Drive/Julia/images_30_fps/{index}.png'
        plt.savefig(fname)
        plt.close()