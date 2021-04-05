import numpy as np
import matplotlib.pyplot as plt


class BifurcationMap:
    __FIGSIZE = (14, 11)

    def __init__(self, x0=0.5, n=5000, iterations=500, last_m_points=200):
        self.r = np.linspace(0, 4, n)
        self.iterations = iterations
        self.last_m_values = self.iterations - last_m_points
        self.x = np.repeat(x0, n)

    def __logistic_map(self, ax):
        logistic_map = lambda r, x: r * x * (1 - x)
        ax = self.__create_ax(ax, logistic_map)

        return ax

    def __create_ax(self, ax, mapping):
        x = self.x.copy()
        for iteration in range(self.iterations):
            x = mapping(self.r, x)

            if iteration >= self.last_m_values:
                ax.plot(self.r, x, ',m', alpha=0.3)

        return ax

    def plot_bifurcation_map(self, mapping='logistic'):
        fig, ax = plt.subplots(1, 1, figsize=self.__FIGSIZE)

        if mapping == 'logistic':
            ax = self.__logistic_map(ax)
        else:
            raise NameError(f"Mapping {mapping} doesn't exist")

        ax.set_xlabel('r', fontsize=20)
        ax.set_ylabel('x', fontsize=20)
        ax.tick_params(axis='both', labelsize=20)
        ax.set_title(f'Bifurcation map via {mapping} map', fontsize=20)