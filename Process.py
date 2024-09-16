from typing import Dict
from typing import List
import Lattice
from Lattice import Lattice
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes


def init_second_moments(second_moments, size):
    for r in range(size):
        second_moments[r] = np.zeros((size, size))


def calc_second_moments(matrix):
    second_moments = dict()
    size = matrix.shape[0]
    for r in range(size):
        second_moments[r] = np.zeros((size, size))
        for i in range(second_moments[r].shape[0]):
            for j in range(second_moments[r].shape[1]):
                second_moments[r][i][j] = matrix[i][j] * matrix[i][(j + r) % size]
    return second_moments


class Process:
    _time: int
    _num_steps_to_steady_state: int
    _num_steps_in_steady_state: int
    _lattice: Lattice
    _population_first_moments: np.ndarray
    _current_population_first_moments: np.ndarray
    _magnetization_first_moments: np.ndarray
    _current_magnetization_first_moments: np.ndarray
    _second_moments: Dict[int, np.ndarray]
    _current_second_moments: Dict[int, np.ndarray]
    _population_initial_condition: np.ndarray
    _magnetization_initial_condition: np.ndarray
    _mag_history: List[float]
    _heatmap: Axes
    _fig: Figure
    _axes: Axes

    def __init__(self, size=10, temperature=1.0, diffusion=0.5, epsilon=0.5, num_steps_to_steady_state=200000,
                 num_steps_in_steady_state=100000, seed=None):
        self.reset_time()
        self._num_steps_to_steady_state = num_steps_to_steady_state
        self._num_steps_in_steady_state = num_steps_in_steady_state
        self._lattice = Lattice(size=size, temperature=temperature, diffusion=diffusion, epsilon=epsilon, seed=seed)
        self._population_first_moments = np.zeros((size, size))
        self._current_population_first_moments = np.zeros((size, size))
        self._magnetization_first_moments = np.zeros((size, size))
        self._current_magnetization_first_moments = np.zeros((size, size))
        self._population_initial_condition = np.zeros((size, size))
        self._magnetization_initial_condition = np.zeros((size, size))
        self._mag_history = []
        '''
        self._second_moments = dict()
        init_second_moments(self._second_moments, size)
        self._current_second_moments = dict()
        init_second_moments(self._second_moments, size)
        '''
    def increment_time(self):
        self._time = self._time + 1

    def set_seed(self,seed):
        self._lattice.set_seed(seed)
    def get_mag_history(self):
        return self._mag_history

    def get_time(self):
        return self._time

    def reset_time(self):
        self._time = 0

    def start_process(self):
        self.reset_time()
        self._population_first_moments = np.copy(self._current_population_first_moments)
        self._magnetization_first_moments = np.copy(self._current_magnetization_first_moments)

    def stop_process(self, visualize=True):
        if visualize:
            return self.display_first_moments()
        return self._population_first_moments / self._num_steps_to_steady_state, self._magnetization_first_moments / self._num_steps_to_steady_state

    def set_size(self, size):
        self._lattice = Lattice(size=size, temperature=self.get_temperature(), diffusion=self.get_diffusion(),
                                epsilon=self.get_epsilon())

    def set_temperature(self, temperature):
        self._lattice.set_temperature(temperature)

    def get_temperature(self):
        return self._lattice.get_temperature()

    def get_diffusion(self):
        return self._lattice.get_diffusion()

    def get_epsilon(self):
        return self._lattice.get_epsilon()

    def set_number_of_steps_until_steady_state(self, n):
        self._num_steps_to_steady_state = n

    def set_number_of_steps_in_steady_state(self, n):
        self._num_steps_in_steady_state = n

    def run_until_steady_state(self):
        for _ in range(self._num_steps_to_steady_state):
            result = self._lattice.algorithm_step()
            self._mag_history.append(self._lattice.get_mag_per_site())
            self.calc_changes(result)

    def calculate_steady_state(self, visualize=True):
        self.start_process()
        for self._time in range(self._num_steps_in_steady_state - 1):
            result = self._lattice.algorithm_step()
            self._mag_history.append(self._lattice.get_mag_per_site())
            self.calculate_moments(result)
        return self.stop_process(visualize)

    def calculate_moments(self, result):
        self.calc_changes(result)
        # self.fast_calc_second_moments(self._current_first_moments, locations[0])
        # self.fast_calc_second_moments(self._current_first_moments, locations[1])

        self._population_first_moments = np.array(self._current_population_first_moments) + self._population_first_moments
        self._magnetization_first_moments = np.array(self._current_magnetization_first_moments) + self._magnetization_first_moments
        # for r in range(self._lattice.get_size()):
        #    self._second_moments[r] = self._current_second_moments[r] + self._second_moments[r]

    def calc_changes(self, result):
        if result is not None:
            if result[0] == 'spin' and result[1] is not None:
                self.fast_calc_current_magnetization_first_moments(result[2], result[1])
            elif result[0] == 'move' and result[1] is not None:
                self.fast_calc_current_population_first_moments(result[2], result[1])

    def fast_calc_current_population_first_moments(self, location_removed, location_added):
        self._current_population_first_moments[location_removed[0]][location_removed[1]] -= 1
        self._current_population_first_moments[location_added[0]][location_added[1]] += 1

    def fast_calc_current_magnetization_first_moments(self, location, magnetization):
        self._current_magnetization_first_moments[location[0]][location[1]] = magnetization

    def populate_lattice_randomly(self, particle_count=1000):
        self._lattice.clear_lattice()
        self.reset_statistics()
        population_matrix, magnetization_matrix = self._lattice.populate_lattice_randomly(particle_count=particle_count)
        self._population_first_moments = np.copy(population_matrix)
        self._current_population_first_moments = np.copy(population_matrix)
        self._magnetization_first_moments = np.copy(magnetization_matrix)
        self._current_magnetization_first_moments = np.copy(magnetization_matrix)
        # self._second_moments = calc_second_moments(np.copy(matrix))
        # self._current_second_moments = calc_second_moments(np.copy(matrix))
        self._population_initial_condition = np.copy(population_matrix)
        self._magnetization_initial_condition = np.copy(magnetization_matrix)

    def populate_lattice_left_right_polarized(self, particle_count=1000):
        size = self._lattice.get_size()
        self._lattice.clear_lattice()
        self.reset_statistics()
        population_matrix, magnetization_matrix = self._lattice.populate_left_right_polarized(particle_count)
        self._population_first_moments = np.copy(population_matrix)
        self._current_population_first_moments = np.copy(population_matrix)
        self._magnetization_first_moments = np.copy(magnetization_matrix)
        self._current_magnetization_first_moments = np.copy(magnetization_matrix)
        # self._second_moments = calc_second_moments(np.copy(matrix))
        # self._current_second_moments = calc_second_moments(np.copy(matrix))
        self._population_initial_condition = np.copy(population_matrix)
        self._magnetization_initial_condition = np.copy(magnetization_matrix)

    def populate_lattice_uniformly_randomly_polarized(self, particle_count=1000):
        size = self._lattice.get_size()
        self._lattice.clear_lattice()
        self.reset_statistics()
        population_matrix, magnetization_matrix = self._lattice.populate_uniformly_randomly_polarized(particle_count)
        self._population_first_moments = np.copy(population_matrix)
        self._current_population_first_moments = np.copy(population_matrix)
        self._magnetization_first_moments = np.copy(magnetization_matrix)
        self._current_magnetization_first_moments = np.copy(magnetization_matrix)
        # self._second_moments = calc_second_moments(np.copy(matrix))
        # self._current_second_moments = calc_second_moments(np.copy(matrix))
        self._population_initial_condition = np.copy(population_matrix)
        self._magnetization_initial_condition = np.copy(magnetization_matrix)

    def display_first_moments(self, is_initial=False):
        self._fig, self._axes = plt.subplots(1, 2, figsize=(14, 6))
        temp = self._lattice.get_temperature()
        density = self._lattice.get_particle_count() / ((self._lattice.get_size()) ** 2)
        if is_initial is True:
            self._fig.suptitle('Initial Conditions Temperature = {:.2f}, Density = {:.2f}'.format(temp, density), fontsize=16)

            self._axes[0].set_title('Population distribution')
            self._heatmap = sns.heatmap(data=self._population_initial_condition,
                                        cmap='gray',
                                        vmin=0.5 * density, vmax=2 * density, annot=True,
                                        cbar_kws={'label': 'Number of particles'}, ax=self._axes[0], fmt=".0f")

            self._axes[1].set_title('Magnetization distribution')
            self._heatmap = sns.heatmap(data=self._magnetization_initial_condition,
                                        cmap='coolwarm', annot=True, cbar_kws={'label': 'Magnetization'},
                                        ax=self._axes[1], fmt=".0f", vmin=-1 * density, vmax=1 * density)
            plt.show()
            return self._population_initial_condition, self._magnetization_initial_condition
        else:
            self._fig.suptitle('Temperature = {:.2f}, Density = {:.2f}'.format(temp, density), fontsize=16)

            self._axes[0].set_title('Population distribution')
            self._heatmap = sns.heatmap(data=self._population_first_moments / self._num_steps_to_steady_state, cmap='gray',
                                        vmin=0.5 * density, vmax=2 * density, annot=True,
                                        cbar_kws={'label': 'Number of particles'}, ax=self._axes[0], fmt=".0f")

            self._axes[1].set_title('Magnetization distribution')
            self._heatmap = sns.heatmap(data=self._magnetization_first_moments / self._num_steps_to_steady_state,
                                        cmap='coolwarm', annot=True, cbar_kws={'label': 'Magnetization'},
                                        ax=self._axes[1], fmt=".0f", vmin=-1 * density, vmax=1 * density)

            plt.show()
            return self._population_first_moments / self._num_steps_to_steady_state, self._magnetization_first_moments / self._num_steps_to_steady_state

    def reset_statistics(self):
        self._population_first_moments = np.zeros((self._lattice.get_size(), self._lattice.get_size()))
        self._current_population_first_moments = np.zeros((self._lattice.get_size(), self._lattice.get_size()))
        self._magnetization_first_moments = np.zeros((self._lattice.get_size(), self._lattice.get_size()))
        self._current_magnetization_first_moments = np.zeros((self._lattice.get_size(), self._lattice.get_size()))
        # init_second_moments(self._second_moments, self._lattice.get_size())
        # init_second_moments(self._current_second_moments, self._lattice.get_size())
        self._population_initial_condition = np.zeros((self._lattice.get_size(), self._lattice.get_size()))
        self._magnetization_initial_condition = np.zeros((self._lattice.get_size(), self._lattice.get_size()))
        self._mag_history=[]

    def get_population_first_moment(self):
        matrix = self._population_first_moments / self._num_steps_in_steady_state
        return np.sum(matrix) / (self._lattice.get_size() ** 2)

    def get_magnetization_first_moment(self):
        matrix = self._magnetization_first_moments / self._num_steps_in_steady_state
        return np.sum(matrix) / (self._lattice.get_size() ** 2)


'''
    def fast_calc_second_moments(self, matrix, loc):
        size = self._lattice.get_size()
        for r in range(size):
            self._current_second_moments[r][loc[0]][loc[1]] = matrix[loc[0]][loc[1]] * matrix[loc[0]][(loc[1] + r) % size]
'''

'''
    def get_second_moment(self, r):
        matrix = self._second_moments[r] / self._num_steps_in_steady_state
        return np.sum(matrix) / (self._lattice.get_size() ** 2)


    def populate_lattice_bottom_half(self):
        self._lattice.clear_lattice()
        self.reset_statistics()
        self._lattice.populate_bottom_half()
        matrix = np.zeros((self._lattice.get_size(), self._lattice.get_size()))
        matrix[self._lattice.get_size() // 2:, :] = 1
        self._first_moments = np.copy(matrix)
        self._current_first_moments = np.copy(matrix)
        self._second_moments = calc_second_moments(np.copy(matrix))
        self._current_second_moments = calc_second_moments(np.copy(matrix))
        self._initial_condition = np.copy(matrix)

    def populate_lattice_left_half(self):
        self._lattice.clear_lattice()
        self.reset_statistics()
        self._lattice.populate_left_half()
        matrix = np.zeros((self._lattice.get_size(), self._lattice.get_size()))
        matrix[:, : self._lattice.get_size() // 2] = 1
        self._first_moments = np.copy(matrix)
        self._current_first_moments = np.copy(matrix)
        self._second_moments = calc_second_moments(np.copy(matrix))
        self._current_second_moments = calc_second_moments(np.copy(matrix))
        self._initial_condition = np.copy(matrix)

    def populate_lattice_uniformly(self):
        self._lattice.clear_lattice()
        self.reset_statistics()
        self._lattice.populate_lattice_uniformly()
        matrix = np.zeros((self._lattice.get_size(), self._lattice.get_size()))
        matrix[::2, ::2] = 1
        matrix[1::2, 1::2] = 1
        self._first_moments = np.copy(matrix)
        self._current_first_moments = np.copy(matrix)
        self._second_moments = calc_second_moments(np.copy(matrix))
        self._current_second_moments = calc_second_moments(np.copy(matrix))
        self._initial_condition = np.copy(matrix)
'''