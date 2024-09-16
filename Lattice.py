from Node import Node
from Particle import Particle
import numpy as np
from typing import Dict, Tuple


class Lattice:
    _size: int
    _T: float
    _D: float
    _epsilon: float
    _node_dict: Dict[Tuple[int, int], 'Node']
    _particle_dict: Dict[int, 'Particle']
    _lattice: Node
    _particle_count: int
    _mag_per_site: float

    def __init__(self, size, temperature=1.0, diffusion=0.5, epsilon=0.5, seed=None):
        self._size = size
        self._T = temperature
        self._D = diffusion
        self._epsilon = epsilon
        self._node_dict = dict()
        self._particle_dict = dict()
        self._lattice = self._init_lattice()
        self._particle_count = 0
        self._mag_per_site = 0
        if seed is not None:
            np.random.seed(seed)

    def set_seed(self, seed):
        np.random.seed(seed)
    def get_mag_per_site(self):
        return self._mag_per_site
    def get_size(self):
        return self._size

    def get_diffusion(self):
        return self._D

    def get_epsilon(self):
        return self._epsilon

    def get_temperature(self):
        return self._T

    def set_temperature(self, temperature):
        self._T = temperature
        for node in list(self._node_dict.values()):
            node.set_temperature(temperature)

    def get_particle_count(self):
        return self._particle_count

    def _init_lattice(self):
        head_node = self._init_lattice_rec(True, 0, 0)
        return head_node

    def _init_lattice_rec(self, from_left, i, j, node=None):
        if j == self._size:
            if i == 0:
                current_node = self._node_dict[(0, 0)]
                current_node.set_left(node)
                return current_node
            current_node = Node(self._T)
            current_node.set_location((i, 0))
            self._node_dict[(i, 0)] = current_node
            current_node.set_left(node)
            return current_node

        if i == self._size:
            current_node = self._node_dict[(0, j)]
            current_node.set_top(node)
            return current_node

        if j == 0 and i != 0:
            current_node = self._node_dict[(i, j)]
            current_node.set_top(node)
            right_node = self._init_lattice_rec(True, i, j + 1, current_node)
            bottom_node = self._init_lattice_rec(False, i + 1, j, current_node)
            current_node.set_right(right_node)
            current_node.set_bottom(bottom_node)

        if (i, j) in self._node_dict and i+j != 0:
            current_node = self._node_dict[(i, j)]
            if from_left:
                current_node.set_left(node)
            else:
                current_node.set_top(node)
            return current_node

        else:
            current_node = Node(self._T)
            current_node.set_location((i, j))
            self._node_dict[(i, j)] = current_node

        if from_left:
            current_node.set_left(node)
            right_node = self._init_lattice_rec(True, i, j + 1, current_node)
            bottom_node = self._init_lattice_rec(False, i + 1, j, current_node)
            current_node.set_right(right_node)
            current_node.set_bottom(bottom_node)
            return current_node
        else:
            current_node.set_top(node)
            right_node = self._init_lattice_rec(True, i, j + 1, current_node)
            bottom_node = self._init_lattice_rec(False, i + 1, j, current_node)
            current_node.set_right(right_node)
            current_node.set_bottom(bottom_node)
            return current_node

    def populate_lattice_randomly(self, particle_count=1000):
        size = self.get_size()
        population_matrix = np.zeros((size, size))
        magnetization_matrix = np.zeros((size, size))
        self._particle_count = particle_count
        for i in range(particle_count):
            row = np.random.randint(0, self._size)
            col = np.random.randint(0, self._size)
            spin = np.random.choice(a=[1, -1])
            particle = Particle(i, (row, col), spin)
            self._node_dict[(row, col)].add_particle(particle)
            self._particle_dict[i] = particle
            population_matrix[row][col] += 1
            magnetization_matrix[row][col] += spin
            self._mag_per_site += spin
        self._mag_per_site /= (self._size ** 2)
        return population_matrix, magnetization_matrix

    def populate_left_right_polarized(self, particle_count=1000):
        size = self.get_size()
        population_matrix = np.zeros((size, size))
        magnetization_matrix = np.zeros((size, size))
        self._particle_count = particle_count
        particle_count_per_node = particle_count // (size ** 2)
        spin_per_site = 1 / (size ** 2)
        id = 1
        for row in range(size):
            for col in range(size):
                if col < size // 2:
                    for _ in range(particle_count_per_node):
                        neg_particle = Particle(id, (row, col), -1)
                        self._node_dict[(row, col)].add_particle(neg_particle)
                        self._particle_dict[id] = neg_particle
                        population_matrix[row][col] += 1
                        magnetization_matrix[row][col] += -1
                        self._mag_per_site -= spin_per_site
                        id += 1
                else:
                    for _ in range(particle_count_per_node):
                        pos_particle = Particle(id, (row, col), 1)
                        self._node_dict[(row, col)].add_particle(pos_particle)
                        self._particle_dict[id] = pos_particle
                        population_matrix[row][col] += 1
                        magnetization_matrix[row][col] += 1
                        self._mag_per_site += spin_per_site
                        id += 1
        if particle_count_per_node * (size ** 2) < particle_count:
            for i in range(particle_count - particle_count_per_node * (size ** 2)):
                ind = i % size
                if ind < size // 2:
                    neg_particle = Particle(id, (ind, ind), -1)
                    self._node_dict[(ind, ind)].add_particle(neg_particle)
                    self._particle_dict[id] = neg_particle
                    population_matrix[ind][ind] += 1
                    magnetization_matrix[ind][ind] += -1
                    self._mag_per_site -= spin_per_site
                    id += 1

                else:
                    pos_particle = Particle(id, (ind, ind), 1)
                    self._node_dict[(ind, ind)].add_particle(pos_particle)
                    self._particle_dict[id] = pos_particle
                    population_matrix[ind][ind] += 1
                    magnetization_matrix[ind][ind] += 1
                    self._mag_per_site += spin_per_site
                    id += 1
        return population_matrix, magnetization_matrix

    def populate_uniformly_randomly_polarized(self, particle_count=1000):
        size = self.get_size()
        population_matrix = np.zeros((size, size))
        magnetization_matrix = np.zeros((size, size))
        self._particle_count = particle_count
        particle_count_per_node = particle_count // (size ** 2)
        spin_per_site = 1 / (size ** 2)
        id = 1
        for row in range(size):
            for col in range(size):
                for _ in range(particle_count_per_node):
                    if np.random.random() <= 0.5:
                        particle = Particle(id, (row, col), -1)
                        magnetization_matrix[row][col] += -1
                        self._mag_per_site -= spin_per_site
                    else:
                        particle = Particle(id, (row, col), 1)
                        magnetization_matrix[row][col] += 1
                        self._mag_per_site += spin_per_site
                    self._node_dict[(row, col)].add_particle(particle)
                    self._particle_dict[id] = particle
                    population_matrix[row][col] += 1
                    id += 1
        return population_matrix, magnetization_matrix

    def clear_lattice(self):
        self._particle_count = 0
        self._mag_per_site = 0
        self._particle_dict.clear()
        [node.clear_node() for node in list(self._node_dict.values())]

    def algorithm_step(self):
        if self._particle_count == 0:
            self.populate_lattice_randomly()
        particle = np.random.choice(a=list(self._particle_dict.values()))
        old_location = particle.get_location()
        new_location = None
        p_spin = 0.5
        options = ['spin', 'left', 'right', 'top', 'bottom', 'stay']
        probabilities = [p_spin,
                         (self._D / 4) * (1 - p_spin) * (1 + particle.get_spin() * self.get_epsilon()),
                         (self._D / 4) * (1 - p_spin) * (1 - particle.get_spin() * self.get_epsilon()),
                         (self._D / 4) * (1 - p_spin),
                         (self._D / 4) * (1 - p_spin),
                         (1 - self._D) * (1 - p_spin)]
        action = np.random.choice(a=options, p=probabilities)
        if action == 'spin':
            magnetization = self._node_dict[particle.get_location()].change_spin(particle)
            self._mag_per_site += (2 * particle.get_spin()) / (self._size ** 2)
            return 'spin', magnetization, particle.get_location()
        elif action == 'left':
            self._node_dict[particle.get_location()].remove_particle(particle.get_id())
            new_location = particle.move('left', self.get_size())
            self._node_dict[particle.get_location()].add_particle(particle)
        elif action == 'right':
            self._node_dict[particle.get_location()].remove_particle(particle.get_id())
            new_location = particle.move('right', self.get_size())
            self._node_dict[particle.get_location()].add_particle(particle)
        elif action == 'top':
            self._node_dict[particle.get_location()].remove_particle(particle.get_id())
            new_location = particle.move('top', self.get_size())
            self._node_dict[particle.get_location()].add_particle(particle)
        elif action == 'bottom':
            self._node_dict[particle.get_location()].remove_particle(particle.get_id())
            new_location = particle.move('bottom', self.get_size())
            self._node_dict[particle.get_location()].add_particle(particle)
        return 'move', new_location, old_location


'''
    # align
    def populate_bottom_half(self):
        self._density = 0.5
        self._lattice_populated = True
        [node.toggle_occupation() for node in list(self._node_dict.values()) if node.get_location()[1] >= self._size/2]

    # align
    def populate_left_half(self):
        self._density = 0.5
        self._lattice_populated = True
        [node.toggle_occupation() for node in list(self._node_dict.values()) if node.get_location()[0] < self._size/2]

    # align
    def populate_lattice_uniformly(self):
        self._density = 0.5
        self._lattice_populated = True
        [node.toggle_occupation() for node in list(self._node_dict.values())
         if (node.get_location()[0] + node.get_location()[1]) % 2 == 0]
 '''