from typing import Type, Tuple, List, Optional
from Particle import Particle
import numpy as np


class Node:
    _particles: List['Particle']
    _location: Tuple[int, int]
    _right: Optional[Type['Node']]
    _bottom: Optional[Type['Node']]
    _left: Optional[Type['Node']]
    _top: Optional[Type['Node']]
    _particle_count: int
    _magnetization: int
    _T: float

    def __init__(self, temp):
        self._particles = list()
        self._right = None
        self._bottom = None
        self._left = None
        self._top = None
        self._particle_count = 0
        self._magnetization = 0
        self._T = temp

    def __eq__(self, other):
        return self._location is not None and self.get_location() == other.get_location()

    def __ne__(self, other):
        return not self.__eq__(other)

    def set_location(self, location):
        self._location = location

    def get_location(self):
        return self._location

    def get_particle_count(self):
        return self._particle_count

    def set_right(self, node):
        self._right = node

    def get_right(self):
        return self._right

    def set_bottom(self, node):
        self._bottom = node

    def get_bottom(self):
        return self._bottom

    def set_left(self, node):
        self._left = node

    def get_left(self):
        return self._left

    def set_top(self, node):
        self._top = node

    def get_top(self):
        return self._top

    def set_temperature(self, temp):
        self._T = temp

    def get_neighbors(self):
        return [x for x in list([self.get_right(), self.get_bottom(), self.get_left(), self.get_top()])]

    def add_particle(self, particle):
        self._particles.append(particle)
        self._particle_count += 1
        self._magnetization += particle.get_spin()

    def remove_particle(self, id_num):
        item = next((x for x in self._particles if x.get_id() == id_num), None)
        if item is not None:
            self._particles.remove(item)
            self._particle_count -= 1
            self._magnetization -= item.get_spin()

    def change_spin(self, particle):
        if particle not in self._particles:
            return self._magnetization
        w = 1
        if particle.get_spin() * self._magnetization > 0:
            w = np.exp((-particle.get_spin() * self._magnetization) / (self._T * self._particle_count))
        if np.random.uniform(0, 1) < w:
            particle.set_spin(-particle.get_spin())
            self._magnetization += 2 * particle.get_spin()
        return self._magnetization

    def clear_node(self):
        self._particles.clear()
        self._particle_count = 0
        self._magnetization = 0
