from typing import Tuple


class Particle:
    _idNumber: int
    _spin: int
    _location: Tuple[int, int]

    def __init__(self, id_number, location, spin=1):
        self._idNumber = id_number
        self._location = location
        self._spin = spin

    def set_spin(self, spin):
        self._spin = spin

    def get_spin(self):
        return self._spin

    def get_id(self):
        return self._idNumber

    def __eq__(self, other):
        return self._idNumber == other.get_id()

    def __ne__(self, other):
        return not self.__eq__(other)

    def set_location(self, location):
        self._location = location

    def get_location(self):
        return self._location

    def move(self, direction, size):
        if direction == 'left':
            self.set_location((self._location[0], (self._location[1] - 1) % size))
        elif direction == 'right':
            self.set_location((self._location[0], (self._location[1] + 1) % size))
        elif direction == 'top':
            self.set_location(((self._location[0] - 1) % size, self._location[1]))
        elif direction == 'bottom':
            self.set_location(((self._location[0] + 1) % size, self._location[1]))
        return self.get_location()