import numpy as np
from numbers import Real
from math import floor, ceil, trunc


class Angle(Real):
    """Angle class.

    Angle handles modulo arithmetic for adding and subtracting angles
    """

    @staticmethod
    def mod_angle(value):
        return float(value)

    @property
    def degrees(self):
        return self.rad2deg()

    def __init__(self, value):
        self._value = self.mod_angle(value)

    def __hash__(self):
        return hash(self._value)

    def __add__(self, other):
        if isinstance(other, Angle):
            other = other._value
        out = self._value + other
        return self.__class__(out)

    def __radd__(self, other):
        return self.__class__.__add__(self, other)

    def __sub__(self, other):
        if isinstance(other, Angle):
            other = other._value
        out = self._value - other
        return self.__class__(out)

    def __rsub__(self, other):
        return self.__class__.__add__(-self, other)

    def __float__(self):
        return float(self._value)

    def __mul__(self, other):
        if isinstance(other, Angle):
            other = other._value
        return self._value * other

    def __rmul__(self, other):
        return self._value * other

    def __str__(self):
        return str(self._value)

    def __repr__(self):
        return "{0}({1!r})".format(self.__class__.__name__, float(self))

    def __neg__(self):
        return self.__class__(-self._value)

    def __truediv__(self, other):
        if isinstance(other, Angle):
            other = other._value
        return self._value / other

    def __rtruediv__(self, other):
        return other / self._value

    def __eq__(self, other):
        return self._value == other

    def __ne__(self, other):
        return self._value != other

    def __abs__(self):
        abs_val = self.__class__(abs(self._value))
        if abs_val._value < 0:
            abs_val._value = abs(abs_val._value)
        return abs_val

    def __le__(self, other):
        return self._value <= other

    def __lt__(self, other):
        return self._value < other

    def __ge__(self, other):
        return self._value >= other

    def __gt__(self, other):
        return self._value > other

    def __floor__(self):
        return floor(self._value)

    def __ceil__(self):
        return ceil(self._value)

    def __floordiv__(self, other):
        if isinstance(other, Angle):
            other = other._value
        return self._value // other

    def __mod__(self, other):
        return self._value % other

    def __pos__(self):
        return self.__class__(+self._value)

    def __pow__(self, value):
        return pow(self._value, value)

    def __rfloordiv__(self, other):
        return other // self._value

    def __rmod__(self, other):
        return other % self._value

    def __round__(self, ndigits=None):
        return round(self._value, ndigits=ndigits)

    def __rpow__(self, base):
        return NotImplemented

    def __trunc__(self):
        return trunc(self._value)

    def cos(self):
        return np.cos(self._value)

    def sin(self):
        return np.sin(self._value)

    def tan(self):
        return np.tan(self._value)

    def cosh(self):
        return np.cosh(self._value)

    def sinh(self):
        return np.sinh(self._value)

    def tanh(self):
        return np.tanh(self._value)

    def rad2deg(self):
        return np.rad2deg(self._value)
    
    @classmethod
    def average(cls, angles, weights=None):
        """
        Calculates the circular mean for a list of angles.
        
        Parameters
        ----------
        angles : list of :class `~.Angle`
            List of angles to calculate the circular mean of.
        weights : list of float, optional
            List of weights for each angle. Default is None.

        Returns
        -------
        :class `~.Angle`
            The circular mean of the angles.
        """
        if weights is None:
            weight_sum = 1
            weights = 1
        else:
            weight_sum = np.sum(weights)

        result = np.arctan2(
            float(np.sum(np.sin(angles) * weights) / weight_sum),
            float(np.sum(np.cos(angles) * weights) / weight_sum)
        )

        return cls(result)


class Bearing(Angle):
    """Bearing angle class.

    Bearing handles modulo arithmetic for adding and subtracting angles. \
    The return type for addition and subtraction is Bearing.
    Multiplication or division produces a float object rather than Bearing.
    """

    @staticmethod
    def mod_angle(value):
        return (value + np.pi) % (2.0 * np.pi) - np.pi
