import numpy as np

class DualNumber:
    """Dual number implementation

    :param real: The real part of the dual number
    :type real: Union[int, float]
    :param dual: The dual part of the dual number, defaults to 1
    :type dual: Union[int, float], optional
    """

    def __init__(self, real, dual=1):
        self.real = real
        self.dual = dual

    def __add__(self, other):
        """Implements the addition of dual numbers

        :param other: A dual number or scalar
        :type other: Union[DualNumber, Union[int, float]]
        :return: The sum of self with other
        :rtype: DualNumber
        """
        if not isinstance(other, (int, float, DualNumber)):
            raise TypeError(f"Unsupported type `{type(other)}`")
        if isinstance(other, (int, float)):
            # scalar numbers
            return DualNumber(other + self.real, self.dual)
        else:
            # dual number
            return DualNumber(self.real + other.real, self.dual + other.dual)

    def __sub__(self, other):
        """Implements the subtraction of dual numbers

        :param other: A dual number or scalar
        :type other: Union[DualNumber, Union[int, float]]
        :return: The difference of self with other
        :rtype: DualNumber
        """
        if not isinstance(other, (int, float, DualNumber)):
            raise TypeError(f"Unsupported type `{type(other)}`")
        if isinstance(other, (int, float)):
            # scalar numbers
            return DualNumber(self.real - other, self.dual)
        else:
            # dual number
            return DualNumber(self.real - other.real, self.dual - other.dual)

    def __mul__(self, other):
        """Implements the multiplication of dual numbers

        :param other: A dual number or scalar
        :type other: Union[DualNumber, Union[int, float]]
        :return: The product of self with other
        :rtype: DualNumber
        """
        if not isinstance(other, (int, float, DualNumber)):
            raise TypeError(f"Unsupported type `{type(other)}`")
        if isinstance(other, (int, float)):
            # scalar numbers
            return DualNumber(other * self.real, other * self.dual)
        else:
            # dual number
            return DualNumber(
                self.real * other.real,
                self.real * other.dual + self.dual * other.real
            )

    def __truediv__(self, other):
        """Implements the division of dual numbers

        :param other: A dual number or scalar
        :type other: Union[DualNumber, Union[int, float]]
        :return: The division of self by other
        :rtype: DualNumber
        """
        if not isinstance(other, (int, float, DualNumber)):
            raise TypeError(f"Unsupported type `{type(other)}`")
        if isinstance(other, (int, float)):
            # scalar numbers
            return DualNumber(self.real / other, self.dual / other)
        if other.real == 0:
            raise ZeroDivisionError ("Division by zero is impossible")
        else:
            # dual number
            return DualNumber(
                self.real / other.real,
                (self.dual * other.real - self.real * other.dual)/(other.real**2)
            )

    def __pow__(self, other):
        """Implements the power of dual numbers

        :param other: A dual number or scalar
        :type other: Union[DualNumber, Union[int, float]]
        :return: self to the power of other
        :rtype: DualNumber
        """
        if not isinstance(other, (int, float, DualNumber)):
            raise TypeError(f"Unsupported type `{type(other)}`")
        if isinstance(other, (int, float)):
            if other == 0:
                return DualNumber(1,0)
            new_real = self.real ** other.real
            #new_dual = self.real ** (other.real - 1) * self.dual * self.real
            new_dual = other * self.dual * self.real ** (other - 1)
            return DualNumber(new_real, new_dual)
        else:
            new_real = self.real ** other.real
            #new_dual = other.real * (self.real ** (other.real - 1)) * self.dual + np.log(self.real) * (self.real ** (other.real)) * other.dual
            new_dual = other.real * self.dual * self.real ** (other.real - 1) + self.real ** other.real * other.dual * np.log(self.real)
            return DualNumber(new_real, new_dual)

    def __neg__(self):
        """Implements unary negation operator for dual numbers

        :return: Minus self
        :rtype: DualNumber
        """
        return DualNumber(- self.real, - self.dual)


    def __radd__(self, other):
        """Implements the right addition of a dual number with a scalar

        :param other: A scalar
        :type other: Union[int, float]
        :return: The sum of self and other
        :rtype: DualNumber
        """
        return self.__add__(other)

    def __rsub__(self, other):
        """Implements the subtraction of a dual number with a scalar

        :param other: A scalar
        :type other: Union[int, float]
        :return: The difference of other and self
        :rtype: DualNumber
        """
        return - self.__sub__(other)

    def __rmul__(self, other):
        """Implements the right multiplication of a dual number with a scalar

        :param other: A scalar
        :type other: Union[int, float]
        :return: The product of self and other
        :rtype: DualNumber
        """
        return self.__mul__(other)

    def __rtruediv__(self, other):
        """Implements division of a scalar by a dual number

        :param other: A scalar
        :type other: Union[int, float]
        :return: The division of other by self
        :rtype: DualNumber
        """
        if not isinstance(other, (int, float)):
            raise TypeError(f"Unsupported type `{type(other)}`")
        return DualNumber(other, 0) / self


    def __rpow__(self, other):
        """Implements the power of a scalar to a dual number

        :param other: A dual number or scalar
        :type other: Union[int, float]
        :return: other to the power of other
        :rtype: DualNumber
        """
        if not isinstance(other, (int, float)):
            raise TypeError(f"Unsupported type `{type(other)}`")
        return DualNumber(other, 0) ** self

    def __iadd__(self, other):
        """Implements the in-place addition of dual numbers

        :param other: A dual number or scalar
        :type other: Union[DualNumber, Union[int, float]]
        :return: The sum of self with other
        :rtype: DualNumber
        """
        new = self.__add__(other)
        self.real, self.dual = new.real, new.dual
        return self

    def __isub__(self, other):
        """Implements the in-place subtraction of dual numbers

        :param other: A dual number or scalar
        :type other: Union[DualNumber, Union[int, float]]
        :return: The difference of self with other
        :rtype: DualNumber
        """
        new = self.__sub__(other)
        self.real, self.dual = new.real, new.dual
        return self

    def __imul__(self, other):
        """Implements the in-place multiplication of dual numbers

        :param other: A dual number or scalar
        :type other: Union[DualNumber, Union[int, float]]
        :return: The product of self with other
        :rtype: DualNumber
        """
        new = self.__mul__(other)
        self.real, self.dual = new.real, new.dual
        return self

    def __itruediv__(self, other):
        """Implements the in-place division of dual numbers

        :param other: A dual number or scalar
        :type other: Union[DualNumber, Union[int, float]]
        :return: The division of self by other
        :rtype: DualNumber
        """
        new = self.__truediv__(other)
        self.real, self.dual = new.real, new.dual
        return self

    def __ipow__(self, other):
        """Implements the in-place power of dual numbers

        :param other: A dual number or scalar
        :type other: Union[DualNumber, Union[int, float]]
        :return: self to the power of other
        :rtype: DualNumber
        """
        new = self.__pow__(other)
        self.real, self.dual = new.real, new.dual
        return self

    def __str__(self):
        """Prints the dual number
        
        :return: A string representing the dual number
        :rtype: str
        """
        return f"({self.real}, {self.dual})"
    
    def __eq__(self, other):
        """Implements the equality of dual numbers, considering the real part only

        :param other: A dual number
        :type other: DualNumber
        :return: True if the real parts are equal, else False
        :rtype: bool
        """
        return (self.real == other.real)
        
    def __ne__(self, other):
        """Implements the inequality of dual numbers, considering the real part only

        :param other: A dual number
        :type other: DualNumber
        :return: True if the real parts are different, else False
        :rtype: bool
        """
        return (self.real != other.real) 
    
    def __ge__(self,other):
        """Implements greater than or equal to for dual numbers, considering the real part only

        :param other: A dual number
        :type other: DualNumber
        :return: True if the real part of self is greater than or equal to that of other, else False
        :rtype: bool
        """
        return self.real >= other.real

    def __le__(self, other):
        """Implements lower than or equal to for dual numbers, considering the real part only

        :param other: A dual number
        :type other: DualNumber
        :return: True if the real part of self is lower than or equal to that of other, else False
        :rtype: bool
        """
        return self.real <= other.real
        
    def  __gt__(self, other):
        """Implements greater than for dual numbers, considering the real part only

        :param other: A dual number
        :type other: DualNumber
        :return: True if the real part of self is greater than that of other, else False
        :rtype: bool
        """
        return self.real > other.real
        
    def  __lt__(self, other):
        """Implements lower than for dual numbers, considering the real part only

        :param other: A dual number
        :type other: DualNumber
        :return: True if the real part of self is lower than that of other, else False
        :rtype: bool
        """
        return self.real < other.real
    
    






if __name__ == "__main__":
    x = 1
    # z = DualNumber(x1)
    # a0 = 2.0
    # a1 = 2.5
    # a2 = 3.0
    # f = a0 + z * a2 * z + a1 * z
    # print(f.real)
    # print(f.dual)

    y = 2
    z1 = DualNumber(x, y)

    u = 2
    t = 1
    z2 = DualNumber(u, t)
    #z1 += z2
    print(2 ** z2)
