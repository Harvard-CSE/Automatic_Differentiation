#!/usr/env/bin python3
import pytest
import math

from autodiff30.ad import adstruc, adfunction
import autodiff30.functions as adf

# from numpy import log, exp, sin, cos, tan, arcsin, arccos, arctan, sqrt
import numpy as np


def logistic(x):
    return 1 / (1 + np.exp(-x))


class TestUserLevel:

    x = 3

    def test_scalar_simple(self):
        """Simple scalar functions of a single variable using only arithmetic operators"""

        @adfunction
        def foo(x):
            return x

        assert foo(self.x) == 3
        assert foo.grad(self.x) == 1

        @adfunction
        def foo(x):
            return x[0] * x[1]

        assert foo([2, 3]) == 6
        assert foo.grad([2, 3]) == [3, 2]

        @adfunction
        def foo(x):
            return [2 * x]

        assert foo(2) == [4]
        assert foo.grad(2) == [2]

        @adfunction
        def foo(x):
            return [x, 2 * x]

        assert foo(2) == [2, 4]
        assert foo.grad(2) == [1, 2]

        @adfunction
        def foo(x):
            print(x)
            return [x[0] * x[1], x[0] + x[1]]

        assert foo([2, 3]) == [6, 5]
        assert foo.grad([2, 3]) == [[3, 2], [1, 1]]

        @adfunction
        def foo(x):
            return 2 * x

        assert foo(self.x) == 6
        assert foo.grad(self.x) == 2

        @adfunction
        def foo(x):
            return 2 + x

        assert foo(self.x) == 5
        assert foo.grad(self.x) == 1

        @adfunction
        def foo(x):
            return 2 - x

        assert foo(self.x) == -1
        assert foo.grad(self.x) == -1

        @adfunction
        def foo(x):
            return x / 2

        assert foo(self.x) == 1.5
        assert foo.grad(self.x) == 1 / 2

        @adfunction
        def foo(x):
            return 2 / x

        assert foo(self.x) == 2 / 3
        assert foo.grad(self.x) == -2 / 9

        @adfunction
        def foo(x):
            return x**x

        assert foo(self.x) == 27
        assert foo.grad(self.x) == 27 * (np.log(3) + 1)

        @adfunction
        def foo(x):
            return x**3

        assert foo(self.x) == 27
        assert foo.grad(self.x) == 27

    def test_scalar_math(self):
        """Simple scalar function of a single variable using only math operators"""

        @adfunction
        def foo(x):
            return adf.exp(x)

        assert math.isclose(foo(self.x), np.exp(3))
        assert foo.grad(self.x) == np.exp(3)

        @adfunction
        def foo(x):
            return adf.log(x, base=np.e)

        assert math.isclose(foo(self.x), np.log(3))
        assert math.isclose(foo.grad(self.x), 1 / 3)

        @adfunction
        def foo(x):
            return adf.sin(x)

        assert foo(self.x) == np.sin(3)
        assert foo.grad(self.x) == np.cos(3)

        @adfunction
        def foo(x):
            return adf.cos(x)

        assert foo(self.x) == np.cos(3)
        assert foo.grad(self.x) == -np.sin(3)

        @adfunction
        def foo(x):
            return adf.tan(x)

        assert foo(self.x) == np.tan(3)
        assert foo.grad(self.x) == 1 / (np.cos(3) ** 2)

        @adfunction
        def foo(x):
            return adf.arcsin(x)

        assert foo(self.x / 10) == np.arcsin(self.x / 10)
        assert foo.grad(self.x / 10) == 1 / np.sqrt(1 - (self.x / 10) ** 2)

        @adfunction
        def foo(x):
            return adf.arccos(x)

        assert foo(self.x / 10) == np.arccos(self.x / 10)
        assert foo.grad(self.x / 10) == -1 / np.sqrt(1 - (self.x / 10) ** 2)

        @adfunction
        def foo(x):
            return adf.arctan(x)

        assert foo(self.x / 10) == np.arctan(self.x / 10)
        assert foo.grad(self.x / 10) == 1 / (1 + (self.x / 10) ** 2)

        @adfunction
        def foo(x):
            return adf.sqrt(x)

        assert foo(self.x) == np.sqrt(self.x)
        assert foo.grad(self.x) == 1 / (2 * np.sqrt(self.x))

        @adfunction
        def foo(x):
            return adf.logistic(x)

        assert foo(self.x) == logistic(self.x)
        assert foo.grad(self.x) == (1 - logistic(self.x)) * logistic(self.x)

    def test_scalar_complex(self):
        """More complex scalar functions of a single variable using"""

        @adfunction
        def foo(x):
            return (x**3 - (5 * x)) / ((x * x) + 10)

        assert foo(self.x) == 12 / 19
        assert foo.grad(self.x) == 346 / 361

        @adfunction
        def foo(x):
            return (adf.sin(x) + adf.cos(x)) ** 2

        assert math.isclose(foo(self.x), (np.sin(self.x) + np.cos(self.x)) ** 2)
        assert math.isclose(
            foo.grad(self.x), 2 * ((np.cos(self.x) ** 2) - (np.sin(self.x) ** 2))
        )
        # assert foo.grad(self.x) == 2 * ((np.cos(self.x) ** 2) - (np.sin(self.x) ** 2))

        @adfunction
        def foo(x):
            return adf.arcsin(x) * adf.arccos(x)

        assert foo(self.x / 10) == np.arcsin(self.x / 10) * np.arccos(self.x / 10)
        assert foo.grad(self.x / 10) == (
            np.arccos(self.x / 10) - np.arcsin(self.x / 10)
        ) / np.sqrt(1 - ((self.x / 10) ** 2))

        @adfunction
        def foo(x):
            return ((x**2) * adf.log(x, base=np.e)) / adf.exp(x)

        assert math.isclose(foo(self.x), ((self.x**2) * np.log(self.x)) / np.exp(self.x))
        assert math.isclose(
            foo.grad(self.x),
            np.exp(-self.x) * (self.x - (self.x - 2) * self.x * np.log(self.x)),
        )
        # assert foo.grad(self.x) == np.exp(-self.x) * (
        #     self.x - (self.x - 2) * self.x * np.log(self.x)
        # )

        @adfunction
        def foo(x):
            return_val = adf.tan(x) * adf.exp(adf.sin(x)) - adf.cos(x**0.5) * adf.sin(
                (adf.cos(x) ** 2.0 + x**2.0) ** 0.5
            )
            return return_val

        assert foo(self.x) == np.tan(self.x) * np.exp(np.sin(self.x)) - np.cos(
            self.x**0.5
        ) * np.sin((np.cos(self.x) ** 2.0 + self.x**2.0) ** 0.5)
        # assert math.isclose(
        #     foo.grad(self.x),
        #     1.17293,
        # )
        # assert math.isclose(
        #     foo.grad(self.x),
        #     1/(self.x**0.5) * (0.5 * np.sin(self.x**0.5) * np.sin((self.x**2 + (np.cos(self.x)**2)) ** 0.5)) + 1/((self.x**2 + (np.cos(self.x)**2))**0.5) * (np.cos(self.x**0.5) * np.cos((self.x**2 + (np.cos(self.x)**2)) ** 0.5) * (np.sin(self.x) * np.cos(self.x) - 1)) + np.exp(np.sin(self.x)) + np.exp(np.sin(self.x))/((np.cos(self.x))**2) ,
        # )

        assert math.isclose(
            foo.grad(self.x),
            1.17292966,
        )
