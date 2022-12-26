import pytest
import numpy as np
import math

from autodiff30.dual import DualNumber
import autodiff30.functions as adf


def logistic(x):
    return 1 / (1 + np.exp(-x))


class TestMathFunctions:
    """Test class for math functions"""

    values = [(0.3, -0.7), (0.5, 0), (0, 0.5)]

    def test_exp(self):
        for (x, y) in self.values:
            z = DualNumber(x, y)
            res = adf.exp(z)
            assert res.real == np.exp(x) and res.dual == y * np.exp(x)

    def test_log(self):
        for (x, y) in self.values[:-1]:
            for b in [2, np.e, 10]:
                z = DualNumber(x, y)
                res = adf.log(z, base=b)
                assert math.isclose(res.real, np.log(x) / np.log(b)) and math.isclose(
                    res.dual, y / np.log(b) / x
                )
                # assert res.real==np.log(x)/np.log(b) and res.dual==y/np.log(b)/x
        with pytest.raises(ValueError):
            res = adf.log(DualNumber(0, 1))
            res = adf.log(DualNumber(-1, 1))

    def test_sin(self):
        for (x, y) in self.values:
            z = DualNumber(x, y)
            res = adf.sin(z)
            assert math.isclose(res.real, np.sin(x)) and math.isclose(
                res.dual, y * np.cos(x)
            )

            # assert res.real==np.sin(x) and res.dual==y*np.cos(x)

    def test_cos(self):
        for (x, y) in self.values:
            z = DualNumber(x, y)
            res = adf.cos(z)
            assert math.isclose(res.real, np.cos(x)) and math.isclose(
                res.dual, -y * np.sin(x)
            )
            # assert res.real==np.cos(x) and res.dual==-y*np.sin(x)

    def test_tan(self):
        for (x, y) in self.values:
            z = DualNumber(x, y)
            res = adf.tan(z)
            assert math.isclose(res.real, np.tan(x)) and math.isclose(
                res.dual, y * (1 + np.tan(x) ** 2)
            )
            # assert res.real==np.tan(x) and res.dual==y*(1+np.tan(x)**2)

    def test_arcsin(self):
        for (x, y) in self.values:
            z = DualNumber(x, y)
            res = adf.arcsin(z)
            assert math.isclose(res.real, np.arcsin(x)) and math.isclose(
                res.dual, y * 1 / np.sqrt(1 - x**2)
            )
            # assert res.real==np.arcsin(x) and res.dual==y*1/np.sqrt(1-x**2)
        with pytest.raises(ValueError):
            res = adf.arcsin(DualNumber(1, 1))
            res = adf.arcsin(DualNumber(-1.5, 1))

    def test_arccos(self):
        for (x, y) in self.values:
            z = DualNumber(x, y)
            res = adf.arccos(z)
            assert math.isclose(res.real, np.arccos(x)) and math.isclose(
                res.dual, -y * 1 / np.sqrt(1 - x**2)
            )
            # assert res.real==np.arccos(x) and res.dual==-y*1/np.sqrt(1-x**2)
        with pytest.raises(ValueError):
            res = adf.arccos(DualNumber(1, 1))
            res = adf.arccos(DualNumber(-1.5, 1))

    def test_arctan(self):
        for (x, y) in self.values:
            z = DualNumber(x, y)
            res = adf.arctan(z)
            assert math.isclose(res.real, np.arctan(x)) and math.isclose(
                res.dual, y * 1 / (1 + x**2)
            )
            # assert res.real==np.arctan(x) and res.dual==y*1/(1+x**2)

    def test_sqrt(self):
        for (x, y) in self.values[:-1]:
            z = DualNumber(x, y)
            res = adf.sqrt(z)
            assert res.real == np.sqrt(x) and res.dual == y / (2 * np.sqrt(x))
        with pytest.raises(ValueError):
            res = adf.sqrt(DualNumber(-1, 1))

    def test_logistic(self):
        for (x, y) in self.values:
            z = DualNumber(x, y)
            res = adf.logistic(z)
            assert res.real == logistic(x) and res.dual == y * logistic(x) * (
                1 - logistic(x)
            )

    def test_sinh(self):
        for (x, y) in self.values:
            z = DualNumber(x, y)
            res = adf.sinh(z)
            assert math.isclose(res.real, np.sinh(x)) and math.isclose(
                res.dual, y * np.cosh(x)
            )

            # assert res.real==np.sin(x) and res.dual==y*np.cos(x)

    def test_cosh(self):
        for (x, y) in self.values:
            z = DualNumber(x, y)
            res = adf.cosh(z)
            assert math.isclose(res.real, np.cosh(x)) and math.isclose(
                res.dual, y * np.sinh(x)
            )
            # assert res.real==np.cos(x) and res.dual==-y*np.sin(x)

    def test_tanh(self):
        for (x, y) in self.values:
            z = DualNumber(x, y)
            res = adf.tanh(z)
            assert math.isclose(res.real, np.tanh(x)) and math.isclose(
                res.dual, y * (1 / (np.cosh(x) ** 2))
            )
