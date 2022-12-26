#!/usr/env/bin python3
import pytest
import math

from autodiff30.dual import DualNumber
from numpy import log


class TestDualNumber:

    values = [(0.3, -0.7), (0.5, 0.0), (0.0, 0.5)]

    def test_init(self):
        test_dual = DualNumber(1, 2)
        assert test_dual.real == 1
        assert test_dual.dual == 2

    def test_comparison(self):
        test_dual1 = DualNumber(1, 2)
        test_dual2 = DualNumber(1, 3)
        test_dual3 = DualNumber(0, 3)
        assert test_dual1 == test_dual2
        assert test_dual1 != test_dual3
        assert test_dual1 >= test_dual3
        assert test_dual3 <= test_dual1
        assert test_dual1 > test_dual3
        assert test_dual3 < test_dual1
        assert test_dual1 >= 0
        assert test_dual1 == 1

    def test_addition(self):
        # Adding dual numbers
        res = [(0.8, -0.7), (0.5, 0.5)]
        for i in range(len(TestDualNumber.values) - 1):
            test_dual1 = DualNumber(
                TestDualNumber.values[i][0], TestDualNumber.values[i][1]
            )
            test_dual2 = DualNumber(
                TestDualNumber.values[i + 1][0], TestDualNumber.values[i + 1][1]
            )
            test_dual_add = test_dual1 + test_dual2
            assert math.isclose(test_dual_add.real, res[i][0])
            assert math.isclose(test_dual_add.dual, res[i][1])
            #assert test_dual_add.real == res[i][0]
            #assert test_dual_add.dual == res[i][1]

        # Adding dual to int
        res = [(-6, 2), (1, 2), (6, 2)]
        test_dual1 = DualNumber(1, 2)
        for i, (_, d) in enumerate(TestDualNumber.values):
            test_dual_intadd = test_dual1 + int(d * 10)
            assert test_dual_intadd.real == res[i][0]
            assert test_dual_intadd.dual == res[i][1]

        # Adding dual to float
        res = [(0.3, 2), (1.0, 2.0), (1.5, 2.0)]
        for i, (_, d) in enumerate(TestDualNumber.values):
            test_dual_floatadd = test_dual1 + d
            assert math.isclose(test_dual_floatadd.real, res[i][0])
            assert math.isclose(test_dual_floatadd.dual, res[i][1])
            # assert test_dual_floatadd.real == res[i][0]
            # assert test_dual_floatadd.dual == res[i][1]

        # Check incorrect types raise an error
        with pytest.raises(TypeError):
            test_dual1 + "1"
            "1" + test_dual1
            test_dual1 + [1]
            [1] + test_dual1

    def test_subtraction(self):
        # Subtracting dual numbers
        res = [(-0.2, -0.7), (0.5, -0.5)]
        for i in range(len(TestDualNumber.values) - 1):
            test_dual1 = DualNumber(
                TestDualNumber.values[i][0], TestDualNumber.values[i][1]
            )
            test_dual2 = DualNumber(
                TestDualNumber.values[i + 1][0], TestDualNumber.values[i + 1][1]
            )
            test_dual_sub = test_dual1 - test_dual2
            assert math.isclose(test_dual_sub.real, res[i][0])
            assert math.isclose(test_dual_sub.dual, res[i][1])
            # assert test_dual_sub.real == res[i][0]
            # assert test_dual_sub.dual == res[i][1]

        # Subtracting int from dual
        res = [(8, 2), (1, 2), (-4, 2)]
        test_dual1 = DualNumber(1, 2)
        for i, (_, d) in enumerate(TestDualNumber.values):
            test_dual_intsub = test_dual1 - int(d * 10)
            assert test_dual_intsub.real == res[i][0]
            assert test_dual_intsub.dual == res[i][1]

        # Subtracting dual from int
        res = [(-8, -2), (-1, -2), (4, -2)]
        test_dual1 = DualNumber(1, 2)
        for i, (_, d) in enumerate(TestDualNumber.values):
            test_dual_intsub = int(d * 10) - test_dual1
            assert test_dual_intsub.real == res[i][0]
            assert test_dual_intsub.dual == res[i][1]

        # Subtracting float from dual
        res = [(1.7, 2), (1.0, 2.0), (0.5, 2.0)]
        for i, (_, d) in enumerate(TestDualNumber.values):
            test_dual_floatsub = test_dual1 - d
            assert test_dual_floatsub.real == res[i][0]
            assert test_dual_floatsub.dual == res[i][1]

        # Subtracting dual from float
        res = [(-1.7, -2), (-1.0, -2.0), (-0.5, -2.0)]
        for i, (_, d) in enumerate(TestDualNumber.values):
            test_dual_floatsub = d - test_dual1
            assert test_dual_floatsub.real == res[i][0]
            assert test_dual_floatsub.dual == res[i][1]

        # Check incorrect types raise an error
        with pytest.raises(TypeError):
            test_dual1 - "1"
            "1" - test_dual1
            test_dual1 - [1]
            [1] - test_dual1

    def test_multiplication(self):
        # Dual times dual
        res = [(0.3 * 0.5, -0.7 * 0.5), (0.0, 0.5 * 0.5)]
        for i in range(len(TestDualNumber.values) - 1):
            test_dual1 = DualNumber(
                TestDualNumber.values[i][0], TestDualNumber.values[i][1]
            )
            test_dual2 = DualNumber(
                TestDualNumber.values[i + 1][0], TestDualNumber.values[i + 1][1]
            )
            test_dual_mul = test_dual1 * test_dual2
            assert math.isclose(test_dual_mul.real, res[i][0])
            assert math.isclose(test_dual_mul.dual, res[i][1])
            # assert test_dual_mul.real == res[i][0]
            # assert test_dual_mul.dual == res[i][1]

        # Multiplying dual with int
        res = [(-7, -14), (0, 0), (5, 10)]
        test_dual1 = DualNumber(1, 2)
        for i, (_, d) in enumerate(TestDualNumber.values):
            test_dual_intmul = test_dual1 * int(d * 10)
            assert test_dual_intmul.real == res[i][0]
            assert test_dual_intmul.dual == res[i][1]

        # Multiplying dual with float
        res = [(-0.7, -1.4), (0.0, 0.0), (0.5, 1.0)]
        test_dual1 = DualNumber(1, 2)
        for i, (_, d) in enumerate(TestDualNumber.values):
            test_dual_floatmul = test_dual1 * d
            assert test_dual_floatmul.real == res[i][0]
            assert test_dual_floatmul.dual == res[i][1]

        # Check incorrect types raise an error
        with pytest.raises(TypeError):
            test_dual1 * "1"
            "1" * test_dual1
            test_dual1 * [1]
            [1] * test_dual1

    def test_division(self):
        # Dual divided by dual
        res = [(1 / 3, 2 / 9)]
        test_dual1 = DualNumber(1, 2)
        test_dual2 = DualNumber(3, 4)
        test_dual_div = test_dual1 / test_dual2
        assert test_dual_div.real == res[0][0]
        assert test_dual_div.dual == res[0][1]

        # Dividing dual with int
        res = [(0.5, 1.0)]
        test_dual_intdiv = test_dual1 / 2
        assert test_dual_intdiv.real == res[0][0]
        assert test_dual_intdiv.dual == res[0][1]

        # Dividing dual with float
        res = [(0.5, 1.0)]
        test_dual_floatdiv = test_dual1 / 2.0
        assert test_dual_floatdiv.real == res[0][0]
        assert test_dual_floatdiv.dual == res[0][1]

        # Check incorrect types raise an error
        with pytest.raises(TypeError):
            test_dual1 / "1"
            "1" / test_dual1
            test_dual1 / [1]
            [1] / test_dual1

    def test_power(self):
        # Dual to the power of dual
        res = [(8, 32 * log(2) + 24)]
        test_dual1 = DualNumber(2, 2)
        test_dual2 = DualNumber(3, 4)
        test_dual_pow = test_dual1**test_dual2
        assert test_dual_pow.real == res[0][0]
        assert test_dual_pow.dual == res[0][1]

        # Dual to the power of int
        res = [(27, 12 * 9)]
        test_dual_intpow = test_dual2**3
        assert test_dual_intpow.real == res[0][0]
        assert test_dual_intpow.dual == res[0][1]

        # Dual to the power of float
        res = [(27.0, 12.0 * 9.0)]
        test_dual_floatpow = test_dual2**3.0
        assert test_dual_floatpow.real == res[0][0]
        assert test_dual_floatpow.dual == res[0][1]

        # Check incorrect types raise an error
        with pytest.raises(TypeError):
            test_dual1 ** "1"
            "1" ** test_dual1
            test_dual1 ** [1]
            [1] ** test_dual1

    def test_reflective_operators(self):
        test_dual1 = DualNumber(2, 1)
        test_dual2 = DualNumber(4, 3)
        test_dual_add = test_dual1 + test_dual2
        test_dual_radd = test_dual2 + test_dual1
        test_int_add = test_dual1 + 3
        test_int_radd = 3 + test_dual1
        test_float_add = test_dual1 + 5.0
        test_float_radd = 5.0 + test_dual1
        test_dual_sub = test_dual1 - test_dual2
        test_dual_rsub = test_dual2 - test_dual1
        test_dual_mul = test_dual1 * test_dual2
        test_dual_rmul = test_dual2 * test_dual1
        test_int_mul = test_dual1 * 3
        test_int_rmul = 3 * test_dual1
        test_float_mul = test_dual1 * 5.0
        test_float_rmul = 5.0 * test_dual1

        assert test_dual_add.real == test_dual_radd.real
        assert test_dual_add.dual == test_dual_radd.dual
        assert test_int_add.real == test_int_radd.real
        assert test_int_add.dual == test_int_radd.dual
        assert test_float_add.real == test_float_radd.real
        assert test_float_add.dual == test_float_radd.dual
        assert test_dual_sub.real == -test_dual_rsub.real
        assert test_dual_sub.dual == -test_dual_rsub.dual
        assert test_dual_mul.real == test_dual_rmul.real
        assert test_dual_mul.dual == test_dual_rmul.dual
        assert test_int_mul.real == test_int_rmul.real
        assert test_int_mul.dual == test_int_rmul.dual
        assert test_float_mul.real == test_float_rmul.real
        assert test_float_mul.dual == test_float_rmul.dual
