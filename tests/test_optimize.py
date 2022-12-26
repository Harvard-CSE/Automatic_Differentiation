#!/usr/env/bin python3
import pytest
import math
import numpy as np
from autodiff30.ad import adstruc, adfunction
import autodiff30.functions as adf
import autodiff30.optimization as ad_opt



class TestOptimize:

    xs_1D = [-10.0, 0.0, 10.0]
    xs_2D = [[-10.0, -10.0], [-10.0, 0.0], [0.0, 10.0], [10.0, 10.0]]
    xs_3D = [[-10.0, -10.0, -10.0], [-10.0, 0.0, 0.], [0.0, 10.0, 0.], [10.0, 10.0, 0.]]
    xs_4D = [[-10.0, -10.0, -10.0, -10.], [-10.0, 0.0, 0., 10.], [0.0, 10.0, 0., 10.], [10.0, 10.0, 0., -10.]]

    def test_1D(self):
        """Simple scalar functions of a single variable R1-> R1"""

        @adfunction
        def foo(x):
            return x**2

        for x in self.xs_1D:
            assert np.linalg.norm(ad_opt.GD(foo, x) - 0) < 1e-5
            assert np.linalg.norm(ad_opt.Adam(foo, x) - 0) < 1e-5

        @adfunction
        def foo(x):
            return x**2 + 4*x + 4

        for x in self.xs_1D:
            assert np.linalg.norm(ad_opt.GD(foo, x) + 2) < 1e-5
            assert np.linalg.norm(ad_opt.Adam(foo, x) + 2) < 1e-5

        @adfunction
        def foo(x):
            return x

        for x in self.xs_1D:
            with pytest.raises(RuntimeWarning):
                ad_opt.GD(foo, x)
            with pytest.raises(RuntimeWarning):
                ad_opt.Adam(foo, x)

    def test_2D(self):
        """Simple scalar functions of two variables R2-> R1"""


        @adfunction
        def foo(x):
            return x[0]**2  +  x[1] ** 2

        for x in self.xs_2D:
            assert np.linalg.norm(np.array(ad_opt.GD(foo, x)) - np.array([0,0])) < 1e-5
            assert np.linalg.norm(np.array(ad_opt.Adam(foo, x)) - np.array([0,0])) < 1e-5

        @adfunction
        def foo(x):
            return x[0]**2 + x[1]**2 + 4*x[0] + 4*x[1] + 4

        for x in self.xs_2D:
            assert np.linalg.norm(np.array(ad_opt.GD(foo, x)) - np.array([-2,-2])) < 1e-5
            assert np.linalg.norm(np.array(ad_opt.Adam(foo, x)) - np.array([-2,-2])) < 1e-5

        @adfunction
        def foo(x):
            return x[0] + x[1]

        for x in self.xs_2D:
            with pytest.raises(RuntimeWarning):
                ad_opt.GD(foo, x)
            with pytest.raises(RuntimeWarning):
                ad_opt.Adam(foo, x)
    def test_3D(self):
        """Simple scalar functions of two variables R2-> R1"""


        @adfunction
        def foo(x):
            return x[0]**2  +  x[1] ** 2 + x[2] ** 2

        for x in self.xs_3D:
            assert np.linalg.norm(np.array(ad_opt.GD(foo, x)) - np.array([0,0,0])) < 1e-5
            assert np.linalg.norm(np.array(ad_opt.Adam(foo, x)) - np.array([0,0,0])) < 1e-5

        @adfunction
        def foo(x):
            return x[0]**2 + x[1]**2 + x[2]**2 + 4*x[0] + 4*x[1] + 4*x[2] + 4

        for x in self.xs_3D:
            assert np.linalg.norm(np.array(ad_opt.GD(foo, x)) - np.array([-2,-2, -2])) < 1e-5
            assert np.linalg.norm(np.array(ad_opt.Adam(foo, x)) - np.array([-2,-2,-2])) < 1e-5

        @adfunction
        def foo(x):
            return x[0] + x[1] + x[2]

        for x in self.xs_3D:
            with pytest.raises(RuntimeWarning):
                ad_opt.GD(foo, x)
            with pytest.raises(RuntimeWarning):
                ad_opt.Adam(foo, x)
    def test_4D(self):
        """Simple scalar functions of two variables R2-> R1"""


        @adfunction
        def foo(x):
            return x[0]**2  +  x[1] ** 2 + x[2] ** 2 + x[3]**2

        for x in self.xs_4D:
            assert np.linalg.norm(np.array(ad_opt.GD(foo, x)) - np.array([0,0,0, 0])) < 1e-5
            assert np.linalg.norm(np.array(ad_opt.Adam(foo, x)) - np.array([0,0,0,0])) < 1e-5

        @adfunction
        def foo(x):
            return x[0]**2 + x[1]**2 + x[2]**2 + x[3]**2+ 4*x[0] + 4*x[1] + 4*x[2] + 4*x[3]+ 4

        for x in self.xs_4D:
            assert np.linalg.norm(np.array(ad_opt.GD(foo, x)) - np.array([-2,-2, -2, -2])) < 1e-5
            assert np.linalg.norm(np.array(ad_opt.Adam(foo, x)) - np.array([-2,-2,-2, -2])) < 1e-5

        @adfunction
        def foo(x):
            return x[0] + x[1] + x[2] + x[3]

        for x in self.xs_4D:
            with pytest.raises(RuntimeWarning):
                ad_opt.GD(foo, x)
            with pytest.raises(RuntimeWarning):
                ad_opt.Adam(foo, x)

