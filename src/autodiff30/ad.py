#!/usr/env/bin python3

from .dual import DualNumber
import numpy as np
from typing import Union, List, Callable


number = Union[int, float]
OptListDualNumber = Union[DualNumber, List[DualNumber]]
OptListNumber = Union[number, List[number]]


def _select_part(x:OptListDualNumber, part:str) -> OptListNumber:
    """Returns either the real or dual parts (according to the param part) of x if x is a dual number,
    or of all elements in x if x is a list of dual numbers.

    :param x: A dual number or list of dual numbers
    :type x: Union[DualNumber, List[DualNumber]]
    ...
    :return: Either the real or dual parts of all dual numbers in input
    :rtype: Union[float, List[float]]
    """
    assert isinstance(x, (DualNumber, list))
    if isinstance(x, DualNumber):
        return x.real if part == "real" else x.dual
    else:
        return [y.real for y in x] if part == "real" else [y.dual for y in x]

class adstruc:
    """Structure that is returned from the decorator that implements calling and gradient

    :param f: A function that can work on dual numbers (or list of dual numbers)
    :type f: Callable[[Union[DualNumber, List[DualNumber]]], Union[DualNumber, List[DualNumber]]]
    """

    def __init__(self, f:Callable[[OptListDualNumber], OptListDualNumber]) -> None:
        self.f = f

    def __call__(self, x:OptListNumber) -> OptListNumber:
        """Computes the function f on the input x

        :param x: A scalar or list of scalars
        :type x: Union[Union[int, float], List[Union[int, float]]]
        :return: The value f(x)
        :rtype: Union[Union[int, float], List[Union[int, float]]]
        """
        assert isinstance(x, (list, int, float))

        if isinstance(x, (int, float)):
            res = self.f(DualNumber(x, 0))

        else:
            ds = [DualNumber(elt, 0) for elt in x]
            res = self.f(ds)

        return _select_part(res, "real")


    def grad(self, x:OptListNumber) -> OptListNumber:
        """Computes the gradient of the function f at the input x

        :param x: A scalar or list of scalars
        :type x: Union[Union[int, float], List[Union[int, float]]]
        :return: The value grad(f)(x)
        :rtype: Union[Union[int, float], List[Union[int, float]]]
        """
        assert isinstance(x, (list, int, float))

        if isinstance(x, (int, float)):
            d = DualNumber(x, 1)
            return _select_part(self.f(d), "dual")

        else:
            J = []
            for i in range(len(x)):
                ps = [0]*len(x)
                ps[i] = 1
                ds = [DualNumber(elt, p) for elt, p in zip(x, ps)]
                res = self.f(ds)
                J.append(_select_part(res, "dual"))
            return np.array(J).T.tolist()



def adfunction(f:Callable) -> adstruc:
    """Functor for function decoration for calling and gradient

    :param f: A function that can work on dual numbers (or list of dual numbers)
    :type f: Callable[[Union[DualNumber, List[DualNumber]]], Union[DualNumber, List[DualNumber]]]
    """

    return adstruc(f)


if __name__ == "__main__":

    @adfunction
    def foo(x):
        """
        x is a numeric type
        foo is a scalar function
        """
        return x * x

    x = 3
    print(foo(x))
    print(foo.grad(x))
