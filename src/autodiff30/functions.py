from autodiff30.dual import DualNumber
import numpy as np


def sin(x):
    """An implementation of the trigonometric function sine for dual numbers

    :param x: A dual number
    :type x: class `DualNumber`
    ...
    :return: A dual number that is sin(x)
    :rtype: class `DualNumber`
    """
    new_real = np.sin(x.real)
    new_dual = x.dual * np.cos(x.real)
    return DualNumber(new_real, new_dual)


def cos(x):
    """An implementation of the trigonometric function cosine for dual numbers

    :param x: A dual number
    :type x: class `DualNumber`
    ...
    :return: A dual number that is cos(x)
    :rtype: class `DualNumber`
    """
    new_real = np.cos(x.real)
    new_dual = x.dual * (-np.sin(x.real))
    return DualNumber(new_real, new_dual)


def tan(x):
    """An implementation of the trigonometric function tangent for dual numbers

    :param x: A dual number
    :type x: class `DualNumber`
    ...
    :return: A dual number that is tan(x)
    :rtype: class `DualNumber`
    """
    if (x.real - np.pi / 2) % np.pi == 0:
        raise ValueError("Tan is not defined on odd multiple of pi/2")
    else:
        new_real = np.tan(x.real)
        new_dual = x.dual * (1 / (np.cos(x.real) ** 2))
        return DualNumber(new_real, new_dual)


def arccos(x):
    """An implementation of the inverse trigonometric function arccosine for dual numbers

    :param x: A dual number
    :type x: class `DualNumber`
    ...
    :return: A dual number that is arccos(x)
    :rtype: class `DualNumber`
    """
    if x.real > -1 and x.real < 1:
        new_real = np.arccos(x.real)
        new_dual = x.dual * (-1 / (np.sqrt(1 - x.real**2)))
        return DualNumber(new_real, new_dual)
    else:
        raise ValueError("Arccos is not differentiable outside [-1,1]")


def arcsin(x):
    """An implementation of the inverse trigonometric function arcsine for dual numbers

    :param x: A dual number
    :type x: class `DualNumber`
    ...
    :return: A dual number that is arcsin(x)
    :rtype: class `DualNumber`
    """
    if x.real > -1 and x.real < 1:
        new_real = np.arcsin(x.real)
        new_dual = x.dual * (1 / (np.sqrt(1 - x.real**2)))
        return DualNumber(new_real, new_dual)
    else:
        raise ValueError("Arcsin is not differentiable outside [-1,1]")


def arctan(x):
    """An implementation of the inverse trigonometric function arctangent for dual numbers

    :param x: A dual number
    :type x: class `DualNumber`
    ...
    :return: A dual number that is arctan(x)
    :rtype: class `DualNumber`
    """
    new_real = np.arctan(x.real)
    new_dual = x.dual * (1 / (1 + x.real**2))
    return DualNumber(new_real, new_dual)


def exp(x):
    """An implementation of the exponential e^x function for dual numbers

    :param x: A dual number
    :type x: class `DualNumber`
    ...
    :return: A dual number that is arccos(x)
    :rtype: class `DualNumber`
    """
    new_real = np.exp(x.real)
    new_dual = x.dual * np.exp(x.real)
    return DualNumber(new_real, new_dual)


def log(x, base=np.e):
    """An implementation of the logarithmic function for dual numbers

    :param x: A dual number
    :type x: class `DualNumber`
    :param base: The base to take the logarithm with respect to, defaults to natural log
    :type x: int or float, optional
    ...
    :return: A dual number that is log_{base}(x)
    :rtype: class `DualNumber`
    """
    if not isinstance(base, (int, float)):
        raise TypeError(f"Unsupported base type `{type(base)}`")
    if base <= 1:
        raise ValueError("Log is not defined on base <=1")
    if x.real > 0:
        new_real = np.log(x.real) / np.log(base)
        new_dual = x.dual * 1 / x.real / np.log(base)
        return DualNumber(new_real, new_dual)
    else:
        raise ValueError("Log is not defined outside [0,inf]")


def sqrt(x):
    """An implementation of the square root function for dual numbers

    :param x: A dual number
    :type x: class `DualNumber`
    ...
    :return: A dual number that is sqrt(x)
    :rtype: class `DualNumber`
    """
    if x.real > 0:
        new_real = np.sqrt(x.real)
        new_dual = x.dual * 1 / (2 * np.sqrt(x.real))
        return DualNumber(new_real, new_dual)
    else:
        raise ValueError("Sqrt is not differentiable outside [0,inf]")


def sigmoid_real(x):
    """The sigmoid function for real numbers

    :param x: A real number
    :type x: int or float
    ...
    :return: The sigmoid output of x
    :rtype: int or float
    """
    return 1 / (1 + np.exp(-x))


def logistic(x):
    """An implementation of the logistic function for dual numbers

    :param x: A dual number
    :type x: class `DualNumber`
    ...
    :return: A dual number that is logistic(x)
    :rtype: class `DualNumber`
    """
    new_real = sigmoid_real(x.real)
    new_dual = x.dual * new_real * (1 - new_real)
    return DualNumber(new_real, new_dual)


def sinh(x):
    """An implementation of the hyperbolic sinh function for dual numbers

    :param x: A dual number
    :type x: class `DualNumber`
    ...
    :return: A dual number that is sinh(x)
    :rtype: class `DualNumber`
    """
    new_real = np.sinh(x.real)
    new_dual = x.dual * np.cosh(x.real)
    return DualNumber(new_real, new_dual)


def cosh(x):
    """An implementation of the hyperbolic cosh function for dual numbers

    :param x: A dual number
    :type x: class `DualNumber`
    ...
    :return: A dual number that is cosh(x)
    :rtype: class `DualNumber`
    """
    new_real = np.cosh(x.real)
    new_dual = x.dual * np.sinh(x.real)
    return DualNumber(new_real, new_dual)


def tanh(x):
    """An implementation of the hyperbolic tanh function for dual numbers

    :param x: A dual number
    :type x: class `DualNumber`
    ...
    :return: A dual number that is tanh(x)
    :rtype: class `DualNumber`
    """
    new_real = np.tanh(x.real)
    new_dual = x.dual * (1 / (np.cosh(x.real) ** 2))  # sech^2(x)
    return DualNumber(new_real, new_dual)


if __name__ == "__main__":
    x = DualNumber(50, 1)
    res = tan(x) * exp(sin(x)) - cos(x**0.5) * sin((cos(x) ** 2.0 + x**2.0) ** 0.5)
    print(res.real)
    print(res.dual)
