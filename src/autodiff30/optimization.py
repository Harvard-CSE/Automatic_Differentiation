#!/usr/env/bin python3
import numpy as np
from .ad import adfunction
from .dual import DualNumber


def GD(f, initial_guess, lr = 0.01, max_iter = 1000, epsilon = 1e-7):
    """
    GD is a basic gradient descent algorithm to minimize a function
    f: the function to optimize (f must have values in a space of dimension = 1)
    initial_guess: (int, float, list) to start the gradient descent with
    lr: learning rate
    max_iter: maximum number of iteration
    epsilon: stopping condition
    """
    assert isinstance(initial_guess, (int, float, list))

    #Check that the output dimension is 1
    if not isinstance(f(initial_guess), (int, float)):
        raise ValueError("For optimization, function output should be int or float. Consider optimizing the  norm of the function.")

    #Optimize the given function with gradient descent
    opt = np.array(initial_guess)
    for t in range(max_iter):
        _grad = np.array(f.grad(opt.tolist()))
        update = lr * _grad
        _grad_norm = np.abs(np.linalg.norm(_grad.real)) if isinstance(_grad, DualNumber) else np.abs(np.linalg.norm(_grad))
        _grad_norm = np.abs(np.linalg.norm(_grad))
        if _grad_norm < epsilon:
            break
        if t == max_iter - 1:
            raise RuntimeWarning("GD algorithm did not converge")
        opt -= update
    return opt



def Adam(f, initial_guess, lr = 0.001, beta1 = 0.9, beta2 =0.999, max_iter = 1000, epsilon = 1e-7):
    """
    Adam is an optimize and more advanced gradient descent algorithm to minimize a function
    f: the function to optimize (f must have values in a space of dimension = 1)
    initial_guess: (int, float, list) to start the gradient descent with
    lr: learning rate
    beta1, beta2: hyperparameters
    max_iter: maximum number of iteration
    epsilon: stopping condition
    """
    assert isinstance(initial_guess, (int, float, list))
    #Check that the output dimension is 1
    if not isinstance(f(initial_guess), (int, float)):
        raise ValueError("For optimization, function output should be int or float. Consider optimizing the  norm of the function.")

    #Optimize the given function with Adam
    t = 0
    m = 0
    v = 0
    opt = np.array(initial_guess)
    while t < max_iter:
        t +=1
        _grad = np.array(f.grad(opt.tolist()))
        m = beta1 * m + (1 - beta1) * _grad
        b = beta2 * v + (1 - beta2) * _grad**2
        m_ = m / (1 - beta1**t)
        v_ = v / (1 - beta2**t)
        opt = opt - lr * m_ / np.sqrt(v + 1e-7)
        _grad_norm = np.abs(np.linalg.norm(_grad.real)) if isinstance(_grad, DualNumber) else np.abs(np.linalg.norm(_grad))
        if _grad_norm < epsilon:
            break
        if t == max_iter - 1:
            raise RuntimeWarning("GD algorithm did not converge")
    return opt







