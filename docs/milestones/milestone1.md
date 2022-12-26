## Introduction
[Describe the problem the software solves and why it's important to solve that problem.]::

Automatic differentiation (AD or autodiff throughout) is a set of techniques for numerically evaluating a gradient. It achieves this by breaking the function-to-be-differentiated into a set of elementary operations and using the chain rule. We can contrast this with two other flavours of differentiation: symbolic and finite-differences. Symbolic differentiation aims to calculate a symbolic mathematical expression for the derivative of a particular function, while finite-differences numerically approximates a derivative at a particular point. Compared to the latter of these, AD achieves machine precision accuracy without sacrificing computational complexity and does not slow down as much when input dimensionality is high [<sup>[1]</sup>](https://arxiv.org/abs/1502.05767). Compared to symbolic differentation, AD is not limited to closed-form functions, and can handle loops, recursion and control flow.

Not only is autodiff powerful compared to other differentiation techniques, but it is already used widely. Gradient-based optimization, using the negative gradient of a function to step towards its local minimum, is pivotal to deep learning implementations. These have come to dominate many machine learning domains including computer vision and natural language processing. Implementations of computational methods for probabilistic inference, such as Hamiltonian Monte Carlo and Variational Inference, have also come to rely on automatic differentiation for the computation of necessary gradients.


## Background
[Describe (briefly) the mathematical background and concepts as you see fit. You do not need to give a treatise on automatic differentiation or dual numbers. Just give the essential ideas (e.g. the chain rule, the graph structure of calculations, elementary functions, etc). Do not copy and paste any of the lecture notes. We will easily be able to tell if you did this as it does not show that you truly understand the problem at hand.]::

We can consider any function that we may want the derivative of as being made up of elementary operations applied sequentially, such as arithmetic operations, switching sign, exponential, logarithmic, and trigonometric operators. We can easily compute the derivatives of such operations. Combining these derivatives through the chain rule gives the foundations of automatic differentiation.

#### The Chain Rule
In its simplest form, for real-valued functions of one variable, with $y = f(u)$ and $u = g(x)$ then the chain rule states
```math
\frac{dy}{dx}=\frac{dy}{du}\frac{du}{dx}
```

Generalizing to an input vector $\mathbf{x}$ of dimensionality $m$ and a scalar function $f$ action on a vector of $n$ other functions $y_i(\mathbf{x})$, we also have a more general form of the chain rule:
```math
\nabla_x f = \sum_{i=1}^n \frac{\partial f}{\partial y_i} \nabla y_i(\mathbf{x})
```

#### Example
This example is motivated by [[1]](https://arxiv.org/abs/1502.05767). Consider the function $y = f(x1, x2) = \cos(x1) + x1x2 − \exp(x2)$ where we want to find $\frac{\partial y}{\partial x_1}$ at $(x_1, x_2) = (2,4)$. We can decompose this problem into the steps in the table below.

|Elementary operations| Derivative w.r.t $x_1$ |
| ------------------- | ---------------------- |
|$v_{-1}=x_1$|$\dot{v_{-1}}=1$|
|$v_{0}=x_2$|$\dot{v_{0}}=0$|
|$v_{1}=\cos(v_{-1})$|$\dot{v_{1}}=-\sin(v_{-1}) \dot{v_{-1}}$|
|$v_{2}=v_{-1}v_{0}$|$\dot{v_{2}} = \dot{v_{-1}}v_{0} + v_{-1} \dot{v_{0}}$|
|$v_{3}=\exp(v_{0})$|$\dot{v_{3}} = \dot{v_{0}} \exp(v_{0})$|
|$v_{4}=v_{1}+v_{2}$|$\dot{v_{4}} = \dot{v_{1}} + \dot{v_{2}}$|
|$v_{5}=v_{4}-v_{3}$|$\dot{v_{5}} = \dot{v_{4}} - \dot{v_{3}}$|
|$y=v_{5}$|$\dot{y} =\dot{v_{5}} $|

The function can equivalently be displayed as a computational graph, where the nodes are intermediate variables and edges represent elementary operators:
<img src=./images/example_computational_graph.png" width="100">

Automatic differentiation can operate in two modes:
 - Forward mode
     - The principal mode this package will focus on
     - Follow the chain rule from "inside" to "outside", i.e. in this example starting with $\frac{\partial v_{-1}}{\partial x_1} = 1$ and ending with $\frac{\partial v_5}{\partial x_1} = \frac{\partial y}{\partial x_1}$
     - Goes in the same direction as the evaluation of the function itself
     - Starts from the right-hand end of the chain rule expression above
     - Preferred when output space is much bigger than input space
- Reverse mode
    - Follow the chain rule from "outside" to "inside"
    - Goes in the reverse order as the evaluation of the function 
    - Starts from the left-hand side of the chain rule expression above
    - Preferred when input space is much bigger than output space
    
The first column in the table is called the **Primal trace** and the second column is the **Forward tangent trace**. We can see how evaluating the intermediate steps (primals) as well as their derivatives (tangents) at the required point $(x_1, x_2) = (2,4)$ starting from $v_{-1}$ and $\dot{v_{-1}}$ eventually gives the required derivative evaluated at the correct point.

We can equivalently define a directional derivative operator $D_p y_i = \sum_{j=1}^m \frac{\partial y_i}{\partial x_j} p_j$, were we are projecting the gradient vector in the direction of a vector $p$ with elements $p_i$. By selecting a $p$ vector appropriate for the derivative we want, and letting the $y_i$ be the intermediate variables in the trace, we achieve the same thing, except strictly now we are computing $\mathbf{J} \cdot \mathbf{p}$ were $\mathbf{J}$ is the Jacobian. It takes $m$ passes to compute the full Jacobian.

#### Dual numbers
Forward mode autodiff can be achieved through as representing a function using dual numbers $v + \dot{v} \epsilon$ were $\epsilon \ne 0$ and $\epsilon^2 = 0$. The primal trace is carried in hte real part, and the tangent trace in the dual part.

This is a useful format because addition and multiplication of these dual numbers results in dual parts that respect differentiation rules w.r.t their real parts, e.g. $(v + \dot{v} \epsilon)(u +  \dot{u} \epsilon) = (vu) + (v  \dot{u}u +  \dot{v} u) \epsilon$. Most significantly, this formulation respects the chain rule, in that if we have $z_j = v_j + D_p v_j \epsilon$ then applying a function $f$ gives $f(z_j) = f(v_j) + f^{\prime} (v_j)D_p v_j$. So defining our dual number in this way we can progress through the intermediate steps of AD and the real and dual parts will give us the function and derivative evaluation naturally.

More detail can be found [here](https://en.wikipedia.org/wiki/Automatic_differentiation)

## How to use
[How do you envision that a user will interact with your package? What should they import? How can they instantiate AD objects? Note: This section should be a mix of pseudo code and text. It should not include any actual operations yet. Remember, you have not yet written any code at this point.]::

Users will be able to access specific individual objects to use for AD but will also have access to a top-level class to perform end-to-end AD. 

#### Automatic differentiation example
```
import autodiff30.autodiff30 as ad

@ad.adfunction(ndim = 2)
def foo(x):
    """
    x is a list-like object of size ndim
    foo is a vector function that returns an array of outputs
    """
    return_list = [ad.sin(x[0]), ad.cos(x[0]*x[1])]
    return return_list

x = [1,1]
foo_at_x = foo(x)
gradient_at_x = foo.grad(x)
```

#### Optimization example
```
import autodiff30.autodiff30 as ad
import autodiff30.optimize as ad_opt

@ad.adfunction(ndim = 2)
def foo(x):
    """
    x is a list-like object of size ndim
    foo is a vector function that returns an array of outputs
    """
    return_list = [ad.sin(x[0]), ad.cos(x[0]*x[1])]
    return return_list
    
initial_guess = [0,0]
min_f, min_inputs = ad_opt.SGD(foo, initial_guess)
```

## Software Organization
[Discuss how you plan on organizing your software package. What will the directory structure look like? What modules do you plan on including? What is their basic functionality? Where will your test suite live? How will you distribute your package (e.g. PyPI with PEP517/518 or simply setuptools)? Other considerations?]::



- What will the directory structure look like?

```  
autodiff30
|   x-- LICENSE
|   x-- README.md
|   x-- pyproject.toml
|   x-- setup.cfg
|   x-- src
|   |   x-- autodiff30
|   |   |   x-- __init__.py
|   |   |   x-- __main__.py
|   |   |   x-- autodiff30
|   |   |   |   x-- __init__.py
|   |   |   |   x-- ad.py
|   |   |   |   x-- dual.py
|   |   |   |   x-- functions.py
|   |   |   x-- optimize
|   |   |   |    x-- __init__.py
|   |   |   |    x-- optimization.py
|   x-- tests
|   |   x-- autodiff30
|   |   |   x-- test_ad.py
|   |   |   x-- test_dual.py
|   |   |   x-- test_functions.py
|   |   x-- optimize
|   |   |   x-- test_optimization.py
|   x-- examples
|       x-- examples.py
|       ...
```

- What modules do you plan on including? What is their basic functionality?

    -  ```ad```: ```adfunction``` main class
    - ```dual```: dual numbers objects and basic arithmetic
    - ```functions```: math functions that can be used as building blocks
    
- Where will your test suite live?

    - The tests in /tests will be executed through GitHub Actions.

- How will you distribute your package (e.g. PyPI with PEP517/518 or simply setuptools)?
    
    - We will distribute it on the PyPI test server.  Then, it can be installed with 
    ``` python -m pip install -i https://test.pypi.org/simple/ autodiff30```
    
- Other considerations?

## Implementation
[Discuss how you plan on implementing the forward mode of automatic differentiation. What are the core data structures? What classes will you implement? What method and name attributes will your classes have? What external dependencies will you rely on? How will you deal with elementary functions like sin, sqrt, log, and exp (and all the others)?]::
[Be sure to consider a variety of use cases. For example, don't limit your design to scalar functions of scalar values. Make sure you can handle the situations of vector functions of vectors and scalar functions of vectors. Don't forget that people will want to use your library in algorithms like Newton's method (among others).]::

1. What classes do we need? Which one will we implement first
    
    We need a class for dual number with all elementary operations implemented: mul, add, radd, rmul, div....The two attributes of that class will be the real and the dual part.Since our implementation of AD entirely relies on dual numbers, we start by implementing the dual class. We will also have a decorator class that uses a dual number implementation to achieve automatic differentiation.
    
2. What are the core data structures? How will you incorporate dual numbers? 
    
    We will incorporate dual numbers through our class dual numbers. Our class will support an $__init__$ method that initializes the real and the dual parts of the dual number. The dual number class will be used in our implementation of AD in the adfunction class.
    
3. What method and name attributes will your classes have?

    The dual class will support methods to compute elementary operations such as addition, subtraction, multiplication, divisions, along their swapped versions: \_\_add\_\_, \_\_radd\_\_, \_\_mul\_\_, \_\_rmul\_\_, \_\_sub\_\_, \_\_rsub\_\_, \_\_pow\_\_, \_\_rpow\_\_. The adfunction class will ave ```__call__``` and grad methods.
    
4. Will you need some graph class to resemble the computational graph in forward mode or maybe later for reverse mode? Note that in milestone 2 you propose an extension for your project, an example could be reverse mode.
    
    For our implementation of AD, we won't be needing a class to resemble the computational graph in forward mode. Since our proposed extension will not involve the reverse mode, we do not need to keep track of the graph in our implementation. We are going to build an optimization application for our package, which should not require a graph.
    
5. Think about how your basic operator overloading template should look like. How will you deal with elementary functions like sin, sqrt, log, and exp (and many others)?
    
    Our implementation relies on dual numbers. However, most of the elementary functions such as sin, cos, sqrt... are coded on numpy, which does not support the type dual. We will therefore have to overload all NumPy functions so that they could take into input dual numbers. Practically, we will not be overloading NumPy functions, rather defining new functions that compute elementary operations (sin, cos, ln...) for dual number inputs. 
    
6. How do you want to handle cases for $f\colon\mathbb{R}^m \rightarrow \mathbb{R} or later f\colon\mathbb{R}^{m} \rightarrow \mathbb{R}^{n}?$ Would it make sense to design a high-level function object to model arbitrary functions f? You could think further and possibly plan for a grad() method or similar in a class that models ff, since computing the gradient (or Jacobian) is an operation that is often required.
    
    Our main class will have a grad method that will handle multivariate inputs and outputs.
    
7. Do you want/need to depend on other libraries? (e.g. NumPy). 
    
    We will import NumPy and rely on it for basic operations on supported type (e.g int, float...).
 

## Licensing
[Licensing is an essential consideration when you create new software. You should choose a suitable license for your project. A comprehensive list of licenses can be found here. The license you choose depends on factors such as what other software or libraries you use in your code (copyleft, copyright). will you have to deal with patents? How can others advertise software that makes use of your code (or parts thereof)?]::
[Briefly motivate your license choice in the milestone1 document and add a LICENSE file to the root of your project.]::

Te key library that will be used in our source code is NumPy, which is issued under a BSD 3-Clause "New" or "Revised" License, which is a generally permissive licence. We are implementing an automatic differentiation library. Many implementations of equivalent thins exist, likely with levels of optimization that ours does not have. AD is also already a part of popular packages like JAX. As such patents will not be something we need to consider. Ours is also not likely to be a huge program. As such we do not believe it necessary to have an especially strong licence or one that enforces copyleft. But given that the project originated within a Harvard class, we do not deem it appropriate that others should be able to advertise software that uses this code. Given that our project will mostly be using NumPy functions, we will use the same licence that they do.

## Feedback
### Milestone 1
 - How to use: I would encourage you to add an example of how you would expect the user to use your optimization feature.
    - We have added pseudo code showing how to use the optimization element of the package in the How To Use section
 - Implementation : I would expect to see a bit more on how you would expect the users to interact with your package. This will also help you to think in detail about the design of your functions, etc.
    - We believe this is covered in the How To Use section
