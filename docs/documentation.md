# Documentation

## Introduction
[Describe the problem the software solves and why it's important to solve that problem.]::

Automatic differentiation (AD or autodiff throughout) is a set of techniques for numerically evaluating a gradient. It achieves this by breaking the function-to-be-differentiated into a set of elementary operations and using the chain rule. We can contrast this with two other flavours of differentiation: symbolic and finite-differences. Symbolic differentiation aims to calculate a symbolic mathematical expression for the derivative of a particular function, while finite-differences numerically approximates a derivative at a particular point. Compared to the latter of these, AD achieves machine precision accuracy without sacrificing computational complexity and does not slow down as much when input dimensionality is high [<sup>[1]</sup>](https://arxiv.org/abs/1502.05767). Compared to symbolic differentation, AD is not limited to closed-form functions, and can handle loops, recursion and control flow.

Not only is autodiff powerful compared to other differentiation techniques, but it is already used widely. Gradient-based optimization, using the negative gradient of a function to step towards its local minimum, is pivotal to deep learning implementations. The importance of such optimization led us to develop an optimization feature as part of autodiff30, which provides a simple user interface and uses AD in the backend. These have come to dominate many machine learning domains including computer vision and natural language processing. Implementations of computational methods for probabilistic inference, such as Hamiltonian Monte Carlo and Variational Inference, have also come to rely on automatic differentiation for the computation of necessary gradients.


## Background
[Describe (briefly) the mathematical background and concepts as you see fit. You do not need to give a treatise on automatic differentiation or dual numbers. Just give the essential ideas (e.g. the chain rule, the graph structure of calculations, elementary functions, etc). Do not copy and paste any of the lecture notes. We will easily be able to tell if you did this as it does not show that you truly understand the problem at hand.]::

#### Automatic Differentiation
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

#### Optimization
Optimization means selecting the best option from a set of possible options, according to some criterion. Finding the minimum or maximum of some function is a classic example of this, e.g. finding the $(x,y)$ that maximize  $f(x,y)=(x^2 + y^2)+6$, giving $(x^*,y^*) = (0,0)$. This is a trivially simple example, with a simple closed form expression for $f$. But in general we will not have such a form. For example in neural networks, we aim to minimize the loss function with respect to the network weights and biases, of which there can be billions forming a highly non-linear function. To tackle this we proceed in an iterative manner, taking small steps from some starting point in the function space and heading in the directions we believe will lead us to a maximum or minimum. We will explain two algorithms to achieve this next.

#### Gradient Descent
Assume we have a function $f(\mathbf{x})$ defined on an n-dimensional space, such that $\mathbf{x}$ has dimension n. Assuming $f$ is differentiable, then the n-dimensional gradient of $f$ always points in the direction of greatest increase of $f$ (and similarly the negative of the gradient points in the direction of greatest decrease). So, for some starting point $\mathbf{x_0}$ we can move iteratively towards a minimum using the equation $\mathbf{x_{i+1}} = \mathbf{x_{i}} - \gamma f(\mathbf{x_i})$, where $\gamma$ is a tunable parameter called the learning rate that determines the size of step we take at each iteration. Note that this only guarantees that we will find a local minima, not a global minima, unless additional constraints are imposed on $f$.

#### Adam
Adam optimization looks to improve upon some of the drawbacks of gradient descent, and is one of the most common algorithms used in training neural networks. First, rather than actually computing the gradient from the entire data set, it estimates it through random sampling. This is useful as taking the gradient of high-dimensional, complex functions can be intractable or very slow. Second, Adam uses exponential moving averages of previous gradients to include a sense of "momentum" into the changes. Gradient descent suffers from the drawback that if it gets into areas of parameter space where the gradient is very small (e.g. at a minimum, or at a saddle/inflexion point) it tends to get stuck there as the changes become small. The "momentum" in Adam counteracts this. More details can be found here https://arxiv.org/abs/1412.6980.

## How to use
[How do you envision that a user will interact with your package? What should they import? How can they instantiate AD objects? Note: This section should be a mix of pseudo code and text. It should not include any actual operations yet. Remember, you have not yet written any code at this point.]::

Please see HowToUse.ipynb for details on installation, basic and advanced use of the package.

## Software Organization
[Discuss how you plan on organizing your software package. What will the directory structure look like? What modules do you plan on including? What is their basic functionality? Where will your test suite live? How will you distribute your package (e.g. PyPI with PEP517/518 or simply setuptools)? Other considerations?]::

- Our directory is structured as below

```  
team30
|   x-- LICENSE
|   x-- README.md
|   x-- pyproject.toml
|   x-- src
|   |   x-- autodiff30
|   |   |   x-- __init__.py
|   |   |   x-- ad.py
|   |   |   x-- dual.py
|   |   |   x-- functions.py
|   |   |   x-- optimization.py
|   x-- tests
|   |   x-- test_dual_number.py
|   |   x-- test_math_functions.py
|   |   x-- test_optimize.py
|   |   x-- test_user_level.py
|   |   x-- run_tests.sh
|   |   x-- check_coverage.sh
|   x-- docs
|   |   x-- README.md
|   |   x-- documentation.md
|   |   x-- documentation.pdf
|   |   x-- examples
|   |   |   x-- HowToUse.ipynb
|   |   |   x-- compute_gradient.py
|   |   |   x-- how_to_use.py
|   |   |   x-- optimize.py
|   |   x-- images
|   |   |   ...
|   |   x-- milestones
|   |   |   ...
```

- Key modules and data structures are detailed in HowToUse.ipynb, but the most important is ad.py which contains the the ```adfunction``` decorator and the ```adstruc``` class. The decorator transforms a user-defined function into an ```adstruc``` object, which has a ```grad``` method that implements automatic differenation. dual.py contains our dual number implementation, implementing multiplication, addition comparison etc. for dual numbers, and functions.py contains mathematical functions that are used to compose a user function.
    
- All tests live in the test directory and written to be used with pytest. Github workflows in the directory run the tests at every push in custom containers (see team30/.github/workflows for the YAML).

- The package is distributed through PyPI (https://pypi.org/project/autodiff30/), both consumers and developers should use the package from there (instructions in HowToUse.ipynb).

## Implementation
[Discuss how you plan on implementing the forward mode of automatic differentiation. What are the core data structures? What classes will you implement? What method and name attributes will your classes have? What external dependencies will you rely on? How will you deal with elementary functions like sin, sqrt, log, and exp (and all the others)?]::
[Be sure to consider a variety of use cases. For example, don't limit your design to scalar functions of scalar values. Make sure you can handle the situations of vector functions of vectors and scalar functions of vectors. Don't forget that people will want to use your library in algorithms like Newton's method (among others).]::

 - Core data structures and classes
    - We do not use any parituclarly exotic data structures - most of our implementation uses built-in python objects like lists and our own ```DualNumber``` class
    - The key elements of our implementation as described above are the ```adstruc``` class, which actually implements the automatic differentation, and the ```adfunction``` decorator which just passes a user defined function to the  ```adstruc``` and returns that object

- Important attributes
    - The key method is ```adstruc.grad()``` which uses dual numbers to implement automatic differentation. ```adstruc.__call__()``` allows the user-defined function to still be called after the decorator is applied, and that function in its original form and datatype can be accessed by ```adstruc.f```, though there is no reason to due to the ```__call__``` method
    - We handle multivariate inputs and outputs in ```adstruc.grad()``` by checking the type and dimensions of our input and using a loop with a seed vector to create the Jacobian matrix
    - The ```DualNumber``` class, which can be accessed by users for uses other than automatic differentation, has key attributes ```real``` and ```dual```.

- Dependencies
    - In the backend our implementation relies heavily on Numpy, this is our key and only real dependency

- Elementary functions
    - We implement elementary trigonometric, logarithmic etc. functions that support dual numbers. We use Numpy to correctly manipulate the real and dual parts of the input dual number. These individual functions are accessible to the user to use in defining their own functions that they want to differentiate

## Extension: Optimization

Automatic Differentiation can be highly leveraged in optimizations problems, that is why we have developed a module for optimization in autodiff30. Most Machine Learning applications rely on the Gradient Descent (GD) algorithm, hence our module will implement it (along with other optimization algorithms) in order to find the minimum of a given function. Users can of course use autodiff30 to define their own optimizers too!

Let $f : \mathbb{R}^n  \longrightarrow \mathbb{R}^m$, the GD algorithm starts from a given initial point $\theta_0$ and updates its parameters as follows,
$$ \theta_{t+1} = \theta_t - \eta \nabla_{\theta} f $$ where $\eta$ is often called the learning rate.

We define two functions, Adam and GD, which use the automatic differentation to take the gradient required in the update rule above. See the Background section and HowToUse.ipynb for more information on the mathematical underpinnings and user-level use of these functions.

## Future
- The key thing we would like to add in future work is reverse mode for automatic differentiation. As noted in Background, neural networks, one of the key motivating factors for interest in automatic differentiation, require reverse mode to work efficiently as the input space of wights and biases is much larger than the one-dimensional output loss
- We also believe there could be performance improvements that could be made throguh increased use of vectorization
- We would also like to investigate ways to visualize the calculations happening under the hood for ease of interpretability (see below)
- Finally, inspired by much of the work from groups like Deepmind, we would like to investigate additional cutting-edge applications in the physical sciences

## Broader Impact and Inclusivity Statement
- Impacts of AD likely to stem from its use in other applications
- Neural networks are a key use of AD, and they can be used maliciously

We believe the impacts of automatic differentiation are likely to stem from its use in various different applications. For example, neural networks, a key user of automatic differentiation, are becoming increasingly ubiquitous. However, it is well documented that their application without proper consideration or governance can be harmful. A clear arena where there misuse is harmful is in medicine, where clearly an incorrect diagnosis or treatment suggestion could seriously impact poeples lives. In this case, interpretability is very useful, which filters down to the design of our package and motivates us to consider visualization in future iterations of the package. This touches on one of the key ethical considerations in any software that might use our package - how it interacts with people and whether that is fair. It is our belief and intention that software should enhance people's lives, rather than replace or reduce their livelihoods. We would like to investigate if any licencing changes might enable us to control more tightly how downstream users deploy our package to maintain alignment with our beliefs.

We acknowledge that given the origin of this project in a Harvard class and despite our best efforts, there are likely shortcuts we have taken and assumptions we have made about the level of expertise and background of users of and contributors to our package - namely that they understand all the mathematical and programming nomenclature we have used and have the time and core skills to contribute if they wanted. We have not, for example, provided any links to help users get started with things like Git if they are unfamiliar. As only the core developers have so far been submitting and accepting pull requests, these may have unclear names and descriptions, as they can be easily clarified in our whatsapp group if required. This practice would need to change to open the developer pool more widely. A similar change likely needs to take place with our commit messages. We also only have our documentation in English - which we would like to address in future iterations using Sphinx for translation.

