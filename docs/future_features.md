# Future features
## Optimization package
Automatic Differentiation can be highly leveraged in optimizations problems, that is why we will develop a module for optimization in autodiff30. Most of Machine Learning applications rely on the Stochatic Gradient Descent (SGD) algorithm, hence our module will implement it (along with other optimizations algorithms) in order to find the minimum of a given function. 

Let $f : \mathbb{R}^n  \longrightarrow \mathbb{R}^m$, the SGD algorithm starts from a given initial point $\theta_0$ and updates its parameters as follow,

$$ \theta_{t+1} = \theta_t - \eta \nabla_{\theta} f $$
where $\eta$ is often called the learning rate.

### User level
For our end user to use Stochatic Gradient Descent (SGD) algorithm he will simply have to execute the following commands,

```
import autodiff30.autodiff30 as ad
import autodiff30.optimize as ad_opt

@ad.adfunction
def foo(x):
    """
    x is a list-like object of size ndim
    foo is a vector function that returns an array of outputs
    """
    return_list = [ad.sin(x[0]), ad.cos(x[0]*x[1])]
    return return_list
    
initial_guess = [0,0]
min_x = ad_opt.GD(foo, initial_guess)
```
### Structure
Our optimization module would be implemented in the optimize folder where we will develop the algorithms like SGD. We will also add some tests for our optimize module.

```  
autodiff30
|   x-- LICENSE
|   x-- README.md
|   x-- pyproject.toml
|   x-- setup.cfg
|   x-- src
|   |   x-- autodiff30
|   |   |   x-- __init__.py
.   .   .   .
.   .   .   .
.   .   .   .
|   |   |   x-- optimize
|   |   |   |    x-- __init__.py
|   |   |   |    x-- optimization.py
|   x-- tests
.   .   . 
.   .   .  
|   |   x-- optimize
|   |   |   x-- test_optimization.py
|   x-- examples
|       x-- examples.py
|       ...
```

### Other possible algorithms
On top of SGD, others variations leveraging differents techniques (momentum,...) have been developed. One possibility would be to develop some other classical optimization algorithms along with SGD so our end user would have more choice depending on the parameters of his optimization problem.
-   RMSProp    
-	Nesterov
-   Adam
