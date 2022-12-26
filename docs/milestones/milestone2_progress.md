# Milestone 2B

## Task assignment
1. Setting up package infrastructure (configuration - not python)- Elie
2. Writing the decorator ```ad``` class - Sam & Elie
3. Writing the elemental operations (exp and trig first) - Lea & Jason
4. Writing the dual number class - Lea
5. Writing test suite - Raphael (decorator, dual) & Sam (elemental, workflows)
6. Writing the future features note - Jason

### User level
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

### Decorator function
```
class adfunction:
    def __init__(self, ndim : int) -> None:
        pass

    def __call__(self, f : func) -> func return type:
        return adstruc

class adstruc:
    def __init__()

    def __call__()

    def __grad__()

# Must follow the "how to use" example
```

## What was completed
- Raphael has been working on implementing the base class adfunction  
- Lea and Jason have started writing function overloading for specific operations in the dual class: multiplication, rmul, addition, subtraction, division, exponential, sinus, cosinus...
- Elie and Sam have been writing and deploying a test suite. 



