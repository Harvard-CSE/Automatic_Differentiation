from autodiff30.ad import adfunction
from autodiff30.optimize.optimization import GD
from autodiff30.optimize.optimization import Adam

if __name__ == '__main__':

    ###########################################
    ########### AUTO-DIFF #####################
    ###########################################
    
    print("\nComputing gradients with autodiff30\n")
    @adfunction
    def foo(x):
        """
        foo is a R -->R function.
        x: float, at which f is estimated
        """
        return x**2
    x = 2.
    print(f" The derivative of foo : x |--> x^2 is {foo.grad(x)}")
    
    @adfunction
    def bar(x):
        """
        bar is a R2 -->R convex function.
        x: list of floats, a 2D point at which g is estimated
        """
        return x[0]**2 + x[1]**2
    x = [2.,2.]
    print(f" The gradient of bar :   x |--> x^2 + y^2 is {bar.grad(x)}\n")
    
    ###########################################
    ########### OPTIMIZATION ##################
    ###########################################
    
    print("\nOptimizing functions using autodiff30\n")
    
    @adfunction
    def f(x):
        """
        f is a R -->R function.
        x: float, at which f is estimated
        """
        return x**2
    x0 = 1.
    res1 = GD(f, x0)
    res2 = Adam(f, x0)

    print("\nMinimizing f: R -->R\n")
    print(f" Solution for minimizing f with x0 = {x0} and algorithm GD is {res1}\n")
    print(f" Solution for minimizing f with x0 = {x0} and algorithm Adam is {res2}\n")

    @adfunction
    def g(x):
        """
        g is a R2 -->R convex function.
        x: list of floats, a 2D point at which g is estimated
        """
        return x[0]**2 + x[1]**2

    x0 = [1.,1.]
    res1 = GD(g, x0)
    res2 = Adam(g, x0)
    print("\nMinimizing g: R2 -->R\n")
    print(f" Solution for minimizing g with x0 = {x0} and algorithm GD is {res1}\n")
    print(f" Solution for minimizing g with x0 = {x0} and algorithm Adam is {res2}\n")
    
    


