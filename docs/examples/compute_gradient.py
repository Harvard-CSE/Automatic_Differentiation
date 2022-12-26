from autodiff30.ad import adfunction

if __name__ == '__main__':
    print("\nComputing gradients \n")
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
        bar is a R2 -->R function.
        x: list of floats, a 2D point at which g is estimated
        """
        return x[0]**2 + x[1]**2
    x = [2.,2.]
    print(f" The gradient of bar :   x |--> x^2 + y^2 is {bar.grad(x)}")
    
