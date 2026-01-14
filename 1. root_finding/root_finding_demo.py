import numpy as np

def bisection(f, a, b, tol=1e-6, maxit=100):
    fa = f(a)
    fb = f(b)

    for it in range(1, maxit + 1):
        c = 0.5 * (a + b)
        fc = f(c)

        if abs(fc) < np.finfo(float).eps: # e μηχανής
            return c, it

        if 0.5 * (b - a) < tol:
            return c, it

        if fa * fc < 0:
            b = c
            fb = fc
        else:
            a = c
            fa = fc

    return c, it



def newton(f, df, x0, tol=1e-6, maxit=100):
    x = x0

    for it in range(1, maxit + 1):
        fx  = f(x)
        dfx = df(x)

        if dfx == 0:
            return x, it

        x_new = x - fx / dfx

        if abs(x_new - x) < tol:
            return x_new, it

        x = x_new

    return x, maxit



def secant(f, x0, x1, tol=1e-6, maxit=100):
    fn_1 = f(x0)
    fn   = f(x1)
    
    if abs(fn) < np.finfo(float).eps:
        return x1, 0
    
    for it in range(1, maxit + 1):
        if (fn - fn_1) == 0:
            print("Τερματισμός. Παρονομαστής=0. f(x_n)==f(x_n-1)")
            return x1, it

        x_new = x1 - fn * (x1 - x0) / (fn - fn_1)
        
        if abs(x_new - x1) < tol:
            return x_new, it
        
        x0 = x1
        fn_1 = fn
        x1 = x_new
        fn = f(x_new)
        
    return x1, maxit