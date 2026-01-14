import numpy as np

def mod_bisection(f, a, b, tol=1e-6, maxit=100):
    fa = f(a)
    fb = f(b)

    for it in range(1, maxit + 1):
        if abs(fa) < abs(fb):
            c = a + (b - a) / 3.0
        else:
            c = b - (b - a) / 3.0
            
        fc = f(c)
        
        if abs(fc) < np.finfo(float).eps or (b - a) / 2.0 < tol:
            return c, it
            
        if fa * fc < 0:
            b = c
            fb = fc
        else:
            a = c
            fa = fc
            
    return c, maxit




def mod_newton(f, df, d2f, x0, tol=1e-6, maxit=100):
    x = x0
    for it in range(1, maxit + 1):
        fx = f(x)
        dfx = df(x)
        d2fx = d2f(x)
        
        if dfx == 0:
            return x, it
            
        if fx == 0:
            return x, it

        paro = (dfx / fx) - 0.5 * (d2fx / dfx)
        
        if paro == 0:
            return x, it
            
        x_new = x - 1.0 / paro
        
        if abs(x_new - x) < tol:
            return x_new, it
            
        x = x_new
        
    return x, maxit




def mod_secant(f, x0, x1, x2, tol=1e-6, maxit=100):
    pts = [x0, x1, x2]
    
    for it in range(1, maxit + 1):
        xn, xn1, xn2 = pts[0], pts[1], pts[2]
        
        fn = f(xn)
        fn1 = f(xn1)
        fn2 = f(xn2)
        
        if fn1 == 0 or fn == 0 or fn2 == 0:
            return xn2, it
            
        q = fn / fn1
        r = fn2 / fn1
        s = fn2 / fn

        paro = (q - 1) * (r - 1) * (s - 1)
        if paro == 0:
             return xn2, it
             
        arithm = r*(r - q)*(xn2 - xn1) + (1 - r)*s*(xn2 - xn)
        
        x_new = xn2 - arithm / paro
        
        if abs(x_new - xn2) < tol:
            return x_new, it
            
        pts.pop(0)
        pts.append(x_new)
        
    return pts[-1], maxit