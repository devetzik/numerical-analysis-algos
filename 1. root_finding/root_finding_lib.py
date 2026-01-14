import numpy as np
from ask1_lib import bisection, newton, secant

def f(x):
    return 14*x*np.exp(x-2) - 12*np.exp(x-2) \
           - 7*x**3 + 20*x**2 - 26*x + 12

def df(x):
    return (2 + 14*x)*np.exp(x-2) \
           - 21*x**2 + 40*x - 26

tol = 1e-6
maxit = 100

# Ρίζα 1 = 0.8
diastimata1 = [
    (0.6, 1.2),
    (0.7, 1.1),
    (0.5, 1.3),
    (0.75, 1.0),
    (0.8, 1.2),
    (0.65, 1.15),
    (0.55, 1.25),
    (0.72, 1.05),
    (0.78, 1.22),
    (0.68, 1.18),
]

print("\n---- Διχοτόμηση για ρίζα κοντά στο 0.8 ----")
for k, (a, b) in enumerate(diastimata1, start=1):
    root, it = bisection(f, a, b, tol, maxit)
    print(f"Δοκιμή {k:2d}: διάστημα [{a:.2f}, {b:.2f}] "
          f"--> Ρίζα = {root:.6f}, Επαναλήψεις = {it}")


# Ρίζα 2 = 2
diastimata2 = [
    (1.8, 2.9),
    (1.7, 2.5),
    (1.6, 2.4),
    (1.9, 2.3),
    (1.75, 2.2),
    (1.85, 2.4),
    (1.65, 2.35),
    (1.8, 2.2),
    (1.7, 2.1),
    (1.6, 2.0),
]

print("\n---- Διχοτόμηση για ρίζα κοντά στο 2.000 ----")
for k, (a, b) in enumerate(diastimata2, start=1):
    root, it = bisection(f, a, b, tol, maxit)
    print(f"Δοκιμή {k:2d}: διάστημα [{a:.2f}, {b:.2f}] "
          f"--> Ρίζα = {root:.6f}, Επαναλήψεις = {it}")



# Ρίζα 1 = 0.8
arxika1 = [0.2, 0.3, 0.4, 0.5, 0.6,
             0.7, 0.8, 0.9, 1.0, 1.1]

print("\n\n---- Newton-Raphson για ρίζα κοντά στο 0.8 ----")
for k, x0 in enumerate(arxika1, start=1):
    root, it = newton(f, df, x0, tol, maxit)
    print(f"Δοκιμή {k:2d}: x0 = {x0:.1f} --> ρίζα = {root:.6f}, επαναλήψεις = {it}")

# Ρίζα 2 = 2
arxika2 = [1.5, 1.6, 1.7, 1.8, 1.9,
             2.1, 2.2, 2.3, 2.4, 2.5]

print("\n---- Newton-Raphson για ρίζα κοντά στο 2 ----")
for k, x0 in enumerate(arxika2, start=1):
    root, it = newton(f, df, x0, tol, maxit)
    print(f"Δοκιμή {k:2d}: x0 = {x0:.1f} --> ρίζα = {root:.6f}, επαναλήψεις = {it}")



# Ρίζα 1 = 0.8
zevgi1 = [
    (0.7, 0.8), (0.8, 0.9), (0.6, 0.7), (0.9, 1.0), (0.5, 0.6),
    (1.0, 1.1), (0.75, 0.85), (0.85, 0.95), (0.65, 0.75), (0.8, 1.0)
]

print("\n\n---- Μέθοδος Τέμνουσας για ρίζα κοντά στο 0.8 ----")
for k, (x0, x1) in enumerate(zevgi1, start=1):
    root, it = secant(f, x0, x1, tol, maxit)
    print(f"Δοκιμή {k:2d}: x0={x0}, x1={x1} --> Ρίζα = {root:.6f}, Επαναλήψεις = {it}")

# Ρίζα 2 = 2
zevgi2 = [
    (1.8, 1.9), (2.1, 2.2), (1.9, 2.1), (1.7, 1.8), (2.2, 2.3),
    (1.5, 1.6), (2.4, 2.5), (1.8, 2.2), (1.95, 2.05), (1.6, 1.7)
]

print("\n---- Μέθοδος Τέμνουσας για ρίζα κοντά στο 2.0 ----")
for k, (x0, x1) in enumerate(zevgi2, start=1):
    root, it = secant(f, x0, x1, tol, maxit)
    print(f"Δοκιμή {k:2d}: x0={x0}, x1={x1} --> Ρίζα = {root:.6f}, Επαναλήψεις = {it}")