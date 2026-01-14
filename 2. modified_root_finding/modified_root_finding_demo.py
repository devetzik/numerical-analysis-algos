import numpy as np
from ask1_lib import bisection, newton, secant
from ask2_lib import mod_bisection, mod_newton, mod_secant
import random

def f(x):
    return 54*x**6 + 45*x**5 - 102*x**4 - 69*x**3 + 35*x**2 + 16*x - 4

def df(x):
    return 324*x**5 + 225*x**4 - 408*x**3 - 207*x**2 + 70*x + 16

def d2f(x):
    return 1620*x**4 + 900*x**3 - 1224*x**2 - 414*x + 70

tol = 1e-7
maxit = 100

print("=== 1. Εύρεση Ριζών (Τροποποιημένες Μέθοδοι) ===")

times_ekk = [
    (-1.3, [-1.5, -1.1], [-1.4, -1.3, -1.2]),
    (-0.6, [-0.8, -0.4], [-0.9, -0.8, -0.7]),
    (0.3,  [0.1, 0.5],   [0.1, 0.2, 0.3]),
    (0.5,  [0.4, 0.6],   [0.4, 0.5, 0.6]),
    (1.1,  [0.9, 1.3],   [1.0, 1.1, 1.2])
]

for guess, interval, sec_pts in times_ekk:
    print(f"\n--- Αναζήτηση κοντά στο {guess} ---")
    
    try:
        rb, itb = mod_bisection(f, interval[0], interval[1], tol)
        print(f"Τροπ. Διχοτόμηση: Ρίζα={rb:.7f}, Επαναλήψεις={itb}")
    except ValueError:
        print("Τροπ. Διχοτόμηση: Απέτυχε (μη αλλαγή προσήμου)")

    rn, itn = mod_newton(f, df, d2f, guess, tol)
    print(f"Τροπ. Newton    : Ρίζα={rn:.7f}, Επαναλήψεις={itn}")

    rs, its = mod_secant(f, sec_pts[0], sec_pts[1], sec_pts[2], tol)
    print(f"Τροπ. Τέμνουσα  : Ρίζα={rs:.7f}, Επαναλήψεις={its}")



print("\n=== 2. Έλεγχος Συνέπειας Mod-Bisection (20 εκτελέσεις με τυχαία άκρα) ===\n")

print(f"{'Δοκιμή':<4} | {'Διάστημα':<18} | {'Ρίζα':<10} | {'Επαναλήψεις'}")

for i in range(1, 21):
    a_rand = -1.8 + (random.random() - 0.5) * 0.2
    b_rand = -1.2 + (random.random() - 0.5) * 0.1
    
    if f(a_rand) * f(b_rand) > 0:
        a_rand, b_rand = -1.8, -1.2

    root, it = mod_bisection(f, a_rand, b_rand, tol)
    
    print(f"{i:<6} | [{a_rand:.4f}, {b_rand:.4f}] | {root:.7f} | {it}")




print("\n=== 3. Σύγκριση Ταχύτητας με Τυχαία Inputs (100 εκτελέσεις) ===")

iter_newton, iter_mod_newton = [], []
iter_bis, iter_mod_bis = [], []
iter_sec, iter_mod_sec = [], []

for i in range(100):
    
    x0 = -1.6 + random.random() * 0.5 
    
    _, it1 = newton(f, df, x0, tol, maxit)
    _, it2 = mod_newton(f, df, d2f, x0, tol, maxit)
    iter_newton.append(it1)
    iter_mod_newton.append(it2)

    a_rand = -1.8 + random.random() * 0.4
    b_rand = -1.2 + random.random() * 0.4
    
    _, it1 = bisection(f, a_rand, b_rand, tol, maxit)
    _, it2 = mod_bisection(f, a_rand, b_rand, tol, maxit)
    iter_bis.append(it1)
    iter_mod_bis.append(it2)

    s0 = -1.7 + random.random() * 0.1
    s1 = -1.5 + random.random() * 0.1
    s2 = -1.2 + random.random() * 0.1
    
    _, it1 = secant(f, s0, s1, tol, maxit)
    _, it2 = mod_secant(f, s0, s1, s2, tol, maxit)
    iter_sec.append(it1)
    iter_mod_sec.append(it2)

print(f"\n--- Αποτελέσματα (Μέσος όρος επαναλήψεων) ---\n")
print(f"Κλασσική Newton-Ralphson:  {np.mean(iter_newton):.2f}")
print(f"Τροπ/μένη Newton-Ralphson: {np.mean(iter_mod_newton):.2f}\n")

print(f"Κλασσική Διοχοτόμηση:  {np.mean(iter_bis):.2f}")
print(f"Τροπ/μένη Διχοτόμηση:  {np.mean(iter_mod_bis):.2f}\n")

print(f"Κλασσική Τέμνουσα:     {np.mean(iter_sec):.2f}")
print(f"Τροπ/μένη Τέμνουσα:    {np.mean(iter_mod_sec):.2f}\n")