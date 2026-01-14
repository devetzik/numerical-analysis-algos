import numpy as np
from ask3_lib import lu_decomposition, solve_lu, cholesky, gauss_seidel, create_system

print("\n=== Ερώτημα 1: PA=LU ===\n")
A1 = np.array([
    [2, 1, 1],
    [4, 3, 3],
    [8, 7, 9]
], dtype=float)
b1 = np.array([4, 10, 24], dtype=float)

print("Πίνακας A:\n", A1)
P, L, U = lu_decomposition(A1)
print("\nP:\n", P)
print("L:\n", L)
print("U:\n", U)

x_lu = solve_lu(P, L, U, b1)
print("\nΛύση συστήματος (LU): x =", x_lu)



print("\n\n\n=== Ερώτημα 2: Cholesky ===\n")
A2 = np.array([
    [4, 12, -16],
    [12, 37, -43],
    [-16, -43, 98]
], dtype=float)

print("Συμμετρικός Πίνακας A:\n", A2)
L_chol = cholesky(A2)
print("\nΠίνακας L (Cholesky):\n", L_chol)



print("\n\n\n=== Ερώτημα 3: Gauss-Seidel ===")

print(f"\n--- Επίλυση για n = 10 ---")
A1, b1 = create_system(10)
x_gs1, epan1 = gauss_seidel(A1, b1, tol=1e-5)

print(f"Αριθμός Επαναλήψεων: {epan1}")
print("Λύση x:", x_gs1)

print(f"\n--- Επίλυση για n = 5000 ---")
A2, b2 = create_system(5000)
x_gs2, epan2 = gauss_seidel(A2, b2, tol=1e-5)

print(f"Αριθμός Επαναλήψεων: {epan2}")
print("Λύση x (πρώτα 5):", x_gs2[:5])
print("Λύση x (τελευταία 5):", x_gs2[-5:])