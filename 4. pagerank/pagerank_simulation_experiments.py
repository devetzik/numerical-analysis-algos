import numpy as np
from ask4_lib import PRSolver
np.set_printoptions(precision=6, suppress=True, linewidth=200)

A_ekf=np.array([
        [0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0]
    ], dtype=float)

print("=== Άσκηση 4: PageRank ===\n")



print("--- 2. Επαλήθευση Αρχικού Γραφήματος (q=0.15) ---\n")
solver = PRSolver(A_ekf, q=0.15)
p, iters = solver.power_method()

print(f"Σύγκλιση σε {iters} επαναλήψεις.\n")
print("Υπολογισμένο PageRank:\n", p)
print("\nTop 5 Σελίδες:")
theseis = np.argsort(p)[::-1]
for i in theseis[:5]:
    print(f"Page {i+1}: {p[i]:.7f}")



print("\n\n--- 3. Τροποποίηση Γραφήματος (Βελτίωση Σελίδας 5) ---\n")

A_mod = A_ekf.copy()
A_mod[12, 4] = 1 
A_mod[14, 4] = 1
A_mod[10, 4] = 1
A_mod[13, 4] = 1
A_mod[4, 0] = 0

solver_mod = PRSolver(A_mod, q=0.15)
p_mod, _ = solver_mod.power_method()

print(f"PageRank Σελίδας 5 (Αρχικό):      {p[4]:.6f}")
print(f"PageRank Σελίδας 5 (Βελτιωμένο):  {p_mod[4]:.6f}")


print("\n\n--- 4. Ευαισθησία στο q ---")
for q_val in [0.02, 0.6]:
    s = PRSolver(A_mod, q=q_val)
    p_q, _ = s.power_method()
    print(f"\nΓια q = {q_val}:")
    thes = np.argsort(p_q)[::-1]
    print("Top 3 Pages:", [int(i)+1 for i in thes[:3]])
    print(f"PageRank Σελίδας 5: {p_q[4]:.6f}")



print("\n\n--- 5. Ανταγωνισμός Σελίδα 11 vs 10 ---\n")
A_5 = A_ekf.copy()

s1 = PRSolver(A_5)
p1, _ = s1.power_method()
print(f"Αρχικά: Rank(10)={p1[9]:.6f}, Rank(11)={p1[10]:.6f}")

A_5[7, 10] = 3
A_5[11, 10] = 3

s2 = PRSolver(A_5)
p2, _ = s2.power_method()
print(f"Μετά : Rank(10)={p2[9]:.6f}, Rank(11)={p2[10]:.6f}")




print("\n\n--- 6. Διαγραφή Σελίδας 10 ---\n")

A_del = A_ekf.copy()
A_del = np.delete(A_del, 9, axis=0)
A_del = np.delete(A_del, 9, axis=1)

s_del = PRSolver(A_del, q=0.15)
p_del, _ = s_del.power_method()

print(f"{'Σελίδα':<5} | {'Πρίν':<10} | {'Μετά':<10} | {'Διαφορά':<10}")
print("-" * 50)

for i in range(15):
    if i == 9:
        continue
    
    prin = p[i]
    
    if i < 9 :
        deiktes_meta = i 
    else:
        deiktes_meta = i - 1

    meta = p_del[deiktes_meta]
    
    change = (meta - prin)/prin*100
    
    print(f"{i+1:<5} | {prin:.6f}   | {meta:.6f}   | {change:+.2f}%")