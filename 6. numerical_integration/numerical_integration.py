import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return np.sin(x)

a = 0
b = np.pi / 2
simeia = 11
n = simeia - 1

I_true = 1.0

print("11 σημεία (10 διαστήματα) στο [0, π/2].")
print(f"Πραγματική Τιμή: {I_true:.6f}\n")

x = np.linspace(a, b, simeia)
y = f(x)

# Τραπεζίου
apot_trap = ((b-a) / (2*n)) * (y[0] + 2 * np.sum(y[1:-1]) + y[-1])

# Simpson
mona = np.sum(y[1:-1:2])
zyga = np.sum(y[2:-1:2])
apot_simp = ((b-a) / (3*n)) * (y[0] + 4 * mona + 2 * zyga + y[-1])

# Αριθμητικό Σφάλμα
err_trap_real = np.abs(I_true - apot_trap)
err_simp_real = np.abs(I_true - apot_simp)

# Β. Θεωρητικό Σφάλμα (Bound)
max_paragogwn = 1.0
err_trap_theo = ((b - a)**3/ (12*n**2)) * max_paragogwn
err_simp_theo = ((b - a)**5 / (180*n**4)) * max_paragogwn

print("Μέθοδος Τραπεζίου:")
print(f"Αποτέλεσμα: {apot_trap:.8f}")
print(f"Αριθμητικό Σφάλμα: {err_trap_real:.8f}")
print(f"Θεωρητικό Φράγμα: {err_trap_theo:.8f}\n")
print("Μέθοδος Simpson:")
print(f"Αποτέλεσμα: {apot_simp:.8f}")
print(f"Αριθμητικό Σφάλμα: {err_simp_real:.8f}")
print(f"Θεωρητικό Φράγμα: {err_simp_theo:.8f}\n")

# Plot
#=======================================================
plt.plot(x, y, 'k-', linewidth=2, label='sin(x)')
plt.scatter(x, y, color='red', zorder=5, label='Επιλεγμένα Σημεία')
plt.title('sin(x) στο [0, π/2] με 11 ισαπέχοντα σημεία')
plt.xlabel('x')
plt.ylabel('sin(x)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
#=======================================================