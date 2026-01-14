import numpy as np
import matplotlib.pyplot as plt

np.set_printoptions(precision=6, suppress=True)

#====================Lagrange=========================
def lagrange(x_points, y_points, x_val):
    n = len(x_points)
    sum = 0.0
    
    for i in range(n):
        xi, yi = x_points[i], y_points[i]
        term = yi
        for j in range(n):
            if i != j:
                xj = x_points[j]
                term = term * (x_val - xj) / (xi - xj)
        sum += term
        
    return sum
#=====================================================

#=============Ελάχιστα Τετράγωνα======================
def least_squares(x_points, y_points, x_val):
    n = len(x_points)
    deg = 5
    
    A = np.zeros((n, deg + 1))
    for j in range(deg + 1):
        A[:, j] = x_points ** j
        
    ATA = np.dot(A.T, A)
    ATb = np.dot(A.T, y_points)
    
    lisi = np.linalg.solve(ATA, ATb)

    value = 0.0
    for i, c in enumerate(lisi):
        value += c * (x_val ** i)
        
    return value
#=====================================================

#=====================Splines=========================
def calc_spl(x, y):
    n = len(x) - 1
    h = np.diff(x)
    
    a = y[:-1]
    
    num_in = n - 1 
    A = np.zeros((num_in, num_in))
    r = np.zeros(num_in)
    
    for i in range(num_in):
        real_i = i + 1 
        
        A[i, i] = 2 * (h[real_i-1] + h[real_i])
        
        if i < num_in - 1:
            A[i, i+1] = h[real_i]
            
        if i > 0:
            A[i, i-1] = h[real_i-1]
            
        t1 = (3 / h[real_i]) * (y[real_i+1] - y[real_i])
        t2 = (3 / h[real_i-1]) * (y[real_i] - y[real_i-1])
        r[i] = t1 - t2

    c_in = np.linalg.solve(A, r)

    c = np.concatenate(([0], c_in, [0]))
    
    b = np.zeros(n)
    d = np.zeros(n)
    
    for i in range(n):
        b[i] = (y[i+1] - y[i]) / h[i] - h[i] * (c[i+1] + 2 * c[i]) / 3
        d[i] = (c[i+1] - c[i]) / (3 * h[i])
        
    return a, b, c[:-1], d

def eval_spl(x_train, coeffs, x_val):
    a, b, c, d = coeffs
    n = len(x_train) - 1
    
    if x_val <= x_train[0]:
        i = 0
    elif x_val >= x_train[-1]:
        i = n - 1
    else:
        for k in range(n):
            if x_train[k] <= x_val <= x_train[k+1]:
                i = k
                break
                
    dx = x_val - x_train[i]
    value = a[i] + b[i]*dx + c[i]*(dx**2) + d[i]*(dx**3)
    
    return value
#=====================================================


#=====================================================
x_train = np.linspace(-np.pi, np.pi, 10)
y_train = np.sin(x_train)

print("=== Άσκηση 5: Προσέγγιση Ημιτόνου ===\n")
print("Επιλέχθηκαν 10 ισαπέχοντα σημεία στο [-π, π]:")
print(x_train)
print("\nΤιμές sin(x) στα σημεία αυτά:")
print(y_train)

cf = calc_spl(x_train, y_train)

x_test = np.linspace(-np.pi, np.pi, 200)
y_true = np.sin(x_test)

y_lag = []
y_spl = []
y_lst = []

for val in x_test:
    # Lagrange
    result = lagrange(x_train, y_train, val)
    y_lag.append(result)

    # Spline
    result = eval_spl(x_train, cf, val)
    y_spl.append(result)

    # Ελάχιστα Τετράγωνα
    result = least_squares(x_train, y_train, val)
    y_lst.append(result)

# Σφάλματα
err_lag = np.abs(y_true - y_lag)
err_spl = np.abs(y_true - y_spl)
err_lst = np.abs(y_true - y_lst)

# Μέγιστα σφάλματα
max_err_lag = np.max(err_lag)
max_err_spl = np.max(err_spl)
max_err_lst = np.max(err_lst)

print(f"\n--- Αποτελέσματα Ακρίβειας (σε 200 σημεία) ---")
print(f"Lagrange: Max Error = {max_err_lag:.2e}")
print(f"Splines: Max Error = {max_err_spl:.2e}")
print(f"Least Squares: Max Error = {max_err_lst:.2e}")




#=====================================================
plt.figure(figsize=(12, 10))

plt.subplot(2, 1, 1)
plt.plot(x_test, y_true, 'k-', linewidth=2, label='sin(x)')
plt.plot(x_test, y_lag, '--', label='Lagrange')
plt.plot(x_test, y_spl, ':', label='Cubic Spline')
plt.plot(x_test, y_lst, '-.', label='Least Squares')
plt.scatter(x_train, y_train, color='red', zorder=5, label='Επιλεγμένα Σημεία')
plt.title('Σύγκριση Μεθόδων Προσέγγισης')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(2, 1, 2)
plt.plot(x_test, err_lag, label='Lagrange Error')
plt.plot(x_test, err_spl, label='Spline Error')
plt.plot(x_test, err_lst, label='Least Squares Error')
plt.yscale('log') 
plt.title('Απόλυτο Σφάλμα Προσέγγισης')
plt.xlabel('x')
plt.ylabel('|Error|')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()