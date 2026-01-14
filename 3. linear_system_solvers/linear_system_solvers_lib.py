import numpy as np

def lu_decomposition(A):
    n = len(A)
    U = A.copy().astype(float)
    L = np.zeros((n, n))
    P = np.eye(n)

    for k in range(n - 1):
        max_deiktis = np.argmax(np.abs(U[k:, k])) + k
        
        if max_deiktis != k:
            U[[k, max_deiktis]] = U[[max_deiktis, k]]
            P[[k, max_deiktis]] = P[[max_deiktis, k]]
            L[[k, max_deiktis]] = L[[max_deiktis, k]]

        for i in range(k + 1, n):
            pol = U[i, k] / U[k, k]
            L[i, k] = pol
            U[i, k:] -= pol * U[k, k:]
            
    np.fill_diagonal(L, 1)
    
    return P, L, U


def solve_lu(P, L, U, b):
    Pb = np.dot(P, b)
    n = len(b)
    
    y = np.zeros(n)
    for i in range(n):
        y[i] = Pb[i] - np.dot(L[i, :i], y[:i])
        
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (y[i] - np.dot(U[i, i+1:], x[i+1:])) / U[i, i]
        
    return x


def cholesky(A):
    n = len(A)
    L = np.zeros((n, n))

    for i in range(n):
        for j in range(i + 1):
            sum_val = np.dot(L[i, :j], L[j, :j])
            
            if i == j: # Διαγώνια
                val = A[i, i] - sum_val
                L[i, j] = np.sqrt(val)
            else: # Μη διαγώνια
                L[i, j] = (A[i, j] - sum_val) / L[j, j]
    return L



def gauss_seidel(A, b, x0=None, tol=1e-5, maxit=1000):
    n = len(b)
    x = np.zeros(n) if x0 is None else x0.copy()
    
    for it in range(1, maxit + 1):
        x_old = x.copy()
        
        for i in range(n):
            s1 = np.dot(A[i, :i], x[:i])
            s2 = np.dot(A[i, i+1:], x_old[i+1:])
            
            x[i] = (b[i] - s1 - s2) / A[i, i]
            
        error = np.max(np.abs(x - x_old))
        if error < tol:
            return x, it
            
    print("Η μέθοδος Gauss-Seidel δεν συνέκλινε.")
    return x, maxit



def create_system(n):
    A = np.zeros((n, n))
    b = np.ones(n)
    b[0] = 3
    b[n-1] = 3
    
    for i in range(n):
        A[i, i] = 5
        if i > 0:
            A[i, i-1] = -2
        if i < n - 1:
            A[i, i+1] = -2
            
    return A, b