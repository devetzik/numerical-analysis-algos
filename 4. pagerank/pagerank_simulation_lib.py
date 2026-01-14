import numpy as np

class PRSolver:
    def __init__(self, pin_geit, q=0.15):
        self.A = pin_geit.astype(float)
        self.n = self.A.shape[0]
        self.q = q
        
    def create_gm(self):
        G = np.zeros((self.n, self.n))
        
        # Υπολογισμός βαθμού εξόδου (out-degree) για κάθε κόμβο
        n_j = np.sum(self.A, axis=1)

        for j in range(self.n): # Από τον κόμβο j
            for i in range(self.n): # Στον κόμβο i
                G[i, j] = self.q / self.n + (1 - self.q) * (self.A[j, i] / n_j[j])
                
        return G

    def power_method(self, tol=1e-7, max_iter=100):
        G = self.create_gm()
        
        # Αρχικό διάνυσμα πιθανοτήτων (ομοιόμορφη κατανομή)
        p = np.ones(self.n) / self.n
        
        for iteration in range(max_iter):
            p_new = np.dot(G, p)
            
            # Κανονικοποίηση (αν και το G είναι στοχαστικός, καλό είναι να γίνεται)
            p_new = p_new / np.sum(p_new)
            
            # Έλεγχος σύγκλισης (L1 norm)
            if np.linalg.norm(p_new - p, 1) < tol:
                return p_new, iteration + 1
            
            p = p_new
            
        return p, max_iter