import numpy as np
import matplotlib.pyplot as plt

np.set_printoptions(precision=4, suppress=True)

def least_squares(x_points, y_points, x_val, deg):
    n = len(x_points)
    
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

days = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) 

y_aktor = np.array([5.35, 5.37, 5.39, 5.35, 5.39, 5.38, 5.35, 5.29, 5.34, 5.32])
y_titan = np.array([41.50, 41.55, 41.80, 41.55, 41.95, 41.55, 40.70, 39.90, 40.55, 41.65])

real_val_28_5_aktor = 5.34
real_val_28_5_titan = 41.85

print("=== Άσκηση 7: Πρόβλεψη Μετοχών Aktor & Titan) ===")
print(f"Ημερομηνία Πρόβλεψης: 28/05/2025")
print(f"Δεδομένα εκπαίδευσης: 14/05 - 27/05 (10 ημέρες)\n")

stocks = [
    ("Aktor (AKTR)", y_aktor, real_val_28_5_aktor), 
    ("Titan (TITC)", y_titan, real_val_28_5_titan)
]
degrees = [2, 3, 4]
colors = ['blue', 'orange', 'green']

plt.figure(figsize=(14, 12))

for idx, (name, prices, real_val) in enumerate(stocks):
    print(f"--- Ανάλυση για {name} ---")
    print(f"Πραγματική τιμή 28/5: {real_val}")
    
    plt.subplot(2, 1, idx + 1)
    
    plt.scatter(days, prices, color='black', label='Ιστορικά (14/5-27/5)')
    plt.scatter(11, real_val, color='red', marker='*', s=150, zorder=10, label=f'Πραγματικό 28/5 ({real_val})')
    
    for i, deg in enumerate(degrees):
        x_line = np.linspace(1, 16, 100)
        y_line = []
        for val in x_line:
            res = least_squares(days, prices, val, deg)
            y_line.append(res)
        
        plt.plot(x_line, y_line, color=colors[i], linestyle='--', 
                 label=f'Πολυώνυμο βαθμού {deg}')
        
        print(f"  [Βαθμός {deg}]")
        pred_28 = least_squares(days, prices, 11, deg)
        pred_04 = least_squares(days, prices, 16, deg)
        
        diff = abs(pred_28 - real_val)
        print(f"    Πρόβλεψη 28/5: {pred_28:.4f} (Απόκλιση: {diff:.4f})")
        print(f"    Πρόβλεψη 04/06: {pred_04:.4f}")
        
        plt.plot(11, pred_28, marker='o', color=colors[i], markersize=6)
        plt.plot(16, pred_04, marker='x', color=colors[i], markersize=8)

    plt.title(f'Πρόβλεψη Τιμής {name}')
    plt.xlabel('Ημέρες (1=14/05 ... 10=27/05, 11=28/05, 16=04/06)')
    plt.ylabel('Τιμή Κλεισίματος (€)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    print("")

plt.tight_layout()
plt.show()