import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Fonction non différentiable à minimiser (somme des valeurs absolues)
def f(x):
    return np.abs(x[0]) + np.abs(x[1])

# Sous-gradient de la fonction
def subgradient(x):
    grad = np.zeros_like(x)
    for i in range(len(x)):
        if x[i] > 0:
            grad[i] = 1
        elif x[i] < 0:
            grad[i] = -1
        else:
            grad[i] = np.random.uniform(-1, 1)  # Sous-gradient arbitraire pour x_i = 0
    return grad

# Sous-gradient avec pas constant
def subgradient_constant_step(initial_x, learning_rate, max_iters):
    x = np.array(initial_x)
    history = [x]
    for _ in range(max_iters):
        grad = subgradient(x)
        x = x - learning_rate * grad
        history.append(x)
    print(f"Point de convergence avec pas constant : {history[-1]}")  # Afficher le point de convergence
    return history

# Sous-gradient avec pas adaptable (réduction du pas)
def subgradient_adaptive_step(initial_x, learning_rate, max_iters):
    x = np.array(initial_x)
    history = [x]
    for i in range(1, max_iters + 1):
        lr = learning_rate / i  # Le pas diminue au fur et à mesure
        grad = subgradient(x)
        x = x - lr * grad
        history.append(x)
    print(f"Point de convergence avec pas adaptable : {history[-1]}")  # Afficher le point de convergence
    return history

# Sous-gradient avec recherche linéaire (backtracking)
def subgradient_line_search(initial_x, learning_rate, max_iters, beta=0.7, alpha=0.3):
    x = np.array(initial_x)
    history = [x]
    for _ in range(max_iters):
        lr = learning_rate
        grad = subgradient(x)
        while f(x - lr * grad) > f(x) - alpha * lr * np.dot(grad, grad):
            lr *= beta  # Réduire lr
        x = x - lr * grad
        history.append(x)
    print(f"Point de convergence avec recherche linéaire : {history[-1]}")  # Afficher le point de convergence
    return history

# Paramètres
initial_x = [10, 10]  # Point de départ
learning_rate = 0.6
max_iters = 50

# Calcul avec différentes méthodes
history_constant = subgradient_constant_step(initial_x, learning_rate, max_iters)
history_adaptive = subgradient_adaptive_step(initial_x, learning_rate, max_iters)
history_line_search = subgradient_line_search(initial_x, learning_rate, max_iters)

# Tracer la fonction de coût f(x) = |x1| + |x2|
x1_values = np.linspace(-10, 10, 200)
x2_values = np.linspace(-10, 10, 200)
X1, X2 = np.meshgrid(x1_values, x2_values)
Y = np.abs(X1) + np.abs(X2)

# Création de la figure pour l'animation
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Tracer la fonction
for ax in axes:
    ax.contour(X1, X2, Y, levels=50, cmap='coolwarm', alpha=0.6)  # Dégradé coolwarm

# Fonction d'animation pour la convergence de chaque méthode
def update(frame, history, ax, method_name, colors):
    ax.clear()  # Effacer l'ancienne image
    ax.contour(X1, X2, Y, levels=50, cmap='coolwarm', alpha=0.6)  # Redessiner la fonction
    ax.plot([h[0] for h in history[:frame+1]], [h[1] for h in history[:frame+1]], 
            label=method_name, marker='o', color=colors[frame % len(colors)], markersize=7, linestyle='-', linewidth=2)
    
    # Afficher le point de convergence en rouge
    if frame == len(history) - 1:  # Le dernier point (point de convergence)
        x_final = history[frame]
        ax.plot(x_final[0], x_final[1], 'ro', markersize=10)  # Point de convergence en rouge
        ax.text(x_final[0], x_final[1], f"({x_final[0]:.2f}, {x_final[1]:.2f})", 
                color='red', fontsize=12, ha='center', va='center', fontweight='bold')
    
    ax.set_title(f"Convergence - {method_name}", fontsize=14)
    ax.set_xlabel("x1", fontsize=12)
    ax.set_ylabel("x2", fontsize=12)
    ax.legend(loc="best", fontsize=10)

# Choisir une palette de couleurs dégradées pour la convergence
colors_constant = plt.cm.viridis(np.linspace(0, 1, len(history_constant)))  # Gradient viridis
colors_adaptive = plt.cm.inferno(np.linspace(0, 1, len(history_adaptive)))  # Gradient inferno
colors_line_search = plt.cm.plasma(np.linspace(0, 1, len(history_line_search)))  # Gradient plasma

# Création des animations pour chaque méthode
ani_constant = FuncAnimation(fig, update, frames=len(history_constant), fargs=(history_constant, axes[0], "Pas constant", colors_constant), interval=200, repeat=False)
ani_adaptive = FuncAnimation(fig, update, frames=len(history_adaptive), fargs=(history_adaptive, axes[1], "Pas adaptable", colors_adaptive), interval=200, repeat=False)
ani_line_search = FuncAnimation(fig, update, frames=len(history_line_search), fargs=(history_line_search, axes[2], "Recherche linéaire", colors_line_search), interval=200, repeat=False)

# Afficher l'animation
plt.suptitle("Convergence des Méthodes du Sous-Gradient avec Différents Pas", fontsize=16, fontweight='bold')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()
