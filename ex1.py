import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Définir la fonction f(x) = |x| et son sous-gradient
def f(x):
    return np.abs(x)
def subgradient(x):
    return np.sign(x)

# Paramètres de la simulation
iterations = 1000
x_init = 5  # Point de départ
alpha_const = 0.1  # Pas constant
alpha_desc = 0.5  # Pas initial de descente
alpha_adapt = 0.1  # Pas adaptatif
epsilon = 1e-5  # Critère de convergence sur la norme du sous-gradient

# Trajectoires pour les trois types de pas
x_constant = [x_init]
x_desc = [x_init]
x_adapt = [x_init]

# Fonction pour appliquer la méthode du sous-gradient avec différents pas
def update_position(x, alpha, grad):
    return x - alpha * grad

# Fonction de pas adaptatif
def adaptive_step(k, alpha_base=0.1):
    return alpha_base / (1 + np.log(k+1))

# Calcul des trajectoires avec le critère de convergence
for k in range(1, iterations + 1):
    grad_const = subgradient(x_constant[-1])
    grad_desc = subgradient(x_desc[-1])
    grad_adapt = subgradient(x_adapt[-1])
    
    # Pas constant
    x_constant.append(update_position(x_constant[-1], alpha_const, grad_const))
    
    # Pas de descente (1/k)
    x_desc.append(update_position(x_desc[-1], alpha_desc / k, grad_desc))
    
    # Pas adaptatif amélioré
    new_alpha_adapt = adaptive_step(k, alpha_base=alpha_adapt)
    x_adapt.append(update_position(x_adapt[-1], new_alpha_adapt, grad_adapt))
    
    # Critère d'arrêt sur la norme du sous-gradient
    if np.linalg.norm(grad_const) <= epsilon:
        conv_constant = x_constant[-1]
    if np.linalg.norm(grad_desc) <= epsilon:
        conv_desc = x_desc[-1]
    if np.linalg.norm(grad_adapt) <= epsilon:
        conv_adapt = x_adapt[-1]
    
    # Si les trois ont convergé, on peut arrêter la boucle
    if np.linalg.norm(grad_const) <= epsilon and np.linalg.norm(grad_desc) <= epsilon and np.linalg.norm(grad_adapt) <= epsilon:
        break

# Fonction pour vérifier la convergence en fonction de la norme du sous-gradient
def check_convergence(x_vals, epsilon=1e-5):
    for i in range(1, len(x_vals)):
        if np.abs(x_vals[i] - x_vals[i - 1]) < epsilon:
            return x_vals[i]
    return x_vals[-1]

# Calcul des points de convergence
conv_constant = check_convergence(x_constant)
conv_desc = check_convergence(x_desc)
conv_adapt = check_convergence(x_adapt)

print(f"Point de convergence (Pas constant): {conv_constant}")
print(f"Point de convergence (Pas descente): {conv_desc}")
print(f"Point de convergence (Pas adaptatif): {conv_adapt}")

# Création de la figure
fig, ax = plt.subplots(figsize=(8, 6))
x_vals = np.linspace(-6, 6, 400)
y_vals = f(x_vals)

# Tracer la fonction f(x) = |x|
ax.plot(x_vals, y_vals, label="f(x) = |x|", color='black', lw=2)

# Ajouter une ligne de convergence à y=0 pour mieux visualiser l'écart
ax.axhline(0, color='black', linestyle='--', label="Convergence (y=0)")

# Initialisation des points sur la trajectoire
line_const, = ax.plot([], [], 'ro-', label="Pas constant", markersize=6)
line_desc, = ax.plot([], [], 'go-', label="Pas descente", markersize=10)
line_adapt, = ax.plot([], [], 'bo-', label="Pas adaptatif", markersize=6)

# Points de convergence colorés
conv_point_const, = ax.plot([], [], 'ro', markersize=10, label="Point de convergence (Pas constant)", markeredgewidth=2)
conv_point_desc, = ax.plot([], [], 'go', markersize=10, label="Point de convergence (Pas descente)", markeredgewidth=2)
conv_point_adapt, = ax.plot([], [], 'bo', markersize=10, label="Point de convergence (Pas adaptatif)", markeredgewidth=2)

# Définir les limites de l'axe
ax.set_xlim(-6, 6)
ax.set_ylim(0, 6)
ax.set_xlabel("x", fontsize=12)
ax.set_ylabel("f(x)", fontsize=12)
ax.set_title("Méthode du Sous-Gradient avec Pas Constant, de Descente, et Adaptatif", fontsize=14)
ax.legend(fontsize=10)

# Mise en forme du fond pour une meilleure visibilité
ax.set_facecolor('#f0f0f0')
fig.patch.set_facecolor('#f0f0f0')

# Animation
def animate(i):
    # Mise à jour des trajectoires
    line_const.set_data(x_constant[:i+1], f(np.array(x_constant[:i+1])))
    line_desc.set_data(x_desc[:i+1], f(np.array(x_desc[:i+1])))
    line_adapt.set_data(x_adapt[:i+1], f(np.array(x_adapt[:i+1])))
    
    # Mise à jour des points de convergence
    if i == len(x_constant) - 1:
        conv_point_const.set_data(conv_constant, f(conv_constant))
    if i == len(x_desc) - 1:
        conv_point_desc.set_data(conv_desc, f(conv_desc))
    if i == len(x_adapt) - 1:
        conv_point_adapt.set_data(conv_adapt, f(conv_adapt))

    return line_const, line_desc, line_adapt, conv_point_const, conv_point_desc, conv_point_adapt

# Créer l'animation
ani = animation.FuncAnimation(fig, animate, frames=iterations, interval=50, blit=True)

plt.show()
