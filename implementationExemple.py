import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def f(x, y):
    """ Fonction objective f(x, y) = max(|x|, |y|) """
    return max(abs(x), abs(y))

def subgradient(x, y):
    """ Sous-gradient de f(x, y) = max(|x|, |y|) """
    if abs(x) > abs(y):
        return np.sign(x), 0  # Sous-gradient par rapport à x
    elif abs(x) < abs(y):
        return 0, np.sign(y)  # Sous-gradient par rapport à y
    else:
        return np.sign(x), np.sign(y)  # Sous-gradient pour les deux

def constant_step(k, alpha=0.5):
    """ Pas constant """
    return alpha

def decreasing_step(k, alpha=1):
    """ Pas décroissant, 1/(k+1) """
    return alpha / (k + 1)

def adaptive_step(k, alpha=2):
    """ Pas adaptatif, 2/k """
    return alpha / k

def subgradient_method(step_type, max_iter=100, alpha=1):
    """ Méthode du sous-gradient avec différents types de pas """
    x, y = 2, -3 # Point initial
    trajectory = [(x, y)]  # Trajectoire des points

    for k in range( 1,max_iter + 1):
        grad_x, grad_y = subgradient(x, y)  # Calcul du sous-gradient
        step_size = step_type(k, alpha)  # Taille du pas (selon le type)

        # Mise à jour des coordonnées
        x -= step_size * grad_x
        y -= step_size * grad_y
        
        trajectory.append((x, y))

        # Critère de convergence : différence entre points successifs
        if np.linalg.norm([x - trajectory[-2][0], y - trajectory[-2][1]]) < 1e-6:
            break

    return np.array(trajectory)

# Appliquer la méthode du sous-gradient avec trois types de pas
trajectories_constant = subgradient_method(constant_step, alpha=0.5)
trajectories_decreasing = subgradient_method(decreasing_step, alpha=1)
trajectories_adaptive = subgradient_method(adaptive_step, alpha=2)

# Créer la figure et organiser les sous-graphes
fig = plt.figure(figsize=(12, 6))

# Affichage 2D des trajectoires
ax1 = fig.add_subplot(121)  # 1ère sous-figure (à gauche)
ax1.plot(trajectories_constant[:, 0], trajectories_constant[:, 1], marker='o', markersize=15, label='Pas constant')
ax1.plot(trajectories_decreasing[:, 0], trajectories_decreasing[:, 1], marker='x', markersize=12, label='Pas décroissant')
ax1.plot(trajectories_adaptive[:, 0], trajectories_adaptive[:, 1], marker='s', markersize=5, label='Pas adaptatif')

ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_title('Trajectoires du sous-gradient avec différents pas en 2D')
ax1.legend()
ax1.grid(True)

# Affichage 3D des trajectoires
ax2 = fig.add_subplot(122, projection='3d')  # 2ème sous-figure (à droite)
z_constant = [f(x, y) for x, y in trajectories_constant]
z_decreasing = [f(x, y) for x, y in trajectories_decreasing]
z_adaptive = [f(x, y) for x, y in trajectories_adaptive]

ax2.plot(trajectories_constant[:, 0], trajectories_constant[:, 1], z_constant, marker='o', markersize=17, label='Pas constant')
ax2.plot(trajectories_decreasing[:, 0], trajectories_decreasing[:, 1], z_decreasing, marker='x', markersize=12, label='Pas décroissant')
ax2.plot(trajectories_adaptive[:, 0], trajectories_adaptive[:, 1], z_adaptive, marker='s', markersize=5, label='Pas adaptatif')

ax2.set_xlabel('x')
ax2.set_ylabel('y')
ax2.set_zlabel('f(x, y)')
ax2.set_title('Trajectoires du sous-gradient avec différents pas en 3D')
ax2.legend()

plt.tight_layout()  # Pour que les graphiques ne se chevauchent pas
plt.show()

print(f"Point de convergence (Pas constant) : x = {subgradient_method(constant_step, alpha=1)}")
print(f"Point de convergence (Pas décroissant) : x = {subgradient_method(decreasing_step, alpha=1)}")
print(f"Point de convergence (Pas adaptatif) : x = {subgradient_method(adaptive_step, alpha=2)}")
# Affichage des points de convergence pour chaque méthode
print(f"Point de convergence (Pas constant) : x = {trajectories_constant[-1][0]}, y = {trajectories_constant[-1][1]}")
print(f"Point de convergence (Pas décroissant) : x = {trajectories_decreasing[-1][0]}, y = {trajectories_decreasing[-1][1]}")
print(f"Point de convergence (Pas adaptatif) : x = {trajectories_adaptive[-1][0]}, y = {trajectories_adaptive[-1][1]}")
