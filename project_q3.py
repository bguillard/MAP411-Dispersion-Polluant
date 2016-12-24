# -*- coding: utf-8 -*-
"""
Created on Sat Dec 24 17:33:21 2016

@author: benoitguillard

====QUESTION 3====

Schema explicite decentre amont pour l'equation de convection uni-dimensionnelle

"""

# Importation des bibliotheques nécessaires
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation

# Constantes numeriques pour la simulation
L=50		# Longueur du fleuve considérée
T=5		# Durée de la simulation
V=1		# Vitesse de convection
nu=1		# Coefficient de diffusion
dx=0.1	# Pas de discretisation en espace
dt=0.025	# Pas de discretisation en temps
# ====> Prendre dt=0.25 par exemple pour observer la non stabilite du modele

N=int(L/dx)-1	# Nombre de mailles

# Fonction decrivant la répartition initiale du polluant
def f0(x):
	x0=20
	sigma=1
	return 1/(sigma*np.sqrt(2*np.pi))*np.exp(-((x-x0)**2)/(2*sigma**2))

# Construction de la matrice d'iteration
K= 1/(2*dx)*(-np.diag([1]*(N-1),-1) + np.eye(N))
ItMat=np.eye(N)-V*dt*K

# Initialisation des vecteurs a tracer : coordonnees d'espace et concentration du polluant
X=np.linspace(0,L,N)
U=f0(X)

# Creation de la figure
fig = plt.figure()
fig.suptitle('Q3 : Schema explicite decentre amont', fontsize=14, fontweight='bold')
ax = plt.axes(xlim=(0, 50), ylim=(0, 0.5))
line, = ax.plot([], [], lw=2)


# Etiquettage des axes
ax.set_xlabel("Position")
ax.set_ylabel("Concentration en polluant")


# Compteur global du nombre d'iterations
nbrIter=0

# Fonction d'initialisation de l'animation
def init():
	line.set_data([], [])
	return line,

# Fonction a iterer pour animer la figure
def animate(i):
	global U
	global nbrIter
	ax.set_title("t = " + str(int(nbrIter*dt*10)*1.0/10) + "s")	# Affichage de l'instant de simulation
	nbrIter+=1				# Incrementation du nombre d'iterations
	U=np.dot(ItMat,U)		# Iteration par le produit matriciel
	line.set_data(X, U)
	return line,

# Lancement de l'animation
anim = animation.FuncAnimation(fig, animate, init_func=init,
	                               frames=int(T/dt), interval=dt*1000, blit=False)