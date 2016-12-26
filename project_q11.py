# -*- coding: utf-8 -*-
"""
Created on Mon Dec 26 10:22:20 2016

@author: benoitguillard

====QUESTION 11====

Equation de convection diffusion bi-dimensionnelle
"""

# Importation des bibliotheques nécessaires
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy import linalg as lnlg
from scipy.sparse import dia_matrix
from matplotlib import animation


plt.rcParams['animation.ffmpeg_path'] = '/Users/benoitguillard/anaconda2/ffmpeg/ffmpeg'


# Constantes numeriques pour la simulation
nu=1		# Coefficient de diffusion
L=50		# Distance caracteristique considérée
T=10		# Durée de la simulation
h=0.5		# Pas de discretisation en espace
dt=0.1	# Pas de discretisation en temps

x0=25		# Constantes decrivant l'etat initial
y0=25
sigma=1

N=int(L*1/h-1)	# Taille de la maille dans une direction

# Axes de simulation :
axe=np.linspace(0,L,N)
X=np.linspace(0,L,N)
Y=np.linspace(0,L,N)
X,Y=np.meshgrid(X,Y)

# Definition des matrices necessaires a la resolution (G*Un+1 = D*Un)
A=1.0/h**2 * (np.diag([1]*(N**2-1),-1) + np.diag([1]*(N**2-1),1) + 
	np.diag([1]*(N**2-N),-N) + np.diag([1]*(N**2-N),N) - 4*np.eye(N**2))

Kcy=1.0/(2*h) * (np.diag([1]*(N**2-1),1) - np.diag([1]*(N**2-1),-1))

Kcx=1.0/(2*h) * (np.diag([1]*(N**2-N),N) - np.diag([1]*(N**2-N),-N))

G=np.eye(N**2) + 0.5*dt* (Kcx+Kcy) - 0.5*nu*dt* A
D=np.eye(N**2) - 0.5*dt* (Kcx+Kcy) + 0.5*nu*dt* A


# Conditions aux bord :
l=[0]*(N**2)
for i in range(N):
	for j in [i,N**2-1-i,i*N,(i+1)*N-1]:
		D[j]=l
		l[j]=1
		G[j]=l
		l[j]=0

# Preparation des matrices LU sur celle de gauche, creuse pour celle de droite
P,B,H = lnlg.lu(G)
# Si la matrice de permutation P de la décomposition est différente de l'identité, on a un pb..
if (not((P==np.eye(N**2)).all())):
	print ("On va avoir un soucis, car la décomposition LU de G n'est pas usuelle..")

sparseD=dia_matrix(D)

# Transformer un vecteur de longueur N**2 en une matrice de taille N*N par recollage des lignes
def vectToMat(v):
	ans=np.eye(N)
	for i in range(N):
		for j in range(N):
			ans[i][j]=v[(i-1)*N+j]
	return ans

# Transformation inverse : matrice N*N en vecteur N**2, par decoupage des lignes
def matToVect(m):
	ans=np.array([0]*(N**2))
	for i in range(N):
		for j in range(N):
			ans[(i-1)*N+j]=m[i][j]
	return ans

# Fonction decrivant l'état initial :
def f0(x,y):
	return 1.0/(sigma*np.sqrt(2*np.pi))*np.exp(-(((x-x0)**2)+((y-y0)**2))/(2*sigma**2))

# Etat initial :
# Vecteur source :
U=np.array([0.0]*N**2)
for i in range(N):
	for j in range(N):
		U[(i-1)*N+j]=f0(axe[i],axe[j])

# Creation de la figure
fig = plt.figure(figsize=(17, 10), dpi=80)
fig.suptitle('Q11 : Convection diffusion bi-dimensionnelle', fontsize=14, fontweight='bold')
ax = plt.axes(xlim=(0, 50), ylim=(0, 50), zlim=(0,0.2) , projection='3d')

# Etiquettage des axes
ax.set_xlabel("Position en x")
ax.set_ylabel("Position en y")
ax.set_zlabel("Concentration en polluant")

surf = ax.plot_wireframe(X, Y, vectToMat(U), lw=0.5, color='r')

# Compteur global du nombre d'iterations
nbrIter=0



# Fonction a iterer pour animer la figure
def animate(i):
	global nbrIter
	global U
	ax.clear()
	ax.set_title("t = " + str(int(nbrIter*dt*10)*1.0/10) + "s")	# Affichage de l'instant de simulation
	ax.set_xlabel("Position en x")
	ax.set_ylabel("Position en y")
	ax.set_zlabel("Concentration en polluant")
	nbrIter+=1				# Incrementation du nombre d'iterations
	# Iteration par resolution du systeme linéaire
	T=lnlg.solve_triangular(B,sparseD.dot(U) , check_finite=False, lower=True)
	U=lnlg.solve_triangular(H,T, check_finite=False)
	surf = ax.plot_wireframe(X, Y, vectToMat(U), lw=0.5, color='r')
	ax.set_zlim(0,0.2)
	return surf,

# Lancement de l'animation
anim = animation.FuncAnimation(fig, animate, 
	                               frames=int(T/dt), interval=dt*1000, blit=False)
"""
# Sauvegarde de l'animation
FFwriter = animation.FFMpegWriter(bitrate=8192)
anim.save('basic_animation.mp4', writer = FFwriter, fps=10)
"""