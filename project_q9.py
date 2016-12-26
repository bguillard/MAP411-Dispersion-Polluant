# -*- coding: utf-8 -*-
"""
Created on Sun Dec 25 20:48:42 2016

@author: benoitguillard

====QUESTION 9====

Discretisation du Laplacien a 2 dimensions
"""

# Importation des bibliotheques nécessaires
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy import linalg as lnlg
from scipy.sparse import dia_matrix
import time


# Constantes numeriques pour la simulation
L=50		# Distance caracteristique considérée
h=1		# Pas de discretisation en espace
N=L*1/h-1	# Taille de la maille dans une direction

# Axes de simulation :
axe=np.linspace(0,L,N)
Y=np.eye(N)
for i in range(N):
	Y[i]=[axe[i]]*N
X=np.array([axe]*N)

# Definition des matrices necessaires a la resolution
A=1.0/h**2 * (np.diag([1]*(N**2-1),-1) + np.diag([1]*(N**2-1),1) + 
	np.diag([1]*(N**2-N),-N) + np.diag([1]*(N**2-N),N) - 4*np.eye(N**2))

# Conditions aux bord :
for i in range(N**2):
	if i<N or i>=N**2-N or i%N==0 or i%N==N-1:
		l=[0]*(N**2)
		l[i]=1
		A[i]=l
	
# Fonction source :
def f(x,y):
	if x>=20 and x<=30 and y>=20 and y<=30:
		return 1
	else:
		return 0

# Vecteur source :
F=np.array([0]*N**2)
for i in range(N):
	for j in range(N):
		F[(i-1)*N+j]=f(axe[i],axe[j])

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

# Creation de la figure
fig = plt.figure()
fig.suptitle('Q9 : Laplacien a 2 dimensions', fontsize=14, fontweight='bold')
ax = plt.axes(xlim=(0, 50), ylim=(0, 50), zlim=(0,50) , projection='3d')

surf = ax.plot_wireframe(X, Y, vectToMat(F), lw=1, color='r')

# Resolution du systeme lineaire
U=np.linalg.solve(-A,F)
surf = ax.plot_wireframe(X, Y, vectToMat(U), lw=1)

"""
Test de différentes méthodes de stockage/résolution
Conclusion :
 Le plus rapide est d'utiliser la methode LU pour decomposer la matrice de gauche 
	de l'equation
	ET de stocker celle de droite sous forme d'une matrice creuse diagonale (dia_matrix)
"""

def resolNormal():
	for i in range(1000):
		U=np.linalg.solve(-A,F)
	return  U

def resolTriangular():
	P,B,H=lnlg.lu(A)
	start_time = time.time()     
	for i in range(1000):
		Y=lnlg.solve_triangular(B,-np.dot(A,F) , check_finite=False, lower=True)
		U=lnlg.solve_triangular(H,Y, check_finite=False)
	interval = time.time() - start_time  
	print 'Total time in seconds:', interval 
	return U

def resolTriangularSparse():
	P,B,H=lnlg.lu(A)
	sparseA=dia_matrix(A)
	start_time = time.time()
	for i in range(1000):
		Y=lnlg.solve_triangular(B,-sparseA.dot(F) , check_finite=False, lower=True)
		U=lnlg.solve_triangular(H,Y, check_finite=False)
	interval = time.time() - start_time  
	print 'Total time in seconds:', interval
	return U


