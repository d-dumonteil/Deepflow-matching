import numpy as np 
from scipy.optimize import linear_sum_assignment




def matrice_cout(x0, x1):
    ones = np.ones(x0.shape())
    mat_x0 = np.linealg.inv(x0)@ones
    mat_x1 = np.linealg.inv(x1)@ones
    return np.abs(mat_x0-mat_x1)

def permutation(C):
    lignes, colonnes = linear_sum_assignment(C)
    return

def cout_total(C, permutation):
    return 

def cout_random_moyen(x0, x1, n_tirages=1000):
    return 

def visualiser_appariement(x0, x1, permutation, chemin_fichier):
    return 

def generer_donnees_2d(n_par_gaussienne=100, mu=3.0, sigma=1.0):
    return

def main():
    M=32 #batch size 
    data = np.concatenate([np.random.normal(3, 1, size=(100, 2)),np.random.normal(-3, 1, size=(100, 2)),])
    return 0

if __name__ == "__main__":
    main()