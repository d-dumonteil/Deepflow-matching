from bacasable_efm import champ_exact
from bacasable_ot import matrice_cout

from scipy.integrate import solve_ivp

import numpy as np
import torch 
import math as mt

np.random.seed(42)


def generer_donnees_bacasable_2d(n_par_amas, forme="croissants"):
    if forme=="croissants":
        A = np.random.uniform(0,np.pi,n_par_amas)
        B = np.random.uniform(0,np.pi,n_par_amas)
        A=np.column_stack([np.cos(A), np.sin(A)])
        data = np.vstack([A, B])                    # (2n, 2)
        return data + np.random.normal(0, 0.1, data.shape)  # on épaissit les arcs
    else:
        return np.random.normal(0,1,(n_par_amas*2,2))

def generer_bruits(n):
    return np.random.normal(0,1,(n,2))


def integrer_prof_rk45(z, data, rtol=1e-8, atol=1e-8, retour_trajet=False):
    def derivee_scipy(t, y):
        pts = y.reshape(-1, 2)
        return champ_exact(pts, t, data).reshape(-1)

    if retour_trajet:
        t_eval = np.linspace(0.0, 1 - 1e-4, 201)
    else:
        t_eval = [1 - 1e-4]

    sol = solve_ivp(derivee_scipy, (0.0, 1 - 1e-4), z.reshape(-1), rtol=rtol, atol=atol, t_eval=t_eval)

    if retour_trajet:
        trajets = sol.y.T.reshape(201, -1, 2)   
        return trajets[-1], trajets                 
    return sol.y[:, -1].reshape(-1, 2)

def generer_paires(n, data):
    z = generer_bruits(n)
    g=integrer_prof_rk45(z, data)
    return z,g

def champ_reflow_nw(x, t, z, g, eps=0.05):
    instant_i=z+t*(z-g)
    distance = np.sum((x - instant_i) ** 2, axis=1)        
    lambda_i = np.exp(-distance / (2 * max(1 - t, eps) ** 2))
    lambda_i = lambda_i / lambda_i.sum()                       
    moy = lambda_i @ instant_i                                    
    return (moy - x) / max(1 - t, eps)   

def entrainer_mlp_reflow(Z, G, n_epochs=2000, lr=1e-3):
    return 

def integrer_euler(champ, z, n_pas, retour_trajet=False):
    return 

def cout_transport(Z, G):
    return np.square(np.linalg.norm(Z-G))/len(Z)

def cote(a, b, c):
    dx = b[0] - a[0]
    if dx == 0:                                  
        return np.sign(c[0] - a[0])
    s = (b[1] - a[1]) / dx                        
    return np.sign(c[1] - (a[1] + s * (c[0] - a[0])))   

def segments_se_croisent(p1, q1, p2, q2):
    return (cote(p1, q1, p2) * cote(p1, q1, q2) < 0) and (cote(p2, q2, p1) * cote(p2, q2, q1) < 0)

def compter_croisements(Z, G):
    acc = 0
    for i in range(len(Z) - 1):
        for j in range(i + 1, len(Z)):
            if segments_se_croisent(Z[i], G[i], Z[j], G[j]):
                acc += 1
    return acc

def rectitude(trajets, Z, G):
    return 

def distance_w1(echantillons, data):
    return 

def courbe_erreur_nfe(champ, data, liste_pas=(1, 2, 4, 8, 20),n_test=2000):
    return 

def figure_trajectoires(trajets_base, trajets_reflow, data, chemin):
    return 

def figure_croisements(Z_a, G_a, Z_b, G_b, chemin):
    return 

def figure_nfe(courbes: dict, chemin):
    return 

def main():

    return 0

if __name__ == "__main__":
    main()
