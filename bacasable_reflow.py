from bacasable_efm import champ_exact
from bacasable_ot import matrice_cout

from scipy.integrate import solve_ivp

import numpy as np
import torch 

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
    return

def champ_reflow_nw(x, t, Z, G, eps=0.05):
    return

def entrainer_mlp_reflow(Z, G, n_epochs=2000, lr=1e-3):
    return 

def integrer_euler(champ, z, n_pas, retour_trajet=False):
    return 

def cout_transport(Z, G):
    return np.square(np.linalg.norm(Z-G))/len(Z)

def segments_se_croisent(p1, q1, p2, q2):
    return 

def compter_croisements(Z, G):
    return 

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
