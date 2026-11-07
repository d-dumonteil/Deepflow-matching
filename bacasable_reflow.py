from bacasable_efm import champ_exact
from bacasable_ot import matrice_cout

import numpy as np
import torch 

np.random.seed(42)


def generer_donnees_bacasable_2d(n_par_amas, forme="croissants"):
    if forme=="croissants":
        A = np.random.uniform(n_par_amas)
        B = np.random.uniform(n_par_amas)
        A=np.concatenate([np.cos(A), np.sin(A)])
        B=np.concatenate([1-np.cos(B), 0.5-np.sin(B)])
        return A,B
    else:
        return np.random.normal((2,n_par_amas)),np.random.normal((2,n_par_amas))

def generer_bruits(n):
    return np.random.normal((2,n))

def integrer_prof_rk45(z, data, rtol=1e-8, atol=1e-8, retour_trajet=False):
    return 

def generer_paires(n, data):
    return

def champ_reflow_nw(x, t, Z, G, eps=0.05):
    return

def entrainer_mlp_reflow(Z, G, n_epochs=2000, lr=1e-3):
    return 

def integrer_euler(champ, z, n_pas, retour_trajet=False):
    return 

def cout_transport(Z, G):
    return 

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
