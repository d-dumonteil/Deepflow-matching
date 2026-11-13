from bacasable_efm import champ_exact

from scipy.integrate import solve_ivp
from scipy.optimize import linear_sum_assignment

import numpy as np
import torch 
import math as mt
import matplotlib.pyplot as plt


from matplotlib.ticker import NullFormatter

np.random.seed(42)


def generer_donnees_bacasable_2d(n_par_amas, forme="croissants"):
    if forme=="croissants":
        A = np.random.uniform(0,np.pi,n_par_amas)
        B = np.random.uniform(0,np.pi,n_par_amas)
        A=np.column_stack([np.cos(A), np.sin(A)])
        B=np.column_stack([1-np.cos(B), 0.5-np.sin(B)])
        data = np.vstack([A, B])                    # (2n, 2)
        return data + np.random.normal(0, 0.1, data.shape)  # on épaissit les arcs
    else:
        return np.random.normal(0,1,(n_par_amas*2,2))

def generer_bruits(n):
    return np.random.normal(0,1,(n,2))


def integrer_rk45(z, data, rtol=1e-8, atol=1e-8, retour_trajet=False):
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
    g=integrer_rk45(z, data)
    return z,g

def champ_reflow_nw(x, t, z, g, eps=0.05):
    instant_i=z+t*(g-z)
    distance = np.sum((x[:, None, :] - instant_i[None, :, :]) ** 2, axis=2)
    lambda_i = np.exp(-distance / (2 * eps ** 2))
    lambda_i = lambda_i / np.maximum(lambda_i.sum(axis=1, keepdims=True), 1e-12)
    moy = lambda_i @ g
    return (moy - x) / max(1 - t, eps)

def entrainer_mlp_reflow(Z, G, n_epochs=2000, lr=1e-3):
    torch.manual_seed(0)

    Z_t = torch.tensor(Z, dtype=torch.float32) 
    G_t = torch.tensor(G, dtype=torch.float32) 
    


    modele = torch.nn.Sequential(
        torch.nn.Linear(3, 64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, 64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, 2),
    )

    opt = torch.optim.Adam(modele.parameters(), lr=1e-3)
    paires = torch.randint(0,len(Z), (512)) 
    z = Z_t[paires]
    g = G_t[paires]

    t = torch.rand((512,1))

    x = z+t*(g-z)

    entry = torch.cat((x,t), axis=0)

    cible = g-z

    pred = modele(entry)


    perte = ((pred - cible) ** 2).mean()

    opt.zero_grad()
    perte.backward()
    opt.step()

    torch.nn.functional.mse_loss(pred, cible)
    with torch.no_grad():
        resultat = modele(entree)

                            
    return 

def integrer_euler(champ, z, n_pas, retour_trajet=False):  
    dt = (1-1e-4 )/n_pas
    x=z.copy()
    if retour_trajet == True:
        l = [x.copy()]
    for i in range(n_pas):
        t = i*dt
        v = champ(x,t)
        x=x+dt*v
        if retour_trajet == True:
            l.append(x)

    if retour_trajet == False:
        return x
    else:
        return x, np.array(l)

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

def croisements(Z, G):
    acc = 0
    l = []
    for i in range(len(Z) - 1):
        for j in range(i + 1, len(Z)):
            if segments_se_croisent(Z[i], G[i], Z[j], G[j]):
                acc += 1
                l.append([(i,j)])
    return acc, l

def rectitude(trajets, Z, G):
    trajet_ligne = np.sum((Z-G)**2, axis=1)
    trajet_courbe = np.sum(np.diff(trajets, axis=0)**2, axis=1)
    return trajet_courbe/trajet_ligne

def distance_w1(echantillons, data): #faut des tailles égales 
    n = len(echantillons)
    m = len(data)
    x0f = echantillons.reshape(echantillons.shape[0], -1) 
    x1f = data.reshape(data.shape[0], -1)
    n0 = (x0f ** 2).sum(dim=1, keepdim=True) 
    n1 = (x1f ** 2).sum(dim=1).unsqueeze(0)   
    C=np.sqrt(np.maximum(n0 + n1 -2*x0f@x1f.T, 0))
    lignes, colonnes = linear_sum_assignment(C)
    return C[lignes, colonnes].sum() / len(echantillons)

def courbe_erreur_nfe(champ, data, bruits, liste_pas=(1, 2, 4, 8, 20),n_test=2000):
    d = {}
    for i in liste_pas:
        d[i] = distance_w1(integrer_euler(champ, bruits, i), data)
    return d

def figure_trajectoires(trajets_base, trajets_reflow, data, chemin, n_traj=30): #généré par IA
    """Deux panneaux : trajectoires du champ de base vs champ reflow.
    trajets_* : (K+1, n, 2) — axe 0 = temps. Mêmes départs des deux côtés."""
    fig, (ax_g, ax_d) = plt.subplots(1, 2, figsize=(11, 5))

    # bornes communes pour les deux panneaux (comparaison honnête)
    tout = np.vstack([data, trajets_base[-1], trajets_reflow[-1]])
    x_min, y_min = tout.min(axis=0)
    x_max, y_max = tout.max(axis=0)

    for ax, trajets, titre in [
        (ax_g, trajets_base, "Champ de base (trajectoires courbes)"),
        (ax_d, trajets_reflow, "Champ reflow (trajectoires quasi droites)"),
    ]:
        ax.scatter(data[:, 0], data[:, 1], s=3, color="lightgrey", alpha=0.6, zorder=1)
        for i in range(n_traj):                      # mêmes indices des deux côtés
            ax.plot(trajets[:, i, 0], trajets[:, i, 1],
                    linewidth=0.8, alpha=0.6, zorder=2)
        ax.scatter(trajets[0, :n_traj, 0], trajets[0, :n_traj, 1],
                   s=12, color="black", zorder=3)    # points de départ
        ax.set_xlim(x_min - 0.3, x_max + 0.3)
        ax.set_ylim(y_min - 0.3, y_max + 0.3)
        ax.set_aspect("equal")
        ax.set_title(titre)
        ax.grid(alpha=0.2)

    fig.tight_layout()
    fig.savefig(chemin, dpi=150)
    plt.close(fig)

def figure_croisements(Z_a, G_a, Z_b, G_b, chemin, n_seg=500): #généré par IA
    """Gauche : couplage indépendant. Droite : couplage reflow.
    Segments gris = non croisés, rouges = impliqués dans un croisement.
    Le compte dans le titre porte sur le couplage COMPLET, pas le sous-échantillon."""
    fig, (ax_g, ax_d) = plt.subplots(1, 2, figsize=(11, 5))

    for ax, Z, G, titre in [
        (ax_g, Z_a, G_a, "Couplage indépendant"),
        (ax_d, Z_b, G_b, "Couplage reflow"),
    ]:
        n = len(Z)
        total, paires = croisements(Z, G)          # comptage complet
        croises = {i for i, j in paires} | {j for i, j in paires}

        sous = np.random.choice(n, size=min(n_seg, n), replace=False)
        for k in sous:                                      # les autres, gris et fins
            if k not in croises:
                ax.plot([Z[k, 0], G[k, 0]], [Z[k, 1], G[k, 1]],
                        linewidth=0.7, alpha=0.3, color="grey", zorder=1)
        for i, j in paires:                                  # les croisants, rouges et épais
            ax.plot([Z[i, 0], G[i, 0]], [Z[i, 1], G[i, 1]],
                    linewidth=1.3, color="red", alpha=0.8, zorder=2)
            ax.plot([Z[j, 0], G[j, 0]], [Z[j, 1], G[j, 1]],
                    linewidth=1.3, color="red", alpha=0.8, zorder=2)

        ax.set_title(f"{titre} : {total} croisements")
        ax.set_aspect("equal")
        ax.grid(alpha=0.2)

    fig.tight_layout()
    fig.savefig(chemin, dpi=150)
    plt.close(fig)

def figure_nfe(courbes, chemin): #généré par IA
    """W1 vs nombre de pas, une courbe par champ.
    courbes : {nom du champ -> {k -> W1}}, le dict rendu par courbe_erreur_nfe."""
    fig, ax = plt.subplots(figsize=(7, 5))

    for nom, d in courbes.items():
        ks = sorted(d)
        ax.plot(ks, [d[k] for k in ks], marker="o", label=nom)

    ax.set_xscale("log")
    ax.set_xticks([1, 2, 4, 8, 20])
    ax.xaxis.set_minor_formatter(NullFormatter())   # pas de graduations intermédiaires
    ax.set_xlabel("Nombre de pas Euler (NFE)")
    ax.set_ylabel("Distance W1 aux données")
    ax.legend()
    ax.grid(alpha=0.3, which="both")

    fig.tight_layout()
    fig.savefig(chemin, dpi=150)
    plt.close(fig)

def main(): 

    return 0

if __name__ == "__main__":
    main()
