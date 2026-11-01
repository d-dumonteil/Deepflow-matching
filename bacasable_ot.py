import numpy as np 
from scipy.optimize import linear_sum_assignment

def matrice_cout(x0, x1):
    diff = x0[:, None, :] - x1[None, :, :] 
    return np.sum(diff ** 2, axis=2) 

def permutation(C):
    _, colonnes = linear_sum_assignment(C)
    return colonnes

def cout_total(C, permutation):
    return np.sum(C[i,permutation[i]] for i in range(len(permutation)))

def cout_random_moyen(x0, x1, n_tirages=1000):
    C=matrice_cout(x0,x1)
    add=0
    for i in range(n_tirages):
        add = add+cout_total(C,np.random.choice(C.shape[0], C.shape[0], replace=False))/n_tirages
    return add

def main(): #tests générés par IA
    M=32 #batch size 

    data = np.concatenate([np.random.normal(3, 1, size=(100, 2)),np.random.normal(-3, 1, size=(100, 2)),])

    # --- 2. Étude pour chaque taille de batch ---
    resultats = []
    for M in (32, 64, 128):
        # Batch : M images tirées des données, M bruits gaussiens
        idx = np.random.choice(len(data), M, replace=False)
        x1 = data[idx]                          # images
        x0 = np.random.normal(0, 1, size=(M, 2))  # bruits

        # Matrice de coût et appariement
        C = matrice_cout(x0, x1)
        perm = permutation(C)

        # Vérifications
        assert C.shape == (M, M), f"Shape de C invalide : {C.shape}"
        assert (C >= 0).all(), "Coûts négatifs détectés"
        assert sorted(perm) == list(range(M)), "La permutation n'est pas une bijection"

        # Coûts
        c_ot = cout_total(C, perm)
        c_rand = cout_random_moyen(x0, x1, n_tirages=1000)
        rapport = c_rand / c_ot
        facteur_theorique = np.sqrt(M)

        # Sanity check : aucune permutation aléatoire ne doit battre l'optimum
        for _ in range(1000):
            p = np.random.permutation(M)
            assert cout_total(C, p) >= c_ot - 1e-9, "Une permutation aléatoire bat le solveur !"

        print(f"M = {M:3d} | Coût OT : {c_ot:8.2f} | Coût random : {c_rand:8.2f} "
              f"| Rapport : {rapport:5.2f} | Théorie √M : {facteur_theorique:.2f}")
        resultats.append((M, c_ot, c_rand, rapport))

    return resultats

if __name__ == "__main__":
    main()