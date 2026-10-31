import numpy as np

def champ_exact(x, t, data, eps=1e-4):
    lambda_i = np.exp(-np.sum(data**2)/(2*(1-t)))
    lambda_i = lambda_i/np.sum(lambda_i)
    moy = np.sum(lambda_i*x)
    return (moy-x)/(1-t)

def cible_efm(x0, x1, t, muplet):
    x_t = t*x0+(1-t)*x1
    support = [x1]+ muplet
    lambda_i = np.exp(-np.sum(data**2)/(2*(1-t)))
    lambda_i = lambda_i/np.sum(lambda_i)
    return 

def tirer_support(M, data):
    tirage = np.random.uniform(0,len(data), M)
    data_tirage = [data[i] for i in tirage]
    return data_tirage


def main():
    
    data1 = np.random.normal(3, 1, size= (2,100))
    data2 = np.random.normal(-3, 1, size= (2,100))

    np.info('sum')
    return 0

if __name__ == "__main__":
    main()