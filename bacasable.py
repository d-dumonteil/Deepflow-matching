import numpy as np
np.random.seed(42)

def champ_exact(x, t, data, eps=1e-4):
    x_t = (1-t)*x+t*data
    distance = [x_t-t*data for i in support]
    lambda_i = np.exp(-np.square(data)/(2*(1-t)**2))
    lambda_i = lambda_i/np.sum(lambda_i)
    moy = np.sum(lambda_i*data)
    return (moy-x)/(1-t)

def cible_efm(x0, x1, t, muplet):
    x_t = (1-t)*x0+t*x1
    support = [x1]+ muplet
    distance = [x_t-t*i for i in support]
    lambda_i = np.exp(-np.square(distance)/(2*(1-t)**2))
    lambda_i = lambda_i/np.sum(lambda_i)
    moy = np.sum(lambda_i*support)
    return (moy-x_t)/(1-t)

def tirer_support(M, data):
    tirage = np.random.choice(len(data), M-1, replace=false)
    data_tirage = [data[i] for i in tirage]
    return data_tirage


def main():
    
    data1 = np.random.normal(3, 1, size= (200,2))
    data2 = np.random.normal(-3, 1, size= (200,2))
    print(data1.shape)

    mini = np.array([8., 10.])
    print(champ_exact(4.0, 0.0, mini))
    print(champ_exact(4.0, 0.5, mini))
    return 

if __name__ == "__main__":
    main()