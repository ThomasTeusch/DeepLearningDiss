#==================================================================================

import numpy as np
from sklearn.externals import joblib

#==================================================================================
#--- read input
def read_input(data, Y_col, w_col, Ao=1.0):
    f = open(data, "r")
    data = [[np.float64(c) for c in line.split()] for line in f.readlines()]
    f.close()
    #Ao = 0.52917721067
    Ao = 1.0 # Fit Data in a.u.
    data = np.array(data, dtype=np.float64)
    X = np.delete(data, [Y_col, w_col], axis=-1)
    # bohr adjustment
    X[:,1] *= Ao
    X[:,2] *= Ao
    input_dim = X.shape[1]
    Y = data[:, [Y_col]]
    w = 1. / data[:, w_col]
    return X,Y,w

def energies_to_gradients(ori):
    data = ori.copy() 
    save = []
    for i in range(len(data)):
        if(data[i][2] == 35.0):
            save = np.append(save, data[i][3])       
            data[i][3] = 0.0
        else:
            data[i][3] = (data[i+1][3] - data[i][3]) / (data[i+1][2] - data[i][2])
    Y = data[:, [3]]
    X = np.delete(data, [3], axis=-1)
    return X,Y, save

def gradients_to_energies(data_X, data_Y, asympt):
    data = np.concatenate((data_X,data_Y),axis=1)
    i = len(data)-1
    j = len(asympt)-1

    while i >= 0:
       if(data[i][2] == 35.0):
           data[i][3] = asympt[j]
           j-=1
       else:
           data[i][3] = -1*data[i][3]*(data[i+1][2]-data[i][2])+data[i+1][3]
       i-=1

    Y = data[:, [3]]
    X = np.delete(data, [3], axis=-1)
    return X, Y

def shuffle(data_X, data_Y, seed=42):
    data = np.concatenate((data_X,data_Y),axis=1)
    np.random.seed(seed)
    np.random.shuffle(data)
    Y = data[:, [3]]
    X = np.delete(data, [3], axis=-1)
    return X, Y


#================================
#--- transformation of input data
#------ default -> [-1:1]
def trans_def(data, d_min, d_max):
    return (((data - d_min) / (d_max - d_min)) * 2 - 1).astype(np.float32)
def inv_trans_def(trans_data, d_min, d_max):
    return (((trans_data.astype(np.float64) + 1) / 2) * (d_max - d_min) + d_min)
def inv_trans_def_torch(trans_data, d_min, d_max, cuda):
    d_min = torch.tensor(d_min)
    d_max = torch.tensor(d_max)
    if cuda:
        d_min, d_max = d_min.cuda(), d_max.cuda()
    return (((trans_data.double() + 1) / 2) * (d_max - d_min) + d_min)

#------ Morse (MBB) (see Habecker2013)
def trans_MBB(data, z_eq, gamma):
    return ( 1 - np.exp(-gamma*( (data-z_eq)/z_eq )) ) * 1/gamma
def inv_trans_MBB(trans_data, z_eq, gamma):
    return (-(np.log(-trans_data*gamma + 1) * z_eq) / gamma ) + z_eq

#------ COS (see Habecker2013)
def trans_cos(data):
    return np.cos(data / 180*np.pi)
def inv_trans_cos(trans_data):
    return np.arccos(trans_data) * 180/np.pi

#------ exp(-data/k)
def trans_exp(data, k=1.0):
    #X[:, 2] = np.exp(-X[:, 2] / k)
    return np.exp(-data / k)
def inv_trans_exp(trans_data, k=1.0):
    #X[:, 2] = -1*np.log(X[:, 2]) * k
    return -np.log(trans_data)*k

#------ manual interval [interval_min : interval_max]
def trans_min_max(data, minimum, maximum, interval_min=-1, interval_max=1):
    return (interval_max-interval_min) * (data-minimum)/(maximum - minimum) + interval_min
def inv_trans_min_max(trans_data, minimum, maximum, interval_min=-1, interval_max=1):
    return (maximum - minimum) * (trans_data-interval_min) / (interval_max-interval_min) + minimum

#------ TODO: transform energy
def trans_energy(data):
    print("No transformation found!")


#================================
#--- save values for prediction
def save_min_max(X,Y,data_dir):
    X_min = X.min(0)
    X_max = X.max(0)
    Y_min = Y.min(0)
    Y_max = Y.max(0)

    print("\t Using training set: %s" % (data_dir[7:]),flush=True)
    print("")
    joblib.dump({"Y_min": Y_min, "Y_max": Y_max,
                 "X_min": X_min, "X_max": X_max},
                "./models/vals_split_%s.pkl" % data_dir[7:-4])
    return X_min, X_max, Y_min, Y_max


#=== TODO:
#	Delete high energies in Z and replace with analytic expression?

def r12_potential_z(X, Y, X_min, z=3.0):
    m=X_min
    
    for i in range(len(Y)):
        Y[i] = ( m / X[i][2] )**12

    return Y

# UNDER CONSTRUCTION!
def r12_potential_energy(X,Y):
    minimum=max(Y)
    X1 = 0.
    X2 = 0.
    X3 = 0.
    for i in range(len(Y)):
        if(minimum > Y[i]):
            minimum = Y[i]
            X1 = X[i][0]
            X2 = X[i][1]
            X3 = X[i][2]
    print(X1, X2, X3, minimum)     


def train_test_split(X,Y,w,ratio):
    if ratio > .0:
        ratio = int(len(X) * (1-ratio))
        X_test = X[ratio:]
        X_train = X[:ratio]
        Y_test = Y[ratio:]
        Y_train = Y[:ratio]
        w_test = w[ratio:]
        w_train = w[:ratio]
    else:
        print("")
        print("\t Error in train_ratio")
        print("\t Negativ value in ratio")
        print("")
        exit()
    return X_train, X_test, Y_train, Y_test, w_train, w_test

#================================
#--- reduce data set size 

def reduce_data_z(X, Y, bound=3.0):
    zaehler=0
    Y_rep = []
    Y_red = []
    X1 = []
    X2 = []
    X3 = []
    X11 = []
    X22 = []
    X33 = []

    for i in range(len(Y)):
        if(X[i][2] >= abs(bound)):
            Y_red = np.append(Y_red, Y[i])
            X11 = np.append(X11,X[i,0])
            X22 = np.append(X22,X[i,1])
            X33 = np.append(X33,X[i,2])
        else:
            Y_rep = np.append(Y_rep, Y[i])
            X1 = np.append(X1,X[i,0])
            X2 = np.append(X2,X[i,1])
            X3 = np.append(X3,X[i,2])
            zaehler+=1
  
    X11 = np.reshape(np.array(X11),(len(X11),1))
    X22 = np.reshape(np.array(X22),(len(X22),1))
    X33 = np.reshape(np.array(X33),(len(X33),1))
    X1 = np.reshape(np.array(X1),(len(X1),1))
    X2 = np.reshape(np.array(X2),(len(X2),1))
    X3 = np.reshape(np.array(X3),(len(X3),1))

    X_rep = np.concatenate((X1, X2), axis=1)
    X_rep = np.concatenate((X_rep, X3), axis=1)
    X_red = np.concatenate((X11, X22), axis=1)
    X_red = np.concatenate((X_red, X33), axis=1)
    Y_rep = np.reshape(Y_rep,(len(Y_rep),1))
    Y_red = np.reshape(Y_red,(len(Y_red),1))

    print("\t Reduce input data: Delete energies with z values smaller " + str(bound) + " Bohr", flush=True)
    print("\t Input reduced by " + str(zaehler) + " elements" , flush=True)
    print("\t You have " + str(len(Y_red)) + " elements left for training" , flush=True)
    #print("\t Elements will be replaced with an analytical expression", flush=True)
    print("", flush=True)
    return X_red, X_rep, Y_red, Y_rep

def reduce_data_energy(X,Y,bound=5.0):
    zaehler=0
    zaehler2=0
    Y_rep = []
    Y_red = []
    X1 = []
    X2 = []
    X3 = []
    X11 = []
    X22 = []
    X33 = []
  
    for i in range(len(Y)):
        if(Y[i] < abs(bound)):
            Y_red = np.append(Y_red, Y[i])
            X11 = np.append(X11,X[i,0])
            X22 = np.append(X22,X[i,1])
            X33 = np.append(X33,X[i,2])
        else:
            Y_rep = np.append(Y_rep, Y[i])
            X1 = np.append(X1,X[i,0])
            X2 = np.append(X2,X[i,1])
            X3 = np.append(X3,X[i,2])
            zaehler+=1
            if(X[i,2] > zaehler2):
                zaehler2 = X[i,2]

    X11 = np.reshape(np.array(X11),(len(X11),1))
    X22 = np.reshape(np.array(X22),(len(X22),1))
    X33 = np.reshape(np.array(X33),(len(X33),1))
    X1 = np.reshape(np.array(X1),(len(X1),1))
    X2 = np.reshape(np.array(X2),(len(X2),1))
    X3 = np.reshape(np.array(X3),(len(X3),1))

    X_rep = np.concatenate((X1, X2), axis=1)
    X_rep = np.concatenate((X_rep, X3), axis=1)
    X_red = np.concatenate((X11, X22), axis=1)
    X_red = np.concatenate((X_red, X33), axis=1)
    Y_rep = np.reshape(Y_rep,(len(Y_rep),1))
    Y_red = np.reshape(Y_red,(len(Y_red),1))
    
    print("\t Reduce input data: Delete energies larger than " + str(bound) + " eV", flush=True)
    print("\t Input reduced by " + str(zaehler) + " elements", flush=True)
    print("\t You have " + str(len(Y_red)) + " elements left for training" , flush=True)
    #print("\t Elements will be replaced with an analytical expression", flush=True)
    print("", flush=True)
    return X_red, X_rep, Y_red, Y_rep
