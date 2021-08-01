#==================================================================================
# import
import sys
import numpy as np
import warnings
warnings.filterwarnings("ignore")
import argparse
import torch
import pre_processing.data_handling as dh

from pre_processing.data_handling import read_input, train_test_split, trans_def, inv_trans_def, inv_trans_def_torch, trans_MBB, inv_trans_MBB, trans_cos, inv_trans_cos, trans_exp, inv_trans_exp, trans_min_max, inv_trans_min_max, save_min_max, reduce_data_z, reduce_data_energy, r12_potential_energy, r12_potential_z, energies_to_gradients, gradients_to_energies, shuffle
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.externals import joblib
from pre_processing.utils import str2bool

def parse_bool(arg):
    arg = arg.lower()
    if 'true'.startswith(arg): return True
    elif 'false'.startswith(arg): return False
    else: raise ValueError()

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, argument_default=argparse.SUPPRESS)
parser.add_argument('--data-dir',          type=str,      default="inputs/red_root0_shuff23.dat", help='path to data file')
parser.add_argument('--module', '-m',      type=str,      default="pytorch", help='which module to use: sklearn, keras, PyTorch')
parser.add_argument('--project',           type=str,      default="",        help='Name of your project')
parser.add_argument('--model-name', '-n',  type=str,      default="mlp",     help='which model to use: rf, gb, mlp, krr, resnetv3')
parser.add_argument('--activation', '-a',  type=str,      default="relu",    help='activation function for neural network')
parser.add_argument('--test-split',        type=float,    default=0.15,      help='fraction of data for testing')
parser.add_argument('--dropout',           type=float,    default=0.0,       help='only for PyTorch: Apply dropout rate')
parser.add_argument('--learning',          type=float,    default=1.0,       help='only for PyTorch: Initial learning rate')
parser.add_argument('--dim',               type=int,      default=64,        help='MLP: Size of dimension after first layer, RF: Number of Estimators')
parser.add_argument('--layer',             type=int,      default=3,         help='MLP: Number of layers')
parser.add_argument('--boundary',          type=float,    default=7.0,       help='boundary for reducing dataset at which data is deleted')
parser.add_argument('--optimizer',         type=str,      default='lbfgs',   help='only in PyTorch: Choose for optimizer: LBFGS, Adam, SGD, RMSProp')
parser.add_argument('--modus',             type=str,      default='mse',     help='only in PyTorch: Choose for loss func: MSE, MAE, mod_mse_grad, mse_huber')   #, mod_mse_energy')
parser.add_argument('--cuda',              type=str2bool, default=True,      help='enables CUDA training')
parser.add_argument('--training', '-t',    type=str2bool, default=True,      help='True, if you want to train your neural network')
parser.add_argument('--prediction', '-p',  type=str2bool, default=True,      help='True, if you want to make an interpolation')
parser.add_argument('--gradients', '-g',   type=str2bool, default=True,      help='True, if you want to fit for gradients(z) instead of energies')
parser.add_argument('--red-data',          type=int,      default=2,         help="""(1) if one wants to reduce data set by Z value,\
                                                                                     (2) if one wants to reduce data set by energy,\
               				                                             (other) if one does not want to reduce data set.""")
args = parser.parse_args()
args.cuda = args.cuda and torch.cuda.is_available()  
torch.cuda.empty_cache()

if(not args.gradients and args.modus == 'mod_mse_grad'): print("Serious Warning! Loss function mod_mse_grad is not suitable for energy training. Gradients only!") ; exit()
#if(args.gradients and args.modus == 'mod_mse_energy'): print("Serious Warning! Loss function mod_mse_energy is not suitable for gradient training. Energies only!") ; exit()

if(args.gradients == True):
    if("root0" in args.data_dir): 
        file_name = 'gs_%s_%s_%seV_grad_%s_%s_%s_lr%s_dim%s_layer%s_drop%s' % (args.module, args.model_name, args.boundary, args.modus, args.activation, args.optimizer, args.learning, args.dim, args.layer, args.dropout)
    else: 
        file_name = 'es_%s_%s_%seV_grad_%s_%s_%s_lr%s_dim%s_layer%s_drop%s' % (args.module, args.model_name, args.boundary, args.modus, args.activation, args.optimizer, args.learning, args.dim, args.layer, args.dropout)
else:
    if("root0" in args.data_dir): 
        file_name = 'gs_%s_%s_%seV_no_grad_%s_%s_%s_lr%s_dim%s_layer%s_drop%s' % (args.module, args.model_name, args.boundary, args.modus, args.activation, args.optimizer, args.learning, args.dim, args.layer, args.dropout)
    else: 
        file_name = 'es_%s_%s_%seV_no_grad_%s_%s_%s_lr%s_dim%s_layer%s_drop%s' % (args.module, args.model_name, args.boundary, args.modus, args.activation, args.optimizer, args.learning, args.dim, args.layer, args.dropout)
if(args.red_data == 1): file_name = file_name.replace("eV", "z", 1)

print('\t HYPERPARAMETERS:', flush=True)
print('\t Model: %s, #Layer: %s, #Neurons: %s' % (args.model_name, args.layer, args.dim))
print('\t Optimizer: %s, Activation: %s, Loss: %s, Dropout: %s' % (args.optimizer, args.activation, args.modus, args.dropout)) 

if(args.training == True):
    #==================================================================================
    print("", flush=True)
    print("\t --------------- STARTING FITTING PROCEDURE ---------------", flush=True)
    print("", flush=True)
    
    # read data assuming one has 3 pattern, one energy and one gradient column
    X,Y,w = read_input(args.data_dir, 3, 4)
    eV = 27.211383
    E_ref = -3648.512337     #-13211.647242  # energy reference at far distance
    Y = (Y - E_ref) * eV
    minE1 = min(Y)   # get energy minimum ( -> check latter, if in training data and not test data)
    
    #====================================
    # reduce data set size
    if (args.red_data == 1) :
        print("Reducing input data set in Z dimension...")
        X, X_repulsive, Y, Y_repulsive = reduce_data_z(X, Y, bound=args.boundary)
        res = np.concatenate((X,Y), axis=1)
        np.savetxt('./data/potential_cut_training_%s.csv' % (file_name), res)
        res = np.concatenate((X_repulsive,Y_repulsive), axis=1)
        np.savetxt('./data/potential_cut_repulsiv_%s.csv' % (file_name), res)
    elif (args.red_data == 2) :
        print("\t Data reduction for energy not yet fully implemented!", flush=True)
        X, X_repulsive, Y, Y_repulsive = reduce_data_energy(X, Y, bound=args.boundary)
        res = np.concatenate((X,Y), axis=1)
        np.savetxt('./data/potential_cut_training_%s.csv' % (file_name), res)
        res = np.concatenate((X_repulsive,Y_repulsive), axis=1)
        np.savetxt('./data/potential_cut_repulsiv_%s.csv' % (file_name), res)
    else: print("\t No data reduction procedure applied!", flush=True)
    
    # sort data set for prediction
    ori = np.concatenate((X,Y),axis=1)
    ori = ori[np.lexsort((ori[:,2], ori[:,1],ori[:,0]))]
   
    if(args.gradients):  
        print("\n \t Transfer energies to gradients. This is a very good choice! \n", flush=True)
        X, Y, save_asympt = energies_to_gradients(ori)
        X, Y = shuffle(X,Y, seed=42)
    else: print("\n \t No transformation of energy to gradient data applied! \n", flush=True)

    #====================================
    # split into training and test set
    X_train, X_test, Y_train, Y_test, w_train, w_test = train_test_split(X,Y,w,args.test_split)
    minE2 = np.min(Y_train)
    minE2pos = np.argmin(Y_train)
    
    if (not args.gradients and  minE1 != minE2 ): print("\t ERROR! Energy minimum of input file not in training data set. \n \t New data shuffle needed. \n") ; exit()
    
    #====================================
    # transform input data
    print("\t Transforming input data...", flush=True)
    print("\t Assuming the following column structure:  THETA   Y   Z   ENERGY  WEIGHT", flush=True)
    
    print("\t   Theta -> default [-1:1]", flush=True)
    minTheta = X_train[:,0].min(0)
    maxTheta = X_train[:,0].max(0)
    print("\t\t Min: %s" % (minTheta), flush=True)
    print("\t\t Max: %s" % (maxTheta), flush=True)
    trX_trainTheta = trans_def(X_train[:,0], minTheta, maxTheta)
    trX_testTheta = trans_def(X_test[:,0], minTheta, maxTheta)
    
    print("\t   Y -> default [-1:1]", flush=True)
    minY = X_train[:,1].min(0) # minimum in translation Y
    maxY = X_train[:,1].max(0) # maximum in translation Y
    print("\t\t Min: %s" % (minY), flush=True)
    print("\t\t Max: %s" % (maxY), flush=True)
    trX_trainY = trans_def(X_train[:,1], minY, maxY)
    trX_testY = trans_def(X_test[:,1], minY, maxY)
    
    print("\t   Z -> Morse (MBB)", flush=True)
    z_eq = X_train[minE2pos,2] # 
    gamma = 1.00            # adjust according to 1D(Z)
    print("\t\t Z_eq:  %s" % (z_eq), flush=True)
    print("\t\t Gamma: %s" % (gamma), flush=True)
    trX_trainZ = trans_MBB(X_train[:,2], z_eq, gamma)
    trX_testZ = trans_MBB(X_test[:,2], z_eq, gamma)
    
    # combine transformed arrays
    trX_train = np.column_stack( (trX_trainTheta, trX_trainY, trX_trainZ) )
    trX_test = np.column_stack( (trX_testTheta, trX_testY, trX_testZ) )
    
    Y_train_orig = Y_train.copy()

    joblib.dump({"minY": minY, "maxY": maxY, "minTheta": minTheta, "maxTheta": maxTheta, "z_eq": z_eq, "gamma": gamma}, "./models/parameters_transformation_%s.pkl" % file_name)
    print("\t Transformation finished.\n",flush=True)

    #==================================================================================
    # modules and models
    if ("sk" in args.module.lower()) :
        if "rf" == args.model_name.lower():
           from nn_model.net_sklearn import extra_trees
           model = extra_trees(trees=args.dim, crit=args.modus, min_leaf=16)
        elif "gb" == args.model_name.lower():
           from nn_model.net_sklearn import gradient_boosting
           model = gradient_boosting(loss_f='ls', crit='mae', boosting_stages=250, lr=0.1)
        elif "mlp" == args.model_name.lower():
           from nn_model.net_sklearn import neural_net
           model = neural_net(neurons=(64, 64), max_iterations=1000, a=0.0001, activation=args.activation)
        elif "krr" == args.model_name.lower():
           from nn_model.net_sklearn import kernel_ridge_regr
           model = kernel_ridge_regr()
        else: print("\t Error! No model with name " + args.model_name + " found in sklearn! \n", flush=True) ; exit()
    elif ("keras" in args.module.lower()) :
       if "mlp" == args.model_name.lower():
           from nn_model.net_keras import neural_net
           model = neural_net(learning_rate=0.01, neurons=(64, 128, 64), act_func=('relu', 'relu', 'relu'))
       else: print("\t Error! No model with name " + args.model_name + " found in keras! \n", flush=True) ; exit()
    elif ("pytorch" in args.module.lower()) :
        if "mlp" == args.model_name.lower():
            from nn_model.net_pytorch import mlp_pt
            if(args.cuda): print("\t CUDA support activated")
            model = mlp_pt(dim=args.dim, act=args.activation, layer=args.layer, drop=args.dropout, use_cuda=args.cuda, name='mlp_pytorch', optim=args.optimizer, modus=args.modus, suffix=args.project)
            print(model, flush=True)
            print("", flush=True)
        elif "resnetv3" == args.model_name.lower():
            from nn_model.net_pytorch import ResNet18v3
            if(args.cuda): print("\t CUDA support activated")
            model = ResNet18v3(input_dim=3, drop=args.dropout, act=args.activation, use_cuda=args.cuda, name='resnetv3_pytorch', optim=args.optimizer, modus=args.modus, suffix=args.project)
        else: print("\t Error! No model with name " + args.model_name + " found in PyTorch! \n", flush=True) ; exit()
    else: print("\t Error! No module with name " + args.module + " found! \n", flush=True) ; exit()
    
    # Fit Data, save trained model, predict on train data
    if "sklearn" in model.name:
        model.fit(trX_train, Y_train)
        joblib.dump(model, './models/sklearn/%s_model_%s.pkl' % (model.name, args.module))
    elif "keras" in model.name:
        model.fit(trX_train, Y_train, epochs=1000, batch_size=Y_train.shape[0], verbose=0)
        model.save('./models/keras/%s_model.h5' % (model.name))
    elif "pytorch" in model.name:
        model.fit(trX_train, Y_train, epochs=2000, lr=args.learning, optim=args.optimizer, modus=args.modus, suffix=file_name)
    else: print("\t Error! Something serious went wrong! Error in fit", flush=True) ; exit()
    
    Y_pred = model.predict(trX_train)
    Y_pred = np.reshape(Y_pred, Y_train_orig.shape)
    
    #==================================================================================
    # final fit result
    print("\n \t----------------- FIT PERFORMANCE --------------------", flush=True)
    print("\tMean squared error (train): %.8f" % mean_squared_error(Y_train_orig[:, [0]], Y_pred[:, [0]]),flush=True)
    print("\tMean absolute error (train): %.8f" % mean_absolute_error(Y_train_orig[:, [0]], Y_pred[:, [0]]),flush=True)
    print("\tVariance score (train): %.8f" % r2_score(Y_train_orig[:, [0]], Y_pred[:, [0]]),flush=True)
    print("\t------------------------------------------------------ \n", flush=True)
    ftrain = open('./data/comp_train_%s.dat' % (file_name),"w+")
    for i in range(len(Y_train_orig)):
       ftrain.write("\t{0:>12.8f} \t{1:>12.8f}\n".format(Y_train_orig[i][0], Y_pred[i][0]))
    ftrain.close()
    
    if args.test_split > .0:
        Y_pred = model.predict(trX_test)
        Y_pred = np.reshape(Y_pred, Y_test.shape)
        print("\t----------------- TEST PERFORMANCE -------------------", flush=True)
        print("\tMean squared error (test): %.8f" % mean_squared_error(Y_test[:, [0]], Y_pred[:, [0]]),flush=True)
        print("\tMean absolute error (test): %.8f" % mean_absolute_error(Y_test[:, [0]], Y_pred[:, [0]]),flush=True)
        print("\tVariance score (test): %.8f" % r2_score(Y_test[:, [0]], Y_pred[:, [0]]),flush=True)
        print("\t------------------------------------------------------ \n", flush=True)

        ftest = open('./data/comp_train_%s.dat' % (file_name),"w+")
        for i in range(len(Y_test)):
           ftest.write("\t{0:>12.8f} \t{1:>12.8f}\n".format(Y_test[i][0], Y_pred[i][0]))
        ftest.close()
        print("", flush=True)
        

    #==================================================================================
    # predict energy at input data points
    # print file with differences between ground truth and prediction
    if args.test_split > .0:
        ori_X = np.delete(ori, 3, axis=-1)
      
        oriX_Theta = trans_def(ori_X[:,0], minTheta, maxTheta)
        oriX_Y = trans_def(ori_X[:,1], minY, maxY)
        oriX_Z = trans_MBB(ori_X[:,2], z_eq, gamma)
        ori_X2 = np.column_stack( (oriX_Theta, oriX_Y, oriX_Z) )
     
        Y_pred = model.predict(ori_X2)
        Y_pred = np.reshape(Y_pred, (len(ori_X),1))
       
        if(args.gradients):
            ori_X, Y_pred = gradients_to_energies(ori_X, Y_pred, save_asympt)

        fo = open('./data/train_pred_%s.dat' % (file_name),"w+")
        fo.write(" Theta / ° \t Y / \AA \t Z / \AA \t E_pred / eV \t E_true / eV \t Difference / eV\n")
        for i in range(len(Y_pred)):
            if ori_X[i][1] != ori_X[i-1][1] and i != 0:
                fo.write('\n')
            if ori_X[i][0] != ori_X[i-1][0] and i != 0:
                fo.write('\n')
            fo.write("{0:8.2f} \t{1:>8.3f} \t{2:>8.3f} \t{3:>12.8f} \t{4:>12.8f} \t{5:>8.4f}\n".format(ori_X[i][0], ori_X[i][1], ori_X[i][2], Y_pred[i][0], ori[i][3], (Y_pred[i][0]-ori[i][3])))
        fo.close()  

        if(args.gradients):
            print("", flush=True)
            print("\tMSE: %.8f" % mean_squared_error(ori[:, [3]], Y_pred[:, [0]]),flush=True)
            print("\tVariance: %.8f \n" % r2_score(ori[:, [3]], Y_pred[:, [0]]),flush=True)

#if(args.prediction):
#    #==================================================================================
#    # predict interpolation
#    if(args.training == False):
##        norm_dict = joblib.load("./models/vals_split_%s.pkl" % args.data_dir[7:-4])
#        minY = (norm_dict["minY"])
#        maxY = (norm_dict["maxY"])
#        minTheta = (norm_dict["minTheta"])
#        maxTheta = (norm_dict["maxTheta"])
#        z_eq = (norm_dict["z_eq"])
#        gamma = (norm_dict["gamma"])
#     
#        #model = load_model(module=args.module.lower(), model=args.model_name.lower())
#   
#    read_from_file=False
#
#    if(read_from_file):
#         #Read from pot.points
#        f = open("inputs/pot.points", "r")
#        data = [[np.float64(c) for c in line.split()] for line in f.readlines()]
#        f.close()
#        data = np.array(data, dtype=np.float64)
#        grid = np.delete(data, 1, axis=-1)
#    else:
#        b_theta=[0, 180, 30]
#        b_y=[-6.6, 2.6, 256]
#        b_z=[3.0, 35.0, 512]
#
#        step_t=(b_theta[1]-b_theta[0])/b_theta[2]
#        step_y=(b_y[1]-b_y[0])/b_y[2]
#        step_z=(b_z[1]-b_z[0])/b_z[2]
#
#        dim = (b_theta[2]+1)*(b_y[2]+1)*(b_z[2]+1)
#        grid = np.zeros((dim, 3))
#        counter = 0
#        
#        for i in range(b_theta[2]+1):
#            for j in range(b_y[2]+1):
#                for k in range(b_z[2]+1):
#                    grid[counter][0] = b_theta[0] + i*step_t
#                    grid[counter][1] = b_y[0] + j*step_y
#                    grid[counter][2] = b_z[0] + k*step_z
#                    counter+=1
#       
#    grid_Theta = trans_def(grid[:,0], minTheta, maxTheta)
#    grid_Y = trans_def(grid[:,1], minY, maxY)
#    grid_Z = trans_MBB(grid[:,2], z_eq, gamma)
#    grid_pred = np.column_stack((grid_Theta, grid_Y, grid_Z))
#    Y_pred = model.predict(grid_pred)
#    Y_pred = np.reshape(Y_pred, (len(grid),1))
# 
#    if(args.gradients):
#        grid, Y_pred = gradients_to_energies(grid, Y_pred, save_asympt)
#    
#    fo = open('./data/interpolation_%s.dat' % (file_name),"w+") 
#    fo.write(" Theta / ° \t Y / \AA \t Z / \AA \t E_pred / eV \n")
#    for i in range(len(Y_pred)):
#        if grid[i][1] != grid[i-1][1] and i != 0:
#            fo.write('\n')
#        if grid[i][0] != grid[i-1][0] and i != 0:
#            fo.write('\n')
#        fo.write("{0:8.2f} \t{1:>8.3f} \t{2:>8.3f} \t{3:>12.8f} \n".format(grid[i][0], grid[i][1], grid[i][2], Y_pred[i][0])) 
#    fo.close()

del model, Y_pred, ori_X, ori_X2, oriX_Theta, oriX_Y, oriX_Z, X, Y, ori, X_train, X_test, Y_train, Y_test, w_train, w_test, trX_trainTheta, trX_testTheta, trX_trainY, trX_testY, trX_trainZ, trX_testZ, trX_train, trX_test, Y_train_orig

#del model, Y_pred, grid, ori_X, ori_X2, oriX_Theta, oriX_Y, oriX_Z, X, Y, ori, X_train, X_test, Y_train, Y_test, w_train, w_test, trX_trainTheta, trX_testTheta, trX_trainY, trX_testY, trX_trainZ, trX_testZ, trX_train, trX_test, Y_train_orig
torch.cuda.empty_cache()
import gc
gc.collect()

