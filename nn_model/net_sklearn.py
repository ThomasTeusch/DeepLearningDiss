import numpy as np

#--- Scikit Learn functions

def extra_trees(trees=2056, crit='mae', min_leaf=16):
    from sklearn.ensemble import ExtraTreesRegressor
    model = ExtraTreesRegressor(
                      n_estimators=trees, n_jobs=-1, criterion=crit, min_samples_leaf=min_leaf
            )
    model.name = "ExtraTreesRegressor_sklearn"
    print("\t Regressor: \t" + model.name, flush=True)
    print("\t Estimators:\t" + str(trees), flush=True)
    print("\t Sample split:\t" + str(min_leaf), flush=True)
    print("\t Criterion: \t" + crit, flush=True)
    print("", flush=True)
    return model
 
def gradient_boosting(loss_f='ls', crit='mae', boosting_stages=250, lr=0.1):
    from sklearn.ensemble import GradientBoostingRegressor
    model = GradientBoostingRegressor(
                loss=loss_f, criterion=crit, subsample=1.0,
                n_estimators=boosting_stages, learning_rate=lr, verbose=1
            )
    model.name = "GradientBoostingRegressor_sklearn"
    print(model)
    return model

#def neural_net(neurons=(64,128,512), max_iterations=10000, a=0.0001, activation):
def neural_net(neurons, max_iterations, a, activation):
    from sklearn.neural_network import MLPRegressor
    model = MLPRegressor(
                hidden_layer_sizes=neurons, max_iter=max_iterations,
                alpha=a, activation=activation, learning_rate="adaptive",
                solver='lbfgs', verbose=False, tol=0
            )
    model.name = "MLP_sklearn"
    print(model)
    return model

def kernel_ridge_regr():
    from sklearn.kernel_ridge import KernelRidge
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, RationalQuadratic, DotProduct, Matern
#    gp_kernel = RBF(0.1)
    gp_kernel = RationalQuadratic(length_scale=1.0, alpha=1e-10)
#    gp_kernel = DotProduct(sigma_0=1.0)
#    gp_kernel = Matern(length_scale=1.0, nu=0.5)
    model = GaussianProcessRegressor(kernel=gp_kernel, alpha=1e-10, n_restarts_optimizer=1)
    model.name = "KernelRidge_sklearn"
    print(model)
    return model

