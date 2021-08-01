import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from sklearn.externals import joblib

def trans_MBB_torch(data, z_eq, gamma):
    return ( 1 - np.exp(-gamma*( (data-z_eq)/z_eq )) ) * 1/gamma

def loss_fn(output, target, pattern, modus='mse'):
    if(modus.lower() == 'mse'): loss = ((output - target) ** 2).mean()
    elif(modus.lower() == 'mse_huber'): 
        crit = torch.nn.SmoothL1Loss()
        loss = crit(output,target)
    elif(modus.lower() == 'mae'): loss = (abs((output - target))).mean()
    else: print("Error! Unsupported loss_function %s" % modus) ; exit()
    return loss

def activation(f):
    if f.lower() == "leaky_relu": return nn.LeakyReLU()
    elif f.lower() == "relu": return nn.ReLU()
    elif f.lower() == "selu": return nn.SELU()
    elif f.lower() == "celu": return nn.CELU()
    elif f.lower() == "rrelu": return nn.RReLU()
    elif f.lower() == "tanh": return nn.Tanh()
    elif f.lower() == "sigmoid": return nn.Sigmoid()
    elif f.lower() == "tanhshrink": return nn.Tanhshrink()
    elif f.lower() == "softsign": return nn.Softsign()
    elif f.lower() == "hardtanh": return nn.Hardtanh()
    else: print("Error! Unsupported non linearity %s" % f) ; exit()

def fit(model, X, Y, epochs=100, lr=1., weight_decay=1e-5, optim='LBFGS', modus='mse', suffix=''):
    X = torch.tensor(X) ; Y = torch.tensor(Y) ; torch.manual_seed(42)
   
    act_lr=lr 

    if model.use_cuda:
        model.cuda()
        X, Y = X.cuda(), Y.cuda()

    if(optim.lower() == 'lbfgs'): optimizer = torch.optim.LBFGS(model.parameters(), lr=lr) 
    elif(optim.lower() == 'adam'): optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), amsgrad=True, weight_decay=1e-3)
    elif(optim.lower() == 'rmsprop'): optimizer = torch.optim.RMSprop(model.parameters(), lr=lr)
    elif(optim.lower() == 'sgd'): optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, dampening=0.0, nesterov=True)

    print("", flush=True)
    print("\t START TRAINING OF NEURAL NETWORK IN PYTORCH \n", flush=True)
    print("" + str(optimizer) + "\n", flush=True)
    print("\t Train for %s" % modus, flush=True)
    print("\t Use the optimizer %s \n" % optim, flush=True)

    best_loss = np.inf
    early_stop = 0
    lr_run = 0

    for epoch in range(1, epochs + 1):
        model.train()
        loss_epoch = 0

        if ("LBFGS" or "RMSPROP") in str(optimizer):
            def closure():
                optimizer.zero_grad()
                output = model(X)
                loss = loss_fn(output, Y, X, modus)
                loss.backward()
                return loss
            loss = optimizer.step(closure)
            loss_epoch += loss.item()
        else:
            optimizer.zero_grad()
            output = model(X)
            loss = loss_fn(output, Y, X, modus)
            loss.backward()
            optimizer.step()
            loss_epoch += loss.item()

        model.eval()
        with torch.no_grad():
            eval_loss_mse = 0 ; eval_loss = 0 ; eval_loss_mae = 0 ; output = 0
            output = model(X)
            eval_loss_mse = ((output - Y)**2).mean()
            eval_loss_mae = torch.abs(output - Y).mean()
            eval_loss = eval_loss_mse**.5

            if "mse" in modus:
                print('\t Train Epoch: {} \t\tMSE: {:.6f} \tEval: {:.6f}(MSE) {:.6f}(RMSE) {:.6f}(MAE) \t LR: {:.6f}'.format(
                                       epoch, loss_epoch, eval_loss_mse, eval_loss, eval_loss_mae, optimizer.param_groups[0]['lr']), flush=True)

                if best_loss > eval_loss_mse:
                    print('\t Better model found and saved')
                    best_loss = eval_loss_mse
                    torch.save(model.state_dict(),'./models/pytorch/model_%s.pt' % (suffix))
                    early_stop = 0
                else:
                    lr_run = 0
                    early_stop = early_stop+1

            elif "mae" in modus:
                print('\t Train Epoch: {} \t\tMAE: {:.6f} \tEval: {:.6f}(MAE) {:.6f}(MSE) {:.6f}(RMSE) \t LR: {:.6f}'.format(
                                       epoch, loss_epoch, eval_loss_mae, eval_loss_mse, eval_loss, optimizer.param_groups[0]['lr']), flush=True)

                if best_loss > eval_loss_mae: 
                    print('\t Better model found and saved')
                    best_loss = eval_loss_mae
                    torch.save(model.state_dict(),'./models/pytorch/model_%s.pt' % (suffix))
                    early_stop = 0
                    lr_run = lr_run+1
                else:
                    lr_run = 0
                    early_stop = early_stop+1
            
        if early_stop > 150:
            print('\t No better model found since 150 iterations! Apply early stopping procedure')
            break

        #optimizer.step()
        model.load_state_dict(torch.load('./models/pytorch/model_%s.pt' % (suffix)))

        lr_modus='fixed'
      
        if(lr_modus == 'adaptive'):
            if( early_stop == 20):
                print("\t Change LR: LR/1.5")
                act_lr = 1.5/act_lr
                for param_group in optimizer.param_groups:
                    param_group['lr'] = act_lr
                early_stop = 0
            elif( lr_run == 20):
                if(act_lr*1.5 < 2):
                    print("\t Change LR: LR*1.5")
                    act_lr = 1.5*act_lr
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = act_lr
                lr_run = 0     
        elif(lr_modus == 'fixed'):
            if( epoch % 500 == 0):
                print("\t Change LR: 0.5 * LR")
                act_lr = 0.5 *act_lr
                for param_group in optimizer.param_groups:
                    param_group['lr'] = act_lr
        else: print("Error! Unsupported lr_modus %s" % lr_modus) ; exit()

    model.load_state_dict(torch.load('./models/pytorch/model_%s.pt' % (suffix)))
    return model

def predict(model, x):
    model.eval()
    with torch.no_grad():
        x = Variable(torch.tensor(x))
        if model.use_cuda: x = x.cuda()
        output = model(x)
    return output.cpu().numpy()
   
class mlp_pt(nn.Module):
    def __init__(self, dim=64, layer=3, act='reLU', drop=0.2, use_cuda=False, name='mlp_pytorch', optim='LBFGS', modus='mse', suffix=''):
        """
        In this constructor the neural network is created. The parameter are explained below:

            input_dim:  Number of dimensions of input data (here 3: theta, y, z)
            dim:        Number of dimensions of first layer
            act:        Activation function for the layers
            use_cuda:   Whether or not to use GPU acceleration
            name:       Name of the NN
        """
        super().__init__()
        input_dim = 3
        self.use_cuda = use_cuda
        self.name = name
        self.modus = modus
        self.optim = optim
        self.suffix = suffix

        if( layer == 2 ):
            self.model = nn.Sequential(
                nn.Linear(input_dim, dim), activation(act), nn.Dropout(p=drop),
                nn.Linear(dim, dim*2), activation(act), nn.Dropout(p=drop),
                nn.Linear(dim*2, dim*2), activation(act), nn.Dropout(p=drop),
                nn.Linear(dim*2, dim)
            )
        elif( layer == 3 ):
            self.model = nn.Sequential(
                nn.Linear(input_dim, dim), activation(act), nn.Dropout(p=drop),
                nn.Linear(dim, dim*2), activation(act), nn.Dropout(p=drop),
                nn.Linear(dim*2, dim*3), activation(act), nn.Dropout(p=drop),
                nn.Linear(dim*3, dim*2), activation(act), nn.Dropout(p=drop),
                nn.Linear(dim*2, dim)
            )
        elif( layer == 4 ):
            self.model = nn.Sequential(
                nn.Linear(input_dim, dim), activation(act), nn.Dropout(p=drop),
                nn.Linear(dim, dim*2), activation(act), nn.Dropout(p=drop),
                nn.Linear(dim*2, dim*3), activation(act), nn.Dropout(p=drop),
                nn.Linear(dim*3, dim*4), activation(act), nn.Dropout(p=drop),
                nn.Linear(dim*4, dim*2), activation(act), nn.Dropout(p=drop),
                nn.Linear(dim*2, dim)
            )
        elif( layer == 5 ):
            self.model = nn.Sequential(
                nn.Linear(input_dim, dim), activation(act), nn.Dropout(p=drop),
                nn.Linear(dim, dim*2), activation(act), nn.Dropout(p=drop),
                nn.Linear(dim*2, dim*3), activation(act), nn.Dropout(p=drop),
                nn.Linear(dim*3, dim*4), activation(act), nn.Dropout(p=drop),
                nn.Linear(dim*4, dim*3), activation(act), nn.Dropout(p=drop),
                nn.Linear(dim*3, dim*2), activation(act), nn.Dropout(p=drop),
                nn.Linear(dim*2, dim)
            )

        self = self.double()

        joblib.dump({"dim": dim, "act": act, "drop": drop},
             "./models/pytorch/parameters_dim%s_%s_drop%s.pkl" % (dim, act, drop))
        name='pytorch_dim%s_%s_drop%s' %  (dim, act, drop)

    def forward(self, x):
        return self.model(x).mean(-1, keepdim=True)

    def fit(self, X, Y, epochs=10000, lr=0.01, optim='LBFGS', modus='mse', suffix=''):
        return fit(self, X, Y, epochs, lr=lr, optim=optim, modus=modus, suffix=suffix)

    def predict(self, x):
        return predict(self, x)

class ResNet18v3(nn.Module):
    def __init__(self, input_dim=2, drop=0.2, use_cuda=False, name="ResNet18", act='leaky_relu', optim='adam', modus='mse', suffix=''):
        super().__init__()
        self.name = name
        self.use_cuda = use_cuda
        self.optim = optim
        self.suffix = suffix

        self.model = nn.Sequential(
            nn.Linear(input_dim, 8),
            DenseBlock(8, 16, 3, act=act),
            DenseBlock(16, 32, 6, act=act),
            DenseBlock(32, 96, 12,act=act),
            DenseBlock(96, 128, 8,act=act),
            nn.Dropout(p=drop),
            activation(act),
            nn.Linear(128, 512)
        )

#        self.model = nn.Sequential(
#            nn.Linear(input_dim, 16),
#            DenseBlock(16, 32, 6, act=act),
#            DenseBlock(32, 64, 12, act=act),
#            DenseBlock(64, 192, 24,act=act),
#            DenseBlock(192, 256, 16,act=act),
#            nn.Dropout(p=drop),
#            activation(act),
#            nn.Linear(256, 1024)
#        )
        self = self.double()

    def forward(self, X):
        return self.model(X).mean(-1, keepdim=True)

    def fit(self, X, Y, epochs=10000, lr=1., optim='adam', modus='mse', suffix=''):
        return fit(self, X, Y, epochs, lr=lr, optim=optim, modus=modus, suffix=suffix)

    def predict(self, X):
        return predict(self, X)

# for resnet18_v3
class DenseBlock(nn.Module):
    def __init__(self, dim, dim_out, n_layers, act="leaky_relu"):
        super(DenseBlock, self).__init__()
        self.act = act
        self.lin = nn.ModuleList()

        for i in range(1, n_layers+1):
            self.lin.append(nn.Sequential(
                activation(act),
                nn.Linear(dim*i, dim),
            ))
            self.trans = nn.Sequential(
                activation(act),
                nn.Linear(dim*(n_layers+1), dim_out)
            )

    def forward(self, x):
        res = x
        for lin in self.lin:
            out = lin(res)
            res = torch.cat([out, res], 1)
        out = self.trans(res)
        return out

