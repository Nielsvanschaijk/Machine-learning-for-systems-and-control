import numpy as np

out = np.load('training-val-test-data.npz')
th_train = out['th'] #th[0],th[1],th[2],th[3],...
u_train = out['u'] #u[0],u[1],u[2],u[3],...

# data = np.load('test-prediction-submission-file.npz')
data = np.load('hidden-test-prediction-submission-file.npz')
upast_test = data['upast'] #N by u[k-15],u[k-14],...,u[k-1]
thpast_test = data['thpast'] #N by y[k-15],y[k-14],...,y[k-1]
# thpred = data['thnow'] #all zeros


def create_IO_data(u,y,na,nb):
    X = []
    Y = []
    for k in range(max(na,nb), len(y)):
        X.append(np.concatenate([u[k-nb:k],y[k-na:k]]))
        Y.append(y[k])
    return np.array(X), np.array(Y)

na = 5
nb = 5
Xtrain, Ytrain = create_IO_data(u_train, th_train, na, nb)

from sklearn import linear_model
# reg = linear_model.LinearRegression()
# reg.fit(Xtrain,Ytrain)

from torch import nn
import torch
class Network(nn.Module):
    def __init__(self, n_in, n_hidden_nodes):
        super(Network,self).__init__()
        self.lay1 = nn.Linear(n_in,n_hidden_nodes).double()
        self.lay2 = nn.Linear(n_hidden_nodes,1).double()
    
    def forward(self,x):
        x1 = torch.sigmoid(self.lay1(x))
        y = self.lay2(x1)[:,0]
        return y
model = Network(na + nb, 32)  # Use the same architecture as trained
model.load_state_dict(torch.load('trained_model.pth'))
model.eval()

Xtrain = torch.as_tensor(Xtrain)
# Ytrain = torch.as_tensor(Ytrain)


with torch.no_grad():
    Ytrain_pred = model(Xtrain).numpy().flatten()
print("len xtrain", Xtrain.shape)
print("len ytrain", Ytrain.shape)
print("len ytrainpred", Ytrain_pred.shape)
# Ytrain_pred = reg.predict(Xtrain)
print('train prediction errors:')
print('RMS:', np.mean((Ytrain_pred-Ytrain)**2)**0.5,'radians')
print('RMS:', np.mean((Ytrain_pred-Ytrain)**2)**0.5/(2*np.pi)*360,'degrees')
print('NRMS:', np.mean((Ytrain_pred-Ytrain)**2)**0.5/Ytrain.std()*100,'%')

 #only select the ones that are used in the example
Xtest = np.concatenate([upast_test[:,15-nb:], thpast_test[:,15-na:]],axis=1)

Xtest = torch.as_tensor(Xtest)
with torch.no_grad():
    Ypredict = model(Xtest).numpy()
# Ypredict = reg.predict(Xtest)
print("shape", Ypredict.shape)
print("lengtes")
print(len(Ypredict))
print(len(upast_test))
assert len(Ypredict)==len(upast_test), 'number of samples changed!!'

np.savez('hidden-test-prediction-example-submission-file.npz', upast=upast_test, thpast=thpast_test, thnow=Ypredict)