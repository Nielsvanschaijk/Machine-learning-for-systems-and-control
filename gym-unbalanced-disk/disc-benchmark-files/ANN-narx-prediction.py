import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import linear_model
import torch
import matplotlib.pyplot as plt
import numpy as np

def create_IO_data(u,y,na,nb):
    X = []
    Y = []
    for k in range(max(na,nb), len(y)):
        X.append(np.concatenate([u[k-nb:k],y[k-na:k]]))
        Y.append(y[k])
    return np.array(X), np.array(Y)

out = np.load('training-val-test-data.npz')
th_train = out['th'] #th[0],th[1],th[2],th[3],...
u_train = out['u'] #u[0],u[1],u[2],u[3],...

data = np.load('hidden-test-prediction-submission-file.npz')
upast_test = data['upast'] #N by u[k-15],u[k-14],...,u[k-1]
thpast_test = data['thpast'] #N by y[k-15],y[k-14],...,y[k-1]

na = 5
nb = 5
X, Y = create_IO_data(u_train, th_train, na, nb)

Xtrain, Xval, Ytrain, Yval = train_test_split(X, Y, test_size=0.2, random_state=42)

Xtest = np.concatenate([upast_test[:,15-nb:], thpast_test[:,15-na:]],axis=1)

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

n_hidden_nodes = 32
epochs = 101
model = Network(Xtrain.shape[1], n_hidden_nodes)
train_loss_values = []
val_loss_values = []

optimizer = torch.optim.Adam(model.parameters())
Xtrain = torch.as_tensor(Xtrain)
Ytrain = torch.as_tensor(Ytrain)
Xval = torch.as_tensor(Xval)
Yval = torch.as_tensor(Yval)

batch_size = 256  # or whatever fits your memory
n_samples = Xtrain.shape[0]

for epoch in range(epochs):

    indices = torch.randperm(n_samples)

    for start_idx in range(0, n_samples, batch_size):
        end_idx = start_idx + batch_size
        batch_idx = indices[start_idx:end_idx]

        X_batch = Xtrain[batch_idx]
        Y_batch = Ytrain[batch_idx]

        # Forward pass
        pred = model(X_batch)
        loss = torch.mean((pred - Y_batch) ** 2)

        # Backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        train_loss = torch.mean((model(Xtrain) - Ytrain) ** 2).item()
        val_loss = torch.mean((model(Xval) - Yval) ** 2).item()
        train_loss_values.append(train_loss)
        val_loss_values.append(val_loss)
    
    model.train()

    if epoch % 10 == 0:
        print(f"Epoch {epoch}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")

torch.save(model.state_dict(), 'trained_model.pth')

plt.plot(train_loss_values)
plt.plot(val_loss_values)
plt.legend(["train loss", "validation loss"])
plt.show()

Xtest_tensor = torch.as_tensor(Xtest).double()
model.eval()
with torch.no_grad():
    Ypredict = model(Xtest_tensor).numpy()

np.savez('hidden-test-prediction-ANN-submission-file.npz', upast=upast_test, thpast=thpast_test, thnow=Ypredict)