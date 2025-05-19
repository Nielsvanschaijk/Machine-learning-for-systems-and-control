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

data = np.load('hidden-test-simulation-submission-file.npz')
u_test = data['u']
th_test = data['th'] #only the first 50 values are filled the rest are zeros

na = 2
nb = 3
X, Y = create_IO_data(u_train, th_train, na, nb)

Xtrain, Xval, Ytrain, Yval = train_test_split(X, Y, test_size=0.2, random_state=42)

from torch import nn
import torch
import torch.nn.functional as F

class Network(nn.Module):
    def __init__(self, n_in):
        super(Network, self).__init__()
        self.lay1 = nn.Linear(n_in, 64).double()
        self.bn1 = nn.LayerNorm(64).double()
        self.lay2 = nn.Linear(64, 32).double()
        self.bn2 = nn.LayerNorm(32).double()
        self.lay3 = nn.Linear(32, 16).double()
        self.bn3 = nn.LayerNorm(16).double()
        self.lay4 = nn.Linear(16, 1).double()

        #self.dropout = nn.Dropout(p=0.1)  # You can increase p if overfitting

    def forward(self, x):
        x = F.relu(self.bn1(self.lay1(x)))
        x = F.relu(self.bn2(self.lay2(x)))
        x = F.relu(self.bn3(self.lay3(x)))
        #x = self.dropout(F.relu(self.bn2(self.lay2(x))))
        #x = self.dropout(F.relu(self.bn3(self.lay3(x))))
        y = self.lay4(x).squeeze(-1)  # output shape: (batch_size,)
        return y

epochs = 1001
model = Network(Xtrain.shape[1])
train_loss_values = []
val_loss_values = []

optimizer = torch.optim.Adam(model.parameters())
Xtrain = torch.as_tensor(Xtrain)
Ytrain = torch.as_tensor(Ytrain)
Xval = torch.as_tensor(Xval)
Yval = torch.as_tensor(Yval)

batch_size = 512  # or whatever fits your memory
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

model.eval()
torch.save(model.state_dict(), 'trained_model_sim_adv.pth')

plt.plot(train_loss_values)
plt.plot(val_loss_values)
plt.legend(["train loss", "validation loss"])
plt.show()

def simulation_IO_model(model, ulist, ylist, na, nb, skip=50):
    upast = ulist[skip - nb:skip].tolist()
    ypast = ylist[skip - na:skip].tolist()
    Y = ylist[:skip].tolist()
    
    for u in ulist[skip:]:
        x = np.concatenate([upast, ypast])[None, :]  # shape (1, na+nb)
        x_tensor = torch.as_tensor(x).double()
        model.eval()
        with torch.no_grad():
            ypred = model(x_tensor).item()
        Y.append(ypred)
        upast.append(u)
        upast.pop(0)
        ypast.append(ypred)
        ypast.pop(0)
    return np.array(Y)

skip = 50  # you already know the first 50 outputs
th_test_sim = simulation_IO_model(model, u_test, th_test, na, nb, skip=skip)

np.savez('hidden-test-simulation-ANN-adv-submission-file.npz', th=th_test_sim, u=u_test)