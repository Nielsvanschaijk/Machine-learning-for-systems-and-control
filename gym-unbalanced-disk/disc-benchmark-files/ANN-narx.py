import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import matplotlib.pyplot as plt

# Load CSV file
data = pd.read_csv('./training-val-test-data.csv')
X = data['# u']
Y = data[' th']

# First split: 60% training, 40% temporary (validation + test)
Xtrain, Xtemp, Ytrain, Ytemp = train_test_split(X, Y, test_size=0.4, random_state=42)

# Second split: 50% of the temporary data for validation and test (20% each of original data)
Xval, Xtest, Yval, Ytest = train_test_split(Xtemp, Ytemp, test_size=0.5, random_state=42)
Xtrain, Xval, Xtest, Ytrain, Yval, Ytest = [torch.as_tensor(x.values).reshape(-1, 1) for x in [Xtrain, Xval, Xtest, Ytrain, Yval, Ytest]]

Xtrain = Xtrain.float()
Ytrain = Ytrain.float()
Xval = Xval.float()
Yval = Yval.float()
Xtest = Xtest.float()
Ytest = Ytest.float()

from torch import nn
import torch
class Network(nn.Module): #a)
    def __init__(self, n_in, n_hidden_nodes): #a)
        super(Network,self).__init__() #a)
        self.lay1 = nn.Linear(n_in,n_hidden_nodes).float() #a)
        self.lay2 = nn.Linear(n_hidden_nodes,1).float() #a)
    
    def forward(self,x): #a)
        #x = concatenated [upast and ypast] #a)
        x1 = torch.sigmoid(self.lay1(x)) #a)
        y = self.lay2(x1)[:,0] #a)
        return y #a)

n_hidden_nodes = 32 #a)
epochs = 100 #a)
model = Network(Xtrain.shape[1], n_hidden_nodes) #a=)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
Xtrain, Ytrain = Xtrain.to(device), Ytrain.to(device)
Xval, Yval = Xval.to(device), Yval.to(device)
train_loss_values = []
val_loss_values = []

optimizer = torch.optim.Adam(model.parameters()) #a)
Xtrain, Xval, Ytrain, Yval = [torch.as_tensor(x) for x in [Xtrain, Xval, Ytrain, Yval]] #convert it to torch arrays #a)
for epoch in range(epochs): #a)
    Loss = torch.mean((model(Xtrain)-Ytrain)**2) #a)
    optimizer.zero_grad() #a)
    Loss.backward() #a)
    optimizer.step() #a)
    #if epoch%1000==0: #a) monitor
    print(epoch,Loss.item()) #a)
    train_loss_values.append(Loss.item())

    Loss = torch.mean((model(Xval)-Yval)**2)
    val_loss_values.append(Loss.item())

torch.save(model.state_dict(), 'trained_model.pth')

plt.plot(train_loss_values)
plt.plot(val_loss_values)
plt.legend(["Train Loss", "Validation Loss"])
plt.show()
