import torch
import numpy as np
import matplotlib.pyplot as plt
from torch import nn
weight = torch.tensor(0.7)
bias = torch.tensor(0.3)
start = 0
end = 1
step = 0.02

X = torch.arange(start, end, step).unsqueeze(1)
y = X*weight + bias

train_test_split = int(0.8 * len(X))
x_train, y_train = X[:train_test_split], y[:train_test_split]
x_test, y_test = X[train_test_split:], y[train_test_split:]
print(len(x_test), len(x_train))

device = 'cude' if torch.cuda.is_available() else 'cpu'

def plot_prediction(x_train=x_train, 
         y_train = y_train, 
         x_test = x_test, 
         y_test = y_test, 
         prediction = None):
    plt.figure(figsize=(10,7))
    plt.scatter(x_train, y_train, c='b', s=10, label="Training data")
    plt.scatter(x_test, y_test, c='g', s=10, label="Testing data")
    
    if prediction != None:
        plt.scatter(x_test, prediction.cpu(), c='r', s=10, label="Prediction")
    plt.legend(prop={'size':14})
    plt.title("Training vs Testing vs Prediction")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.show()
# plot_prediction()

class LinearRegression(nn.Module):
    def __init__(self):
        super().__init__()
        # self.w = nn.Parameter(torch.randn(1,
        #                 requires_grad=True,
        #                 dtype = torch.float))
        # self.bais = nn.Parameter(torch.randn(1,
        #                 requires_grad=True,
        #                 dtype = torch.float))
        self.linear_layer = nn.Linear(in_features=1,
                                      out_features=1)
    def forward(self, x):
        # return x*self.w + self.bais
        return self.linear_layer(x)
    
model = LinearRegression()
# print(model.a())
loss_fn = nn.L1Loss()
optimizer = torch.optim.SGD(params=model.parameters(), lr=0.02)
epochs = 200
loss = 0
torch.manual_seed(42)

for epoch in range(epochs):
    
    model.train()
    
    y_pred = model(x_train)
    
    loss = loss_fn(y_pred, y_train)
    optimizer.zero_grad()
    loss.backward()
    
    optimizer.step()
    
    model.eval()
    
    with torch.inference_mode():
        test_pred = model(x_test)
        test_loss = loss_fn(test_pred, y_test)
    
    if epoch % 20 ==0:
        print(f"epoch = {epoch}, loss = {loss}")

print(model.state_dict())
with torch.inference_mode():
    
    plt.scatter(y_pred, y_train)
    plt.scatter(test_pred, y_test)
from pathlib import Path

def saver(model, path):
    torch.save(obj=model.state_dict(), 
               f=path)
saver(model, "Nadeem.pt")
print("Good job, ur model has been Saved <3")

import subprocess
try:
    result = subprocess.run(['dir', 'module'], capture_output=True, text=True, shell=True, check=True)
    print('Command output:')
    print(result.stdout)
except subprocess.CalledProcessError as e:
    print(f"An error occurred. The directory 'module' might not exist.")
    print(e.stdout)    
load_model = LinearRegression()
load_model.load_state_dict(torch.load(f="Nadeem.pt"))
print(f"Loaded model : {load_model.state_dict()}")
plt.show()
