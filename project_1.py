import torch
import torch.utils.data as data
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_diabetes

diabetes = load_diabetes()

class BloodDataset(data.Dataset):
    def __init__(self):
        self.data = torch.tensor(diabetes.data, dtype=torch.float32)
        self.target = torch.tensor(diabetes.target, dtype=torch.float32).unsqueeze(1)
        self.length = len(self.data)
    
    def __getitem__(self, item):
        return (self.data[item], self.target[item])
    
    def __len__(self):
        return self.length 

class BloodModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(10, 64)
        self.layer2 = nn.Linear(64, 1)
    
    def forward(self, x):
        # лучше всего показывает себя ReLU
        x = torch.relu(self.layer1(x))
        return self.layer2(x)

model = BloodModel()
model.train()

epochs = 10 
batch_size = 8

d_train = BloodDataset()
train_data = data.DataLoader(d_train, batch_size=batch_size, shuffle=True)

# лучше всего показала себя L1Loss
loss_func = nn.L1Loss()
optimizer = optim.RMSprop(params=model.parameters(), lr=0.01)

for _e in range(epochs):
    for x_train, y_train in train_data:
        predict = model(x_train)
        loss = loss_func(predict, y_train)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

model.eval()
predictions = model(d_train.data)
Q = loss_func(predictions, d_train.target).item()
