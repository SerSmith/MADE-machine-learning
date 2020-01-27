# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import pandas as pd
import numpy as np
import sklearn 
from sklearn import datasets
import torch
import seaborn as sns


# %%
data,target=datasets.load_svmlight_file("data/train.txt")

# %% [markdown]
# ## Предобработка исходных данных

# %%
data_torch=torch.from_numpy(data.todense())
target_torch=torch.from_numpy(target)


# %%
msk = np.random.rand(len(target_torch)) < 0.8
data_train=data_torch[msk]
target_train=target_torch[msk]
data_validate=data_torch[~msk]
target_validate=target_torch[~msk]

# %% [markdown]
# ## Определение модели

# %%
# Higher-level API:
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self, hidden_size=20):
        super(Net, self).__init__()

        print(data_train.shape[1])        
        self.layers = nn.Sequential(
#             nn.Dropout(0.90) ,

            nn.Linear(data_train.shape[1], hidden_size),
#             nn.ReLU(),
#             nn.Linear(hidden_size, hidden_size),
            nn.Sigmoid(),
            nn.Linear( hidden_size, hidden_size),
            nn.Sigmoid(),
            nn.Linear( hidden_size, 1),
#            nn.Softmax()
        )
        
    def forward(self, x):
        return self.layers(x)


# %%
from IPython.display import clear_output
from tqdm import trange

# функция для итераций по минибатчам, из первого семинара
def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.random.permutation(len(inputs))
    for start_idx in trange(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]
import seaborn as sns
sns.set(color_codes=True)

def plot_history(train_history, val_history, rang,  i, quantity_epoch,title='loss'):
    if i%quantity_epoch==0:
        clear_output(True)
        plt.figure(figsize=(20,10))
        plt.plot(train_history)
        plt.plot(np.arange(1, len(val_history) + 1) * (len(train_history)/len(val_history)),val_history, 'y+',markersize=15, markeredgewidth=2)
        plt.xlabel("train steps")
        plt.title("Train Loss: {0} \nVal Loss:  {1}".format(np.round(train_history[-1],4), np.round(val_history[-1],4)))
    #     plt.title(title + "\nloss at %i epoch" %(rang+1)[-1])
    #     plt.ylim([0,10**4])
        plt.show()

# %% [markdown]
#  ## Обучение

# %%

def train(X_train, y_train, model, optimizer, batchsize=32, ModelType="first"):
    loss_log = []
    model.train()

    for x_batch, y_batch in iterate_minibatches(X_train, y_train, batchsize=batchsize, shuffle=True):
        
        data = torch.autograd.Variable(x_batch)
        target = torch.autograd.Variable(y_batch)

        optimizer.zero_grad()
        output = model(data)
        

        loss = torch.sqrt((( output -  target) ** 2 ).mean())
            
#         ====================================================================================
#         ====================================================================================
        
        loss.backward()
        
        optimizer.step()
        loss = loss.item()
        loss_log.append(loss)
    return loss_log


def test(model, X_val, y_val, ModelType="first"):
    loss_log = []
    model.eval()  
    tt = torch.autograd.Variable(X_val)
    target = torch.autograd.Variable(y_val)
    output = model.forward(tt)
 
    loss = torch.sqrt((( output -  target) ** 2 ).mean())
#         ====================================================================================
#         ====================================================================================
    
    
    loss_log.append(loss.item())

    return loss_log


# %%
np.random.seed(123)
train_log = []
val_log = []

model = Net()
# opt = torch.optim.SGD(model.parameters(), lr=0.0001)
opt = torch.optim.Adam(model.parameters(), lr=0.02)
# opt = torch.optim.RMSprop(model.parameters(), lr=0.01)
batchsize = 128

rang = np.arange(400)
for epoch in rang:
    train_loss = train(X_train=data_train, y_train=target_train, model=model, optimizer=opt, batchsize=batchsize)
    train_log.extend(train_loss)
#     train_log.extend([np.array(train_loss).mean()])
    
    val_loss = np.mean(test(model=model, X_val=data_validate, y_val=target_validate))
    val_log.append(val_loss)
    # TODO: график train_loss vs train_steps с точками val_loss vs trained_steps
    # use your plot_history()
    plot_history(train_log, val_log, 1,epoch, 20)

    # hint: train_log and val_log may contain data with different shapes


# %%


