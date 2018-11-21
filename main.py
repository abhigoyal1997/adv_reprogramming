
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
import torch.optim as optim

from torch.autograd import Variable
from torch.nn import functional as F
from torchvision import datasets
from torchvision import transforms
from torchvision.models.inception import *
from torch.utils.data import random_split

from tqdm import tqdm_notebook


# In[2]:


pimg_size = (299,299)
img_size = (28,28)
mask_size = pimg_size

num_channels = 3


# In[3]:


batch_size = 16
test_batch_size = 16
data_dir = 'data/'
models_dir = 'models/'
train_ratio = 0.8

l_pad = int((pimg_size[0]-img_size[0]+1)/2)
r_pad = int((pimg_size[0]-img_size[0])/2)

transform = transforms.Compose([
    transforms.Pad(padding=(l_pad, l_pad, r_pad, r_pad)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
    transforms.Lambda(lambda x: torch.cat([x]*3))
])

dataset = datasets.MNIST(data_dir, train=True, transform=transform, download=True)
train_dataset, valid_dataset = random_split(dataset, [int(train_ratio*len(dataset)), len(dataset) - int(train_ratio*len(dataset))])

train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=batch_size, shuffle=True
)

valid_loader = torch.utils.data.DataLoader(
    valid_dataset,
    batch_size=batch_size, shuffle=True
)

test_loader = torch.utils.data.DataLoader(
    datasets.MNIST(data_dir, train=False, transform=transform),
    batch_size=test_batch_size, shuffle=False
)


# In[4]:


model = inception_v3(pretrained=True)


# In[5]:


device = torch.device('cuda:0')

program = torch.rand(num_channels, *pimg_size, requires_grad=True, device=device)

l_pad = int((mask_size[0]-img_size[0]+1)/2)
r_pad = int((mask_size[0]-img_size[0])/2)

mask = torch.zeros(num_channels, *img_size)
mask = F.pad(mask, (l_pad, r_pad, l_pad, r_pad), value=1).to(device)

optimizer = optim.Adam([program], lr=0.05, weight_decay=0.01)
lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.96)

loss_criterion = nn.CrossEntropyLoss()


# In[11]:


model = model.to(device)
model.eval()


# In[ ]:


num_epochs = 1

best_error_rate = 1

for epoch in range(num_epochs):
    train_loss = run_epoch('train', train_loader, 10, optimizer, epoch, loss_criterion=loss_criterion)
    valid_loss, val_metrics = run_epoch('valid', valid_loader, 10, epoch, loss_criterion=loss_criterion)
    error_rate = val_metrics['error_rate']
    if error_rate < best_error_rate:
        torch.save({'program':program, 'mask':mask}, models_dir+'model_test')
        best_error_rate = error_rate
        
    _, test_metrics = run_epoch('test', test_loader, 10, epoch)
    
    print('Train loss : {}, Validation Loss : {}, Validation_ER : {}, Test Metrics : {}'.format(train_loss, valid_loss, error_rate, str(test_metrics)))


# In[13]:


def run_epoch(mode, data_loader, num_classes=10, optimizer=None, epoch=None, steps_per_epoch=None, loss_criterion=None):
    if mode == 'train':
        program.requires_grad = True
    else:
        program.requires_grad = False

    loss = 0.0
    if mode != 'train':
        y_true = None
        y_pred = None

    if steps_per_epoch is None:
        steps_per_epoch = len(data_loader)

    if epoch is not None:
        ite = tqdm_notebook(
            enumerate(data_loader, 0),
            total=steps_per_epoch,
            desc='Epoch {}: '.format(epoch)
        )
    else:
        ite = tqdm_notebook(enumerate(data_loader, 0))

    for i, data in ite:
        x = data[0].to(device)
        y = data[1].to(device)

        if mode == 'train':
            optimizer.zero_grad()

        x = x + F.tanh(program*mask)
        logits = model(x)
#         import ipdb; ipdb.set_trace()
        logits = logits[:,:num_classes]

        if loss_criterion is not None:
            batch_loss = loss_criterion(logits, y)

            if mode == 'train':
                batch_loss.backward()
                optimizer.step()

            loss += batch_loss.item()

        if mode != 'train':
            if y_true is None:
                y_true = y
            else:
                y_true = torch.cat([y_true, y], dim=0)

            if y_pred is None:
                y_pred = torch.argmax(torch.softmax(logits, dim=1), dim=1)
            else:
                y_pred = torch.cat([y_pred, torch.argmax(torch.softmax(logits, dim=1), dim=1)], dim=0)

        if i >= steps_per_epoch:
            break

    if mode != 'train':
        error_rate = torch.sum(y_true!=y_pred).item()/(y_true.shape[0])
        return loss/steps_per_epoch, {'error_rate': error_rate}

    return loss/steps_per_epoch

