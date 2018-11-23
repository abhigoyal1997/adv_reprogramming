# coding: utf-8
import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import datasets
from torchvision import transforms
from torchvision.models.inception import inception_v3
from torchvision.models.resnet import resnet50
from torch.utils.data import random_split

from tensorboardX import SummaryWriter
from argparse import ArgumentParser
from tqdm import tqdm
from time import time

from utils import save_model
from model import AdvProgram


parser = ArgumentParser()
parser.add_argument("-m","--model-name", default=None,
                    help="Model Name")
parser.add_argument("-l","--log-interval", type=int,default=10,
                    help="Log Interval")
parser.add_argument("--dataset", default="mnist",
                    help="Dataset to be used")
parser.add_argument("--model-type", default="resnet50",
                    help="Model type to be used (resenet50 | inception_v3 | resnet101 | resnet152)")
parser.add_argument("--lr", type=float, default=0.05,
                    help="Learning Rate to be used")
parser.add_argument("--wd", type=float, default=0.00,
                    help="weight decay values")
parser.add_argument("--lr-decay", type=float, default=0.96,
                    help="decay rate of learning rate")
parser.add_argument("--epochs", type=int, default=100,
                    help="number of epochs to train the model")
parser.add_argument("--decay-step", type=int, default=2,
                    help="number of steps for decay")


args = parser.parse_args()
model_name = args.model_name
log_interval = args.log_interval
if args.model_type == "inception_v3":
    pimg_size = (299,299)
else:
    pimg_size = (224,224)

if args.dataset == "mnist":
    img_size = (28,28)
else:
    img_size = (32,32)

mask_size = pimg_size

num_channels = 3

batch_size = 100
test_batch_size = 100
data_dir = 'data/'
models_dir = 'models/'
logs_dir = 'logs/'
train_ratio = 0.8

writer = SummaryWriter("{}{}-{}".format(logs_dir, model_name, time()))

l_pad = int((pimg_size[0]-img_size[0]+1)/2)
r_pad = int((pimg_size[0]-img_size[0])/2)

if args.dataset == "mnist":
    transform = transforms.Compose([
        transforms.Pad(padding=(l_pad, l_pad, r_pad, r_pad)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
        transforms.Lambda(lambda x: torch.cat([x]*3)),
    ])
else:
    transform = transforms.Compose([
        transforms.Pad(padding=(l_pad, l_pad, r_pad, r_pad)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

if args.dataset == "mnist":
    dataset = datasets.MNIST(data_dir, download=True, train=True, transform=transform)
else:
    dataset = datasets.CIFAR10(data_dir, download=True, train=True, transform=transform)

train_dataset, valid_dataset = random_split(dataset, [int(train_ratio*len(dataset)), len(dataset) - int(train_ratio*len(dataset))])

train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=batch_size, shuffle=True
)

valid_loader = torch.utils.data.DataLoader(
    valid_dataset,
    batch_size=batch_size, shuffle=True
)

if args.dataset == "mnist":
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(data_dir, train=False, transform=transform),
        batch_size=test_batch_size, shuffle=False
    )
else:
    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(data_dir, train=False, transform=transform),
        batch_size=test_batch_size, shuffle=False
    )

device = torch.device('cuda')
model = eval(args.model_type)(pretrained=True).to(device)
model.eval()
for param in model.parameters():
    param.requires_grad = False

# program = torch.randn(num_channels, *pimg_size, device=device)
# program.requires_grad = True
#
# l_pad = int((mask_size[0]-img_size[0]+1)/2)
# r_pad = int((mask_size[0]-img_size[0])/2)
#
# mask = torch.zeros(num_channels, *img_size, device=device)
# mask = F.pad(mask, (l_pad, r_pad, l_pad, r_pad), value=1)
#
# batch_norm = nn.BatchNorm2d(3)

adv_program = AdvProgram(img_size, pimg_size, mask_size, device=device)

optimizer = optim.Adam(adv_program.parameters(), lr=args.lr, weight_decay=args.wd)
lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.decay_step, gamma=args.lr_decay)
# lr_scheduler = LRScheduler(optimizer, patience=args.decay_step, factor=args.lr_decay)

loss_criterion = nn.CrossEntropyLoss()


def run_epoch(mode, data_loader, num_classes=10, optimizer=None, epoch=None, steps_per_epoch=None, loss_criterion=None):
    if mode == 'train':
        # program.requires_grad = True
        adv_program.train()
    else:
        # program.requires_grad = False
        adv_program.eval()

    loss = 0.0
    y_true = None
    y_pred = None

    if steps_per_epoch is None:
        steps_per_epoch = len(data_loader)

    if epoch is not None:
        ite = tqdm(
            enumerate(data_loader, 0),
            total=steps_per_epoch,
            desc='Epoch {}: '.format(epoch)
        )
    else:
        ite = tqdm(enumerate(data_loader, 0))

    for i, data in ite:
        x = data[0].to(device)
        y = data[1].to(device)

        if mode == 'train':
            optimizer.zero_grad()

        if mode != 'train':
            with torch.no_grad():
                # x = x + F.tanh(program*mask)
                x = adv_program(x)
                logits = model(x)
        else:
            # x = x + torch.tanh(program*mask)
            x = adv_program(x)
            logits = model(x)

        logits = logits[:,:num_classes]

        if loss_criterion is not None:
            batch_loss = loss_criterion(logits, y)

            if mode == 'train':
                batch_loss.backward()
                optimizer.step()

            loss += batch_loss.item()

        if y_true is None:
            y_true = y
        else:
            y_true = torch.cat([y_true, y], dim=0)

        if y_pred is None:
            y_pred = torch.argmax(torch.softmax(logits, dim=1), dim=1)
        else:
            y_pred = torch.cat([y_pred, torch.argmax(torch.softmax(logits, dim=1), dim=1)], dim=0)

        if i % log_interval == 0 and mode == 'train':
            writer.add_scalar("{}_loss".format(mode), loss/(i+1), epoch*steps_per_epoch + i)
            print("Loss at Step {} : {}".format(epoch*steps_per_epoch + i, loss/(i+1)))

        if i >= steps_per_epoch:
            break

    accuracy = torch.sum(y_true==y_pred).item()/(y_true.shape[0])
    if mode != 'train':
        writer.add_scalar("{}_loss".format(mode), loss/steps_per_epoch, epoch*steps_per_epoch)
    writer.add_scalar("{}_accuracy".format(mode), accuracy, epoch*steps_per_epoch)
    return {'loss': loss/steps_per_epoch, 'accuracy': accuracy}


num_epochs = args.epochs
best_accuracy = 0
checkpoint_path = models_dir+model_name+'_checkpoint'
epoch = 0

while epoch < num_epochs:
    train_metrics = run_epoch('train', train_loader, 10, optimizer, epoch=epoch, loss_criterion=loss_criterion)
    valid_metrics = run_epoch('valid', valid_loader, 10, epoch=epoch, loss_criterion=loss_criterion)
    test_metrics = run_epoch('test', test_loader, 10, epoch=epoch, loss_criterion=loss_criterion)

    print('Train Metrics : {}, Validation Metrics : {}, Test Metrics : {}'.format(str(train_metrics), str(valid_metrics), str(test_metrics)))
    lr_scheduler.step()
    if valid_metrics['accuracy'] > best_accuracy:
        best_accuracy = valid_metrics['accuracy']
        # save_checkpoint(epoch, program, optimizer, best_accuracy, lr_scheduler, file_path=checkpoint_path)
        # save_model(program, mask)
        save_model(adv_program)

    epoch += 1
    # if lr_scheduler.is_impatient(valid_metrics['accuracy'], epoch):
    #     program, epoch, best_accuracy = load_checkpoint(optimizer=optimizer, lr_scheduler=lr_scheduler, file_path=checkpoint_path)
    #     if not lr_scheduler.reduce_lr(epoch):
    #         print("Stopping early: can't reduce lr further")
    #         break
