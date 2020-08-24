import torchvision
#import torchvision.models as models
from torchvision.models import resnet18
from torchsummary import summary
import torch
from torch import optim,nn
import visdom
from torch.utils.data import DataLoader
from data.MyDataset import MyDataset
import visdom
import os
from collections import OrderedDict

#export CUDA_VISIBLE_DEVICES=0,1,2
batchsize=32
lr=1e-3
epochs=30
gpu_list = [0,1,2]
gpu_list_str = ','.join(map(str, gpu_list))
os.environ.setdefault("CUDA_VISIBLE_DEVICES", gpu_list_str)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#device=torch.device('cuda')
torch.cuda.manual_seed_all(1234)



train_db=MyDataset('/home/dl/Documents/project/classfication/Dataset/','train')
val_db=MyDataset('/home/dl/Documents/project/classfication/Dataset/','val')
test_db=('/home/dl/Documents/project/classfication/Dataset/','test')

train_loader=DataLoader(train_db,batch_size=batchsize,shuffle=True,num_workers=4)
val_loader=DataLoader(val_db,batch_size=batchsize,num_workers=4)
#test_loader=DataLoader(test_db,batch_size=batchsize,num_workers=4)
#print(model)

viz=visdom.Visdom()
def evalute(model,loader):
    correct=0
    total=len(loader.dataset)
    for x,y in loader:
        x,y=x.to(device),y.to(device)
        with torch.no_grad():
            logits=model(x)
            pred=logits.argmax(dim=1)
        
        correct+=torch.eq(pred,y).sum().float().item()
    
    return correct/total


def main():
    net=resnet18(pretrained=False)
    model=nn.DataParallel(net)
    model.to(device)
    optimizer=optim.Adam(model.parameters(),lr=lr)
    #optimizer=optim.SGD(model.parameters(),lr=lr,momentum=0.9)
    #scheduler=torch.optim.lr_scheduler.StepLR(optimizer,step_size=10,gamma=0.1)

    criteon=nn.CrossEntropyLoss()

    best_acc,best_epoch=0,0
    global_step=0
    viz.line([0],[-1],win='loss',opts=dict(title='loss'))
    viz.line([0],[-1],win='val_acc',opts=dict(title='val_acc'))

    print("CUDA_VISIBLE_DEVICES :{}".format(os.environ["CUDA_VISIBLE_DEVICES"]))
    print("device_count :{}".format(torch.cuda.device_count()))
    #i=0