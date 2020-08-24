import torchvision
#import torchvision.models as models
from torchvision.models import resnet50
from torchsummary import summary
import torch
from torch import optim,nn
import visdom
from torch.utils.data import DataLoader
from data.MyDataset import MyDataset
import visdom
import os

#export CUDA_VISIBLE_DEVICES=0,1,2
batchsize=8
lr=1e-3
epochs=50
#gpu_list=[0,1]
#gpu_list_str=','.join(map(str,gpu_list))
#os.environ.setdefault("CUDA_VISIBLE_DEVICES",gpu_list_str)
device=torch.device("cuda")

#device=torch.device('cuda')
torch.manual_seed(1234)



train_db=MyDataset('/home/dl/Documents/project/classfication/Dataset/','train')
val_db=MyDataset('/home/dl/Documents/project/classfication/Dataset/','val')

train_loader=DataLoader(train_db,batch_size=batchsize,shuffle=True,num_workers=4)
val_loader=DataLoader(val_db,batch_size=batchsize,num_workers=4)

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
    model=resnet50(pretrained=False)
    #model=nn.DataParallel(net)
    model.to(device)
    optimizer=optim.Adam(model.parameters(),lr=lr)
    criteon=nn.CrossEntropyLoss()

    best_acc,best_epoch=0,0
    global_step=0
    viz.line([0],[-1],win='loss',opts=dict(title='loss'))
    viz.line([0],[-1],win='val_acc',opts=dict(title='val_acc'))

    #print("CUDA.VISIBLE_DEVICES:{}".format(os.environ["CUDA VISIBLE_DEVICES"]))
    #print("device_count:{}".format(torch.cuda.device_count()))
    for epoch in range(epochs):
        for step,(x,y) in enumerate(train_loader):
            #x:[b,6,500,500] y:[b]
            x,y=x.to(device),y.to(device)

            logits=model(x)
            loss=criteon(logits,y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            #print("model outputs.size:{}".format(logits.size()))
            
        
            viz.line([loss.item()],[global_step],win='loss',update='append')
            global_step+=1
            #print("model outputs.size:{}".format(logits.size()))
    
        if epoch%2==0:
            val_acc=evalute(model,val_loader)
            if val_acc>best_acc:
                best_epoch=epoch
                best_acc=val_acc

                torch.save(model.state_dict(),'best.mdl')
                viz.line([val_acc],[global_step],win='val_acc',update='append')
                #print('best acc:',best_acc,'best_epoch:',best_epoch)
            
    
    print('best acc:',best_acc,'best_epoch:',best_epoch)

    model.load_state_dict(torch.load('best.mdl'))

    print('loaded from ckpt:')


if __name__=='__main__':
    main()
