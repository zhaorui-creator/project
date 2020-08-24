

# pretrained_dict =...

# model_dict = model.state_dict()


# # 1. filter out unnecessary keys
# pretrained_dict = {k: v for k, vin pretrained_dict.items() if k inmodel_dict}


# # 2. overwrite entries in the existing state dict
# model_dict.update(pretrained_dict)


# # 3. load the new state dict
# model.load_state_dict(model_dict) 










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
from torchvision import transforms
import os


from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

#export CUDA_VISIBLE_DEVICES=0,1,2
batchsize=32
lr=1e-3
epochs=40
#gpu_list = [0,1,2]
#gpu_list_str = ','.join(map(str, gpu_list))
#os.environ.setdefault("CUDA_VISIBLE_DEVICES", gpu_list_str)
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

device=torch.device('cuda')
#torch.cuda.manual_seed_all(1234)
torch.cuda.manual_seed(1234)




#/home/luo/rui/classification/Dataset/



train_db=MyDataset('/home/luo/rui/classification/Dataset/','train')
val_db=MyDataset('/home/luo/rui/classification/Dataset/','val')
#test_db=('/home/dl/Documents/project/classfication/Dataset/','test')

train_loader=DataLoader(train_db,batch_size=batchsize,shuffle=True,num_workers=4)
val_loader=DataLoader(val_db,batch_size=batchsize,num_workers=4)
#test_loader=DataLoader(test_db,batch_size=batchsize,num_workers=4)
#test_loader=DataLoader(test_db,batch_size=batchsize,num_workers=4)
#print(model)

viz=visdom.Visdom()
def evalute(model,loader):
    correct=0
    total=len(loader.dataset)
    model.eval()
    for step,(x,y) in enumerate(loader):
        print(x.shape)
        x,y=x.to(device),y.to(device)
        with torch.no_grad():
            logits=model(x)
            pred=logits.argmax(dim=1)
    
        
        correct+=torch.eq(pred,y).sum().float().item()
    model.train()
    
    return correct/total

    



def main():
    model=resnet18(pretrained=False)
    flag = 0
    if flag:
        path_state_dict='/home/luo/rui/classification/checkpoints/checkpoint_model_epoch_24_loss0.8995609283447266.pth.tar'
        
        state_dict_load=torch.load(path_state_dict)

        new_state_dict=OrderedDict()
        for k,v in state_dict_load.items():
            namekey=k[7:] if k.startswith('module.') else k
            new_state_dict[namekey]=v

        model.load_state_dict(new_state_dict)
        model.to(device)
    #net=resnet18(pretrained=False)


    pretrained_dict = torch.load('/home/luo/rui/classification/resnet18-5c106cde.pth')
    model_dict = model.state_dict()
    print(model_dict)
    fc_weight = pretrained_dict.pop('fc.weight')
    fc_bias = pretrained_dict.pop('fc.bias')

    pretrained_dict = {k:v for k,v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    



    
    #model=nn.DataParallel(net)
    model.to(device)
    optimizer=optim.Adam(model.parameters(),lr=lr)
    #optimizer=optim.SGD(model.parameters(),lr=lr,momentum=0.9)
    #scheduler=torch.optim.lr_scheduler.StepLR(optimizer,step_size=10,gamma=0.1)
    #scheduler=torch.optim.lr_scheduler.MultiStepLR(optimizer,milestones=[1, 3, 6, 8], gamma=0.1)

    criteon=nn.CrossEntropyLoss()

    best_acc,best_epoch=0,0
    global_step=0
    #global_step2=0
    viz.line([0],[-1],win='train loss',opts=dict(title='train loss'))
    #viz.line([0],[-1],win='test loss',opts=dict(title='test loss'))
    viz.line([0],[-1],win='val_acc',opts=dict(title='val_acc'))

    #print("CUDA_VISIBLE_DEVICES :{}".format(os.environ["CUDA_VISIBLE_DEVICES"]))
    print("device_count :{}".format(torch.cuda.device_count()))
    #i=0
    
    for epoch in range(epochs):
        #i=i+1
        Loss=0
        for step,(x,y) in enumerate(train_loader):
            #x:[b,6,500,500] y:[b]
            optimizer.zero_grad()
            x,y=x.to(device),y.to(device)
            print(y.shape)

            logits=model(x)
            loss=criteon(logits,y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            #print("model outputs.size:{}".format(logits.size()))

            print("model loss:{}".format(loss),"current epoch:{}".format(epoch))
            Loss+=loss
        
            viz.line([loss.item()],[global_step],win='train loss',update='append')
            global_step+=1
            print("model outputs.size:{}".format(logits.size()))
        #scheduler.step()
        file_path=os.path.join('/home/luo/rui/classification/checkpoints/','checkpoint_model_epoch_{}_loss{}.pth.tar'.format(epoch+1,Loss/step))
        torch.save(model.state_dict(),file_path)

        #if (epoch+1)%1==0:
            #for step,(x,y) in enumerate(test_loader):
            #x:[b,6,500,500] y:[b]
                #x,y=x.to(device),y.to(device)
                #with torch.no_grad():
                    #logits=model(x)
                    #loss=criteon(logits,y)
                #viz.line([loss.item()],[global_step],win='test loss',update='append')
            #global_step+=1

    
        if epoch%1==0:
            #model.eval()
            val_acc=evalute(model,val_loader)
            if val_acc>best_acc:
                best_epoch=epoch
                best_acc=val_acc
                #checkpoint={'model_state_dict':model.state_dict(),
                            #'optimizer_state_dict':optimizer.state_dict(),
                            #'epoch':epoch,
                            #'best_epoch':best_epoch,
                            #'best_acc':best_acc

                #}
                #path_checkpoint='./checkpoint.pkl'
                #torch.save(checkpoint,path_checkpoint)

                #torch.save(model.state_dict(),'best8.mdl')
                viz.line([val_acc],[global_step],win='val_acc',update='append')
            
                #print('best acc:',best_acc,'best_epoch:',best_epoch)
        
            
    
    print('best acc:',best_acc,'best_epoch:',best_epoch)

    #path_state_dict='./best1.mdl'
    
    #state_dict_load=torch.load(path_state_dict)

    #new_state_dict=OrderedDict()
    #for k,v in state_dict_load.items():
        #namekey=k[7:] if k.startswith('module.') else k
        #new_state_dict[namekey]=v

    #model.load_state_dict(new_state_dict)
    
    #print('loaded from ckpt:')
    #test_acc=evalute(model,test_loader)
    #print('test_acc:',test_acc)


if __name__=='__main__':
    main()
