from torchvision.datasets import FashionMNIST
from torchvision import transforms
import torch.utils.data as Data
import numpy as np
import matplotlib.pyplot as plt
from model import LeNet
import torch
import torch.nn as nn
import copy
import time
import os
import pandas as pd


def train_val_data_process():
    train_data=FashionMNIST(root='./data',
                        train=True,
                        transform=transforms.Compose([
                            transforms.Resize(size=28),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=[0.2860],std=[0.3530])]),
                        download=True)


    train_data,val_data=Data.random_split(train_data,[round(0.8*len(train_data)),round(0.2*len(train_data))])


    train_dataloader=Data.DataLoader(dataset=train_data,
                                     batch_size=128,
                                     shuffle=True,
                                     num_workers=4)

    val_dataloader=Data.DataLoader(dataset=val_data,
                                         batch_size=128,
                                         shuffle=True,
                                         num_workers=4)

    return train_dataloader,val_dataloader

def train_model_process(model,train_dataloader,val_dataloader,num_epochs):
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")


    optimizer=torch.optim.Adam(model.parameters(),lr=0.001)


    criterion=nn.CrossEntropyLoss()


    model=model.to(device)


    best_model_wts=copy.deepcopy(model.state_dict())

    best_acc=0.0
    train_loss_all=[]
    val_loss_all=[]
    train_acc_all=[]
    val_acc_all=[]

    since=time.time()

    for epoch in range(num_epochs):
        print("Epoch {}/{}".format(epoch+1,num_epochs))
        print("-"*10)



        train_loss=0.0
        train_corrects=0

        val_loss=0.0
        val_corrects=0


        train_num=0
        val_num=0


        model.train()
        for step,(b_x,b_y) in enumerate(train_dataloader):
            b_x=b_x.to(device)
            b_y=b_y.to(device)

            output =model(b_x)

            pre_lab=torch.argmax(output,dim=1)

            loss=criterion(output,b_y)

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            train_loss+=loss.item()*b_x.size(0)

            train_corrects+=torch.sum(pre_lab==b_y.data).item()

            train_num+=b_x.size(0)

        # 验证阶段：eval + no_grad，不再构建计算图、不再占用显存
        model.eval()
        with torch.no_grad():
            for step,(b_x,b_y) in enumerate(val_dataloader):
                b_x=b_x.to(device)
                b_y=b_y.to(device)

                output=model(b_x)

                pre_lab=torch.argmax(output,dim=1)

                loss=criterion(output,b_y)

                val_loss+=loss.item()*b_x.size(0)

                val_corrects+=torch.sum(pre_lab==b_y.data).item()

                val_num+=b_x.size(0)


        train_loss_all.append(train_loss/train_num)
        train_acc_all.append(train_corrects/train_num)
        val_loss_all.append(val_loss/val_num)
        val_acc_all.append(val_corrects/val_num)


        print('{}  Train Loss:{:.4f} Train Acc:{:.4f}'.format(epoch,train_loss_all[-1],train_acc_all[-1]))
        print('{}  Val Loss:{:.4f} Val Acc:{:.4f}'.format(epoch,val_loss_all[-1],val_acc_all[-1]))

        if val_acc_all[-1]>best_acc:
            best_acc=val_acc_all[-1]
            best_model_wts=copy.deepcopy(model.state_dict())


        time_use=time.time()-since
        print("训练耗费时间{:.0f}m{:.0f}s".format(time_use//60,time_use%60))


    # 以下三行必须在 epoch 循环【外面】：否则每个 epoch 都把权重回滚成历史最优，等于破坏了训练
    model.load_state_dict(best_model_wts)
    save_path=os.path.join(os.path.dirname(os.path.abspath(__file__)),'best_model.pth')
    torch.save(model.state_dict(),save_path)
    print("最优验证准确率:{:.4f}，模型已保存至 {}".format(best_acc,save_path))

    train_process=pd.DataFrame(data={"epoch":range(num_epochs),
                                     "train_loss_all":train_loss_all,
                                     "val_loss_all":val_loss_all,
                                     "train_acc_all":train_acc_all,
                                     "val_acc_all":val_acc_all})
    return train_process


def matplot_acc_loss(train_process):
    plt.figure(figsize=(12,4))
    plt.subplot(1,2,1)
    plt.plot(train_process["epoch"],train_process["train_loss_all"],'ro-',label="train loss")
    plt.plot(train_process["epoch"],train_process["val_loss_all"],'bs-',label="val loss")
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel("loss")


    plt.subplot(1,2,2)
    plt.plot(train_process["epoch"],train_process["train_acc_all"],'ro-',label="train acc")
    plt.plot(train_process["epoch"],train_process["val_acc_all"],'bs-',label="val acc")
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel("acc")
    plt.show()



if __name__=="__main__":
    # 原写法 LeNet = LeNet() 会把导入的类名覆盖成实例，改名避免踩坑
    model=LeNet()
    train_dataloader,val_dataloader= train_val_data_process()
    train_process=train_model_process(model,train_dataloader,val_dataloader,20)
    matplot_acc_loss(train_process)
