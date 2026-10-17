import torch
import torch.utils.data as Data
import os
from torchvision import transforms
from torchvision.datasets import FashionMNIST
from model import LeNet


def test_data_process():
    test_data=FashionMNIST(root='./data',
                        train=False,
                        transform=transforms.Compose([
                            transforms.Resize(size=28),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=[0.2860],std=[0.3530])]),
                        download=True)



    test_dataloader=Data.DataLoader(dataset=test_data,
                                     batch_size=128,
                                     shuffle=True,
                                     num_workers=0)

    return test_dataloader
def test_model_process(model,test_dataloader):
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model=model.to(device)

    test_corrects=0.0
    test_num=0

    model.eval()
    # 注意这里是遍历传入的 test_dataloader，不是遍历 test_data_process 这个函数对象
    with torch.no_grad():
        for step,(test_data_x,test_data_y) in enumerate(test_dataloader):
            test_data_x=test_data_x.to(device)
            test_data_y=test_data_y.to(device)

            output=model(test_data_x)
            pre_label=torch.argmax(output,dim=1)

            test_corrects+=torch.sum(pre_label==test_data_y.data).item()

            test_num+=test_data_x.size(0)

    test_acc=test_corrects/test_num
    print("测试准确率:{:.4f}".format(test_acc))
    return test_acc




if __name__=="__main__":
    # 原写法 model = LeNet 拿到的是类本身，不是实例，下一行 load_state_dict 会直接报错
    model=LeNet()

    # 用 __file__ 拼绝对路径，避免换个工作目录就找不到权重文件
    weight_path=os.path.join(os.path.dirname(os.path.abspath(__file__)),'best_model.pth')
    model.load_state_dict(torch.load(weight_path,map_location='cpu'))

    test_dataloader=test_data_process()
    # 原写法调用的是 test_data_process（数据处理函数），这里要调的是 test_model_process
    test_model_process(model,test_dataloader)


