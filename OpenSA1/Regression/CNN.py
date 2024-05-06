"""
    Create on 2021-1-21
    Author：Pengyou Fu
    Describe：this for train NIRS with use 1-D Resnet model to transfer
"""
import csv
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import Dataset
import torchvision
import torch.nn.functional as F
from sklearn.preprocessing import scale,MinMaxScaler,Normalizer,StandardScaler
import torch.optim as optim
from Regression.CnnModel import ConvNet, DeepSpectra, AlexNet
import os
from datetime import datetime
from Evaluate.RgsEvaluate import ModelRgsevaluate, ModelRgsevaluatePro
import matplotlib.pyplot  as plt
from torch.utils.tensorboard import SummaryWriter
from matplotlib.pyplot import MultipleLocator
import pandas as pd

LR = 0.001
BATCH_SIZE = 32
TBATCH_SIZE = 120


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#自定义加载数据集
class MyDataset(Dataset):
    def __init__(self,specs,labels):
        self.specs = specs
        self.labels = labels

    def __getitem__(self, index):
        spec,target = self.specs[index],self.labels[index]
        return spec,target

    def __len__(self):
        return len(self.specs)



###定义是否需要标准化
def ZspPocessnew(X_train, X_test, y_train, y_test, need=True): #True:需要标准化，Flase：不需要标准化

    global standscale
    global yscaler

    if (need == True):
        standscale = StandardScaler()
        X_train_Nom = standscale.fit_transform(X_train)
        X_test_Nom = standscale.transform(X_test)

        #yscaler = StandardScaler()
        yscaler = MinMaxScaler()
        y_train = yscaler.fit_transform(y_train.reshape(-1, 1))
        y_test = yscaler.transform(y_test.reshape(-1, 1))

        X_train_Nom = X_train_Nom[:, np.newaxis, :]
        X_test_Nom = X_test_Nom[:, np.newaxis, :]

        ##使用loader加载测试数据
        data_train = MyDataset(X_train_Nom, y_train)
        data_test = MyDataset(X_test_Nom, y_test)
        return data_train, data_test
    elif((need == False)):
        yscaler = StandardScaler()
        # yscaler = MinMaxScaler()

        X_train_new = X_train[:, np.newaxis, :]  #
        X_test_new = X_test[:, np.newaxis, :]

        y_train = yscaler.fit_transform(y_train)
        y_test = yscaler.transform(y_test)

        data_train = MyDataset(X_train_new, y_train)
        ##使用loader加载测试数据
        data_test = MyDataset(X_test_new, y_test)

        return data_train, data_test


def plot_with_metrics(x, y, R2, RMSE, MAE):
    #x = np.array(x)
    #y = np.array(y)
    #x=x.flatten()
    #y=y.flatten()
    x = x.ravel()
    y = y.ravel()
    fig, ax = plt.subplots(figsize=(3, 3), dpi=300)
    ax.plot((0, 1), (0, 1), linewidth=1, transform=ax.transAxes, ls='--', c='k', label="1:1 line", alpha=0.5)
    ax.plot(x, y, 'o', c='blue', markersize=5)

    parameter = np.polyfit(x, y, 1)
    f = np.poly1d(parameter)
    ax.plot(x, f(x), 'r-', lw=1)

    bbox = dict(boxstyle="round", fc='1', alpha=0.)
    plt.text(0.05, 0.80, "$R^2=%.2f$\n$RMSE=%.2f$\n$MAE=%.2f$" % ((R2), RMSE, MAE), transform=ax.transAxes, size=7,
             bbox=bbox, fontname='Times New Roman')

    ax.set_xlabel('Measured values', fontsize=7, fontname='Times New Roman')
    ax.set_ylabel('Predicted values', fontsize=7, fontname='Times New Roman')
    ax.tick_params(labelsize=7)
    ax.set_title('Silt', fontsize=7, fontname='Times New Roman')
    ax.text(0.02, 1.06, '({})'.format(chr(ord('f'))), transform=ax.transAxes, fontsize=9, va='top', fontname='Times New Roman')

    x_major_locator = MultipleLocator(20)
    ax.xaxis.set_major_locator(x_major_locator)
    y_major_locator = MultipleLocator(20)
    ax.yaxis.set_major_locator(y_major_locator)
    ax.set(xlim=(0, 100), ylim=(0, 100))

    plt.savefig("out6.png", bbox_inches='tight', dpi=300)
    plt.show()

def CNNTrain(NetType, X_train, X_test, y_train, y_test, EPOCH):


    data_train, data_test = ZspPocessnew(X_train, X_test, y_train, y_test, need=True)
    # data_train, data_test = ZPocess(X_train, X_test, y_train, y_test)

    train_loader = torch.utils.data.DataLoader(data_train, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = torch.utils.data.DataLoader(data_test, batch_size=TBATCH_SIZE, shuffle=True)

    if NetType == 'ConNet':
        model = ConvNet().to(device)
    elif NetType == 'AlexNet':
        #model = AlexNet().to(device)
        model_path = "model/10_sturacture.pth"

        model=torch.load(model_path)
        #model.load_state_dict(model_path)
        model.to(device)
    elif NetType == 'DeepSpectra':
        model = DeepSpectra().to(device)
    logger = SummaryWriter(log_dir='F:\keziyi\soil\OpenSA-main\OpenSA\log')


    criterion = nn.MSELoss().to(device)  # 损失函数为焦损函数，多用于类别不平衡的多分类问题
    optimizer = optim.Adam(model.parameters(), lr=LR)#,  weight_decay=0.001)  # 优化方式为mini-batch momentum-SGD，并采用L2正则化（权重衰减）
    # # initialize the early_stopping object
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, verbose=1, eps=1e-06,
                                                           patience=20)
    print("Start Training!")  # 定义遍历数据集的次数
    # to track the training loss as the model trains
    # 在代码中添加这行以创建一个csv文件，并将表头写入文件
    # 在代码开头创建 CSV 文件并写入表头
    with open('train_test_logs.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['avg_train_loss', 'train_avgrmse', 'train_avgr2', 'train_avgmae','avg_val_loss', 'val_avgrmse', 'val_avgr2', 'val_avgmae'])
    #y_label = []
    #y_pred = []
    for epoch in range(EPOCH):
        train_losses = []
        model.train()  # 不训练
        train_rmse = []
        train_r2 = []
        train_mae = []

        for i, data in enumerate(train_loader):  # gives batch data, normalize x when iterate train_loader
            inputs, labels = data  # 输入和标签都等于data
            inputs = Variable(inputs).type(torch.FloatTensor).to(device)  # batch x
            #print(inputs.shape)
            labels = Variable(labels).type(torch.FloatTensor).to(device)  # batch y
            output = model(inputs)  # cnn output
            loss = criterion(output, labels)  # MSE
            optimizer.zero_grad()  # clear gradients for this training step
            loss.backward()  # backpropagation, compute gradients
            optimizer.step()  # apply gradients
            pred = output.detach().cpu().numpy()
            y_true = labels.detach().cpu().numpy()
            train_losses.append(loss.item())
            rmse, R2, mae = ModelRgsevaluatePro(pred, y_true, yscaler)
            pred = yscaler.inverse_transform(pred)
            y_true = yscaler.inverse_transform(y_true)
            # plotpred(pred, y_true, yscaler))
            #y_pred.append(pred.flatten())
            #print(len(y_pred))
            #y_label.append(y_true.flatten())
            # 将 pred1 连接到 pred 的后面
            #print(epoch)

            if epoch == 49:
                y_pred_1 = pred
                y_label_1 = y_true
            # 从 epoch == 20 开始，修改 y_pred_1 和 y_label_1 的行为
            if epoch >= 50 and epoch < 55:
                y_pred_1 = np.concatenate((y_pred_1, pred), axis=0)
                y_label_1 = np.concatenate((y_label_1, y_true), axis=0)
            
            # if epoch==0:
            #     # 如果是第一次迭代，将 y_pred_1 初始化为 pred
            #     y_pred_1 = pred
            # else:
            #     # 如果不是第一次迭代，将 pred 追加到 y_pred_1 后面
            #     y_pred_1 = np.concatenate((y_pred_1, pred), axis=0)
            # if epoch==0:
            #     # 如果是第一次迭代，将 y_pred_1 初始化为 pred
            #     y_label_1 = y_true
            # else:
            #     # 如果不是第一次迭代，将 pred 追加到 y_pred_1 后面
            #     y_label_1 = np.concatenate((y_label_1, y_true), axis=0)

            train_rmse.append(rmse)
            train_r2.append(R2)
            train_mae.append(mae)
        avg_train_loss = np.mean(train_losses)
        avgrmse = np.mean(train_rmse)
        avgr2 = np.mean(train_r2)
        avgmae = np.mean(train_mae)
        logger.add_scalar('train_avgrmse', avgrmse, epoch + 1)
        logger.add_scalar('train_avgr2', avgr2, epoch + 1)
        logger.add_scalar('train_avgmae', avgmae,epoch + 1)
        logger.add_scalar('train_loss', avg_train_loss, epoch + 1)
        #print(pred.shape)
        #if avgr2 > 0.9:
        if epoch == 55:
            plot_with_metrics(y_pred_1, y_label_1, avgr2, avgrmse, avgmae)
        #     print(len(y_pred))
        #     print(len(y_label))
        #     # 创建一个 DataFrame
        #     df = pd.DataFrame({'y_pred': y_pred, 'y_label': y_label})
        #     # 指定要写入的文件名
        #     filename = 'predictions.csv'
        #     # 将 DataFrame 写入 CSV 文件
        #     df.to_csv(filename, index=False)
        #     print("CSV 文件写入完成！")
            #plot_with_metrics(y_pred, y_label, avgr2, avgrmse, avgmae)

        print('Epoch:{}, TRAIN:rmse:{}, R2:{}, mae:{}'.format((epoch+1), (avgrmse), (avgr2), (avgmae)))
        print('lr:{}, avg_train_loss:{}'.format((optimizer.param_groups[0]['lr']), avg_train_loss))
        if avgr2 > 0.98:
            torch.save(model.state_dict(), 'model/com_6_model_epoch_{}.pth'.format(epoch + 1))
            torch.save(model, 'model/com_6_sturacture.pth')
            

        with torch.no_grad():  # 无梯度
            model.eval()  # 不训练
            test_rmse = []
            test_r2 = []
            test_mae = []
            val_losses = []
            for i, data in enumerate(test_loader):
                inputs, labels = data  # 输入和标签都等于data
                inputs = Variable(inputs).type(torch.FloatTensor).to(device)  # batch x
                labels = Variable(labels).type(torch.FloatTensor).to(device)  # batch y
                outputs = model(inputs)  # 输出等于进入网络后的输入
                val_loss = criterion(outputs, labels)
                pred = outputs.detach().cpu().numpy()
                y_true = labels.detach().cpu().numpy()
                rmse, R2, mae = ModelRgsevaluatePro(pred, y_true, yscaler)

                val_losses.append(val_loss.item())
                test_rmse.append(rmse)
                test_r2.append(R2)
                test_mae.append(mae)
            avgrmse1 = np.mean(test_rmse)
            avgr21   = np.mean(test_r2)
            avgmae1 = np.mean(test_mae)
            avg_val_loss = np.mean(val_losses)
            logger.add_scalar('test_avgrmse', avgrmse1, epoch + 1)
            logger.add_scalar('test_avgr2', avgr21, epoch + 1)
            logger.add_scalar('test_avgmae', avgmae1, epoch + 1)
            logger.add_scalar('val_loss', avg_val_loss, epoch + 1)  # 记录验证集平均损失值
            print('EPOCH：{}, TEST: rmse:{}, R2:{}, mae:{}'.format((epoch+1), (avgrmse1), (avgr21), (avgmae1)))
            # 将每次测试结果实时写入acc.txt文件中
            #scheduler.step(rmse1)
            # print('Model saved for epoch {}'.format(epoch + 1))

        with open('train_test_logs.csv', mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([avg_train_loss, avgrmse, avgr2, avgmae,avg_val_loss, avgrmse1, avgr21, avgmae1])
    #torch.save(model.state_dict(), 'model/com_model_epoch_{}.pth'.format(epoch + 1))
    #torch.save(model, 'model/com_sturacture.pth')
    return avgrmse, avgr2, avgmae











#
# def CNN(X_train, X_test, y_train, y_test, BATCH_SIZE, n_epochs):
#
#     CNNTrain(X_train, X_test, y_train, y_test,BATCH_SIZE,n_epochs)
