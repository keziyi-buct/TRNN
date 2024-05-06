import torch
import torch.nn as nn
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import torch.nn.functional as F
from collections.abc import Iterable

# ac_dict = {
#     'relu':nn.ReLU,
#     'lrelu':nn.LeakyReLU
# # nn.LeakyReLU
# }

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet,self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=21, padding=0),
            nn.BatchNorm1d(16),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(16, 32, kernel_size=19, padding=0),
            nn.BatchNorm1d(32),
            nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=17, padding=0),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )
        self.fc = nn. Linear(38080,1) #8960 ,17920
        self.drop = nn.Dropout(0.2)

    def forward(self,out):
      out = self.conv1(out)
      out = self.conv2(out)
      out = self.conv3(out)
      out = out.view(out.size(0),-1)
      # print(out.size(1))
      out = self.fc(out)
      return out

# linear1=[649,5184,5184,5184,5120,5120,5120,5120,4096,4096]
# #3,4,5....
# linear=[33568,33536,33536,33536,33280,32768,32768,32768]
# class AlexNet(nn.Module):
#     def __init__(self,acti='relu',c_num=6):
#         super(AlexNet, self).__init__()
#         self.layers=nn.ModuleList([])
#         input_channel=1
#         output_channel=16
#         for i in range(1,c_num):
#             self.layers.append(nn.Conv1d(input_channel,output_channel,3,padding=1))
#             self.layers.append(nn.BatchNorm1d(num_features=output_channel))
#             self.layers.append(nn.ReLU(inplace=True))
#             self.layers.append(nn.MaxPool1d(2,2))
#             input_channel=output_channel
#             output_channel=output_channel*2
#         #linear[c_num-1]
#         self.reg = nn.Sequential(
#              nn.Linear(linear[c_num-3], 15000),  #根据自己数据集修改
#              nn.ReLU(inplace=True),
#              nn.Linear(15000,500),
#              nn.ReLU(inplace=True),
#              nn.Dropout(0.5),
#              nn.Linear(500, 1),
#         )
#
#     def forward(self,x):
#         out = x
#         for layer in self.layers:
#             out = layer(out)
#         out = out.flatten(start_dim=1)
#         out = out.view(out.size(0), -1)
#         out = self.reg(out)
#         return out

####
# class FireModule(torch.nn.Module):
#     def __init__(self, in_channels, squeeze_channels, expand1x1_channels, expand1x3_channels):
#         super(FireModule, self).__init__()
#         self.squeeze = torch.nn.Conv1d(in_channels, squeeze_channels, kernel_size=1)
#         self.relu = torch.nn.ReLU(inplace=True)
#         self.expand1x1 = torch.nn.Conv1d(squeeze_channels, expand1x1_channels, kernel_size=1)
#         self.expand1x3 = torch.nn.Conv1d(squeeze_channels, expand1x3_channels, kernel_size=3, padding=1)
#
#     def forward(self, x):
#         x = self.squeeze(x)
#         x = self.relu(x)
#         out1x1 = self.expand1x1(x)
#         out1x3 = self.expand1x3(x)
#         out = torch.cat([out1x1, out1x3], dim=1)
#         return self.relu(out)
#
# class AlexNet(nn.Module):
#     def __init__(self, num_classes=1, reduction=16):
#         super(AlexNet, self).__init__()
#         self.features = torch.nn.Sequential(
#             # conv1
#             torch.nn.Conv1d(1, 24, kernel_size=7, stride=2),
#             torch.nn.ReLU(inplace=True),
#             # maxpool1
#             torch.nn.MaxPool1d(kernel_size=3, stride=2),
#             # Fire2
#             FireModule(24, 6, 12, 12),
#             # Fire3
#             FireModule(24, 6, 12, 12),
#             # Fire4
#             FireModule(24, 12, 24, 24),
#             # maxpool4
#             torch.nn.MaxPool1d(kernel_size=3, stride=2),
#
#
#             torch.nn.AdaptiveAvgPool1d(10)
#         )
#         self.reg = nn.Sequential(
#             nn.Linear(480, 200),  #根据自己数据集修改
#             nn.ReLU(inplace=True),
#             # nn.LeakyReLU(inplace=True),
#             nn.Linear(200, 100),
#             nn.ReLU(inplace=True),
#             # nn.LeakyReLU(inplace=True),
#             nn.Dropout(0.5),
#             nn.Linear(100, num_classes),
#         )
#
#     def forward(self, x):
#         out = self.features(x)
#         out = out.flatten(start_dim=1)
#         out = self.reg(out)
#         return out
# class Bottlrneck(torch.nn.Module):
#     def __init__(self,In_channel,Med_channel,Out_channel,downsample):
#         super(Bottlrneck, self).__init__()
#         self.stride = 1
#         if downsample == True:
#             self.stride = 2
#
#         self.layer = torch.nn.Sequential(
#             torch.nn.Conv1d(In_channel, Med_channel, 1, self.stride),
#             torch.nn.BatchNorm1d(Med_channel),
#             torch.nn.ReLU(),
#             torch.nn.Conv1d(Med_channel, Med_channel, 3, padding=1),
#             torch.nn.BatchNorm1d(Med_channel),
#             torch.nn.ReLU(),
#             torch.nn.Conv1d(Med_channel, Out_channel, 1),
#             torch.nn.BatchNorm1d(Out_channel),
#             torch.nn.ReLU(),
#         )
#
#         if In_channel != Out_channel:
#             self.res_layer = torch.nn.Conv1d(In_channel, Out_channel,1,self.stride)
#         else:
#             self.res_layer = None
#
#     def forward(self,x):
#         if self.res_layer is not None:
#             residual = self.res_layer(x)
#         else:
#             residual = x
#         return self.layer(x)+residual
#
# class AlexNet(nn.Module):
#     def __init__(self, in_channels=1,classes=1):
#         super(AlexNet, self).__init__()
#         self.features = torch.nn.Sequential(
#             torch.nn.Conv1d(in_channels,6,kernel_size=7,stride=2,padding=3),
#             torch.nn.MaxPool1d(3,2,1),
#
#             Bottlrneck(6,6,12,False),
#             Bottlrneck(12,6,12,False),
#             Bottlrneck(12,6,12,False),
#             #
#             Bottlrneck(12,6,24, True),
#             Bottlrneck(24,6,24, False),
#             Bottlrneck(24,6,24, False),
#             #
#             Bottlrneck(24,12,48, True),
#             Bottlrneck(48,12,48, False),
#             Bottlrneck(48,12,48, False),
#             #
#             Bottlrneck(48,24,96, True),
#             Bottlrneck(96,24,96, False),
#             Bottlrneck(96,24,96, False),
#
#             #torch.nn.AdaptiveAvgPool1d(20)
#         )
#         self.reg = nn.Sequential(
#             nn.Linear(12672 , 5000),  #根据自己数据集修改
#             nn.ReLU(inplace=True),
#             # nn.LeakyReLU(inplace=True),
#             nn.Linear(5000, 500),
#             nn.ReLU(inplace=True),
#             # nn.LeakyReLU(inplace=True),
#             nn.Dropout(0.5),
#             nn.Linear(500,1),
#         )
#
#     def forward(self, x):
#         out = self.features(x)
#         out = out.flatten(start_dim=1)
#         out = self.reg(out)
#         return out

class Mlp(nn.Module):
    def __init__(self, in_features, pred, hidden_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        #out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.q = nn.Linear(in_features, in_features)
        self.k = nn.Linear(in_features, in_features)
        self.v = nn.Linear(in_features, in_features)
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.LeakyReLU(inplace=True)
        #nn.ReLU(inplace=True)nn.LeakyReLU(inplace=True)
        self.pred = pred
        if pred==True:
            self.fc2 = nn.Linear(hidden_features,1)
        else:
            self.fc2 = nn.Linear(hidden_features, in_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x0 = x
        q = self.q(x).unsqueeze(2)
        k = self.k(x).unsqueeze(2)
        v = self.v(x).unsqueeze(2)
        attn = (q @ k.transpose(-2, -1))
        #print(attn.size())
        attn = attn.softmax(dim=-1)
        x = (attn @ v).squeeze(2)
        #print(x.size())
        x += x0
        x1 = x
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        if self.pred==False:
            x += x1

        x = x.squeeze(0)

        return x



class AlexNet(nn.Module):
    def __init__(self, num_classes=1, reduction=16):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            # conv1
            nn.Conv1d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(num_features=16),
            #nn.ReLU(inplace=True),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),
            # conv2
            nn.Conv1d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(num_features=32),
            #nn.ReLU(inplace=True),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),
            # conv3
            nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(num_features=64),
            #nn.ReLU(inplace=True),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),
            # conv4
            nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(num_features=128),
            #nn.ReLU(inplace=True),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),
            # conv5
            nn.Conv1d(128, 192, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(num_features=192),
            #nn.ReLU(inplace=True),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(192, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(num_features=256),
            #nn.ReLU(inplace=True),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),
            torch.nn.AdaptiveAvgPool1d(1)
            # SELayer(256, reduction),nn.GELU
            # nn.LeakyReLU(inplace=True),
        )
        # self.reg = nn.Sequential(
        #     nn.Linear(25152, 1000),  #根据自己数据集修改
        #     nn.ReLU(inplace=True),
        #     # nn.LeakyReLU(inplace=True),
        #     nn.Linear(1000, 500),
        #     nn.ReLU(inplace=True),
        #     # nn.LeakyReLU(inplace=True),
        #     # nn.Dropout(0.5),,
        #     # nn.Linear(500, num_classes),
        # )

        self.Block1 = Mlp(in_features=256, hidden_features=64, drop=0., pred=False)
        self.Block2 = Mlp(in_features=256, hidden_features=64, drop=0., pred=True)




    def forward(self, x):
        out= self.features(x)
        out = out.view(-1, 256)
        # out = out.flatten(start_dim=1)
        # out = self.reg(out)
        return self.Block2(self.Block1(out))

        #return out

class Inception(nn.Module):
    def __init__(self,in_c,c1,c2,c3,out_C):
        super(Inception,self).__init__()
        self.p1 = nn.Sequential(
            nn.Conv1d(in_c, c1,kernel_size=1,padding=0),
            nn.Conv1d(c1, c1, kernel_size=3, padding=1)
        )
        self.p2 = nn.Sequential(
            nn.Conv1d(in_c, c2,kernel_size=1,padding=0),
            nn.Conv1d(c2, c2, kernel_size=5, padding=2)

        )
        self.p3 = nn.Sequential(
            nn.MaxPool1d(kernel_size=3,stride=1,padding=1),
            nn.Conv1d(in_c, c3,kernel_size=3,padding=1),
        )
        self.conv_linear = nn.Conv1d((c1+c2+c3), out_C, 1, 1, 0, bias=True)
        self.short_cut = nn.Sequential()
        if in_c != out_C:
            self.short_cut = nn.Sequential(
                nn.Conv1d(in_c, out_C, 1, 1, 0, bias=False),

            )
    def forward(self, x):
        p1 = self.p1(x)
        p2 = self.p2(x)
        p3 = self.p3(x)
        out =  torch.cat((p1,p2,p3),dim=1)
        out += self.short_cut(x)
        return out




class DeepSpectra(nn.Module):
    def __init__(self):
        super(DeepSpectra, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=5, stride=3, padding=0)
        )
        self.Inception = Inception(16, 32, 32, 32, 96)
        self.fc = nn.Sequential(
            nn.Linear(20640, 5000),
            nn.Dropout(0.5),
            nn.Linear(5000, 1)
        )
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.Inception(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

