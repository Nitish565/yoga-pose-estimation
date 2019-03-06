import torch
import torchvision
from torchvision.models import vgg19
import torch.nn as nn

class F(nn.Module):
    def __init__(self):
        super(F, self).__init__()
        vgg = vgg19(pretrained=True).features[:23]
        for c in vgg.parameters():
            c.requires_grad = False
        
        self.vgg = vgg
        self.conv_4_3_and_4_4 = nn.Sequential(
                                              nn.Conv2d(512, 256, 3, 1, 1),
                                              nn.ReLU(inplace=True),
                                              nn.Conv2d(256, 128, 3, 1, 1),
                                              nn.ReLU(inplace=True)
                                              )
            
    def forward(self, x):
        return self.conv_4_3_and_4_4(self.vgg(x))
#F(torch.ones(1,3,224,224)).shape => torch.Size([1, 128, 28, 28])

class InitialBlockType(nn.Module):
    def __init__(self, out_channels):
        super(InitialBlockType, self).__init__()
        self.block = nn.Sequential(
                                   nn.Conv2d(128, 128, 3, 1, 1),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(128, 128, 3, 1, 1),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(128, 128, 3, 1, 1),
                                   nn.ReLU(inplace=True),
                                   
                                   nn.Conv2d(128, 512, 1, 1, 0),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(512, out_channels, 1, 1, 0)
                                   )
    
    def forward(self, x):
        return self.block(x)

class RefinementBlockType(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(RefinementBlockType, self).__init__()
        self.block = nn.Sequential(
                                   nn.Conv2d(in_channels, 128, 7, 1, 3),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(128, 128, 7, 1, 3),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(128, 128, 7, 1, 3),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(128, 128, 7, 1, 3),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(128, 128, 7, 1, 3),
                                   nn.ReLU(inplace=True),
                                   
                                   nn.Conv2d(128, 512, 1, 1, 0),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(512, out_channels, 1, 1, 0)
                                   )
    
    def forward(self, x):
        return self.block(x)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        n_heatmaps = 17
        n_pafs = 38
        F_out = 128
        stage_2_in = F_out+n_heatmaps+n_pafs
        self.F = F()
        self.Stage_1_1 = InitialBlockType(n_heatmaps)
        self.Stage_1_2 = InitialBlockType(n_pafs)
        self.Stage_2_1 = RefinementBlockType(stage_2_in, n_heatmaps)
        self.Stage_2_2 = RefinementBlockType(stage_2_in, n_pafs)
        self.Stage_3_1 = RefinementBlockType(stage_2_in, n_heatmaps)
        self.Stage_3_2 = RefinementBlockType(stage_2_in, n_pafs)
        self.Stage_4_1 = RefinementBlockType(stage_2_in, n_heatmaps)
        self.Stage_4_2 = RefinementBlockType(stage_2_in, n_pafs)
        self.Stage_5_1 = RefinementBlockType(stage_2_in, n_heatmaps)
        self.Stage_5_2 = RefinementBlockType(stage_2_in, n_pafs)
        self.Stage_6_1 = RefinementBlockType(stage_2_in, n_heatmaps)
        self.Stage_6_2 = RefinementBlockType(stage_2_in, n_pafs)
    
    
    def forward(self, x):
        image_features = self.F(x)                                           #128x28x28
        o_1_1 = self.Stage_1_1(image_features)                               #17x28x28
        o_1_2 = self.Stage_1_2(image_features)                               #38x28x28
        
        i_2 = torch.cat((image_features, o_1_1, o_1_2), dim=1)               #(128+17+38)x28x28
        o_2_1 = self.Stage_2_1(i_2)                                          #17x28x28
        o_2_2 = self.Stage_2_2(i_2)                                          #38x28x28
        
        i_3 = torch.cat((image_features, o_2_1, o_2_2), dim=1)               #(128+17+38)x28x28
        o_3_1 = self.Stage_3_1(i_3)                                          #17x28x28
        o_3_2 = self.Stage_3_2(i_3)                                          #38x28x28
        
        i_4 = torch.cat((image_features, o_3_1, o_3_2), dim=1)               #(128+17+38)x28x28
        o_4_1 = self.Stage_4_1(i_4)                                          #17x28x28
        o_4_2 = self.Stage_4_2(i_4)                                          #38x28x28
        
        i_5 = torch.cat((image_features, o_4_1, o_4_2), dim=1)               #(128+17+38)x28x28
        o_5_1 = self.Stage_5_1(i_5)                                          #17x28x28
        o_5_2 = self.Stage_5_2(i_5)                                          #38x28x28
        
        i_6 = torch.cat((image_features, o_5_1, o_5_2), dim=1)               #(128+17+38)x28x28
        o_6_1 = self.Stage_6_1(i_6)                                          #17x28x28
        o_6_2 = self.Stage_6_2(i_6)                                          #38x28x28
        
        return o_6_1, o_6_2


#hm,paf = net(torch.ones(1,3,224,224))
