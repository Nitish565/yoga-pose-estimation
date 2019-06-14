import torch
from torch import nn

from modules.conv import conv, conv_dw, conv_dw_no_bn
from collections import OrderedDict


class Cpm(nn.Module):
    def __init__(self, in_channels, out_channels):#--------512,128
        super().__init__()
        self.align = conv(in_channels, out_channels, kernel_size=1, padding=0, bn=False)
        self.trunk = nn.Sequential(
            conv_dw_no_bn(out_channels, out_channels),  #kernel_size=3, padding=1, stride=1, dilation=1
            conv_dw_no_bn(out_channels, out_channels),  #kernel_size=3, padding=1, stride=1, dilation=1
            conv_dw_no_bn(out_channels, out_channels)   #kernel_size=3, padding=1, stride=1, dilation=1
        )
        self.conv = conv(out_channels, out_channels, bn=False)

    def forward(self, x):
        x = self.align(x)
        x = self.conv(x + self.trunk(x))#------------------------------------------------concatenation from mobilenet and trunk
        return x


# class InitialStage(nn.Module):
#     def __init__(self, num_channels, num_heatmaps, num_pafs):
#         super().__init__()
#         self.trunk = nn.Sequential(#----------------------------------------Original OPENPOSE contained two copies of these conv() layers, for PAF and HMs
#             conv(num_channels, num_channels, bn=False),#--------------------Here just 1 in the original lightweight openpose
#             conv(num_channels, num_channels, bn=False),
#             conv(num_channels, num_channels, bn=False)
#         )
#         #--------------------------------------------------------------------Then, give the op to both paf and HM branches
#         self.heatmaps = nn.Sequential(
#             conv(num_channels, 512, kernel_size=1, padding=0, bn=False),
#             conv(512, num_heatmaps, kernel_size=1, padding=0, bn=False, relu=False)
#         )
#         self.pafs = nn.Sequential(
#             conv(num_channels, 512, kernel_size=1, padding=0, bn=False),
#             conv(512, num_pafs, kernel_size=1, padding=0, bn=False, relu=False)
#         )
#
#     def forward(self, x):
#         trunk_features = self.trunk(x)
#         heatmaps = self.heatmaps(trunk_features)
#         pafs = self.pafs(trunk_features)
#         return [heatmaps, pafs]
#
#
# class RefinementStageBlock(nn.Module):#---------------------------------Each refinement stage block should be a [3*3 , 3*3 , 1*1] with residual connection
#     def __init__(self, in_channels, out_channels):#---------------------for each such block
#         super().__init__()
#         self.initial = conv(in_channels, out_channels, kernel_size=1, padding=0, bn=False)
#         self.trunk = nn.Sequential(
#             conv(out_channels, out_channels),
#             conv(out_channels, out_channels, dilation=2, padding=2)
#         )
#
#     def forward(self, x):
#         initial_features = self.initial(x)
#         trunk_features = self.trunk(initial_features)
#         return initial_features + trunk_features
#
#
# class RefinementStage(nn.Module):#------------------------------------------original lightweight contains a set of
#     def __init__(self, in_channels, out_channels, num_heatmaps, num_pafs):
#         super().__init__()
#         self.trunk = nn.Sequential(#----------------------------------------------------In OpenPose five 7 * 7,
#             RefinementStageBlock(in_channels, out_channels),#---------------------------here replaced with  five time [3*3 , 3*3 , 1*1]
#             RefinementStageBlock(out_channels, out_channels),
#             RefinementStageBlock(out_channels, out_channels),
#             RefinementStageBlock(out_channels, out_channels),
#             RefinementStageBlock(out_channels, out_channels)
#         )
#         self.heatmaps = nn.Sequential(
#             conv(out_channels, out_channels, kernel_size=1, padding=0, bn=False),
#             conv(out_channels, num_heatmaps, kernel_size=1, padding=0, bn=False, relu=False)
#         )
#         self.pafs = nn.Sequential(
#             conv(out_channels, out_channels, kernel_size=1, padding=0, bn=False),
#             conv(out_channels, num_pafs, kernel_size=1, padding=0, bn=False, relu=False)
#         )
#
#     def forward(self, x):
#         trunk_features = self.trunk(x)
#         heatmaps = self.heatmaps(trunk_features)
#         pafs = self.pafs(trunk_features)
#         return [heatmaps, pafs]


#--------------------------------------------------------------------------------New Initial Stages-------------------------------------------------

class InitialStage(nn.Module):
    def __init__(self, num_channels, num_heatmaps, num_pafs):#----------num channels means no of in-channels and out-channels for each layer
        super().__init__()
        self.trunk = nn.Sequential(
            conv(num_channels, num_channels, bn=False),
            conv(num_channels, num_channels, bn=False),
            conv(num_channels, num_channels, bn=False)#-----------------------------------
        )
        self.heatmaps = nn.Sequential(
            #conv(num_channels, 512, kernel_size=1, padding=0, bn=False),
            conv(num_channels, 128, kernel_size=1, padding=0, bn=False),

            #conv(512, num_heatmaps, kernel_size=1, padding=0, bn=False, relu=False)
            conv(128, num_heatmaps, kernel_size=1, padding=0, bn=False, relu=False)

        )
        self.pafs = nn.Sequential(
            #conv(num_channels, 512, kernel_size=1, padding=0, bn=False),
            conv(num_channels, 128, kernel_size=1, padding=0, bn=False),

            #conv(512, num_pafs, kernel_size=1, padding=0, bn=False, relu=False)
            conv(128, num_pafs, kernel_size=1, padding=0, bn=False, relu=False)

        )

    def forward(self, x):
        trunk_features = self.trunk(x)#-----------------------------trunk means the single-branch path before it splits
        heatmaps = self.heatmaps(trunk_features)
        pafs = self.pafs(trunk_features)
        return [heatmaps, pafs]
#----------------------------------------------------------------------------------New Refinement Stage block------------------------------------------

class RefinementStageBlock(nn.Module):#------------------------------------------------This one block contains
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.initial = conv(in_channels, out_channels, kernel_size=1, padding=0, bn=False)#----reduce the no of channels to 128
        self.trunk = nn.Sequential(
            conv(out_channels, out_channels),#---------------------------and continue working on only 128 channels in refinement stage
            conv(out_channels, out_channels, dilation=2, padding=2)
        )

    def forward(self, x):
        initial_features = self.initial(x)#------------------------first 1*1 convolution to reduce the number of channels
        trunk_features = self.trunk(initial_features)
        return initial_features + trunk_features#------------------skip connections between blocks even within a single refinement stage


#-------------------------------------------------------------------------------New Refinement Stage------------------------------------------------------
class RefinementStage(nn.Module):
    def __init__(self, in_channels, out_channels, num_heatmaps, num_pafs):#------outchannels converted to 64 from 128 for this call
        super().__init__()
        self.trunk = nn.Sequential(
            RefinementStageBlock(in_channels, out_channels),#---------------
            RefinementStageBlock(out_channels, out_channels),
            RefinementStageBlock(out_channels, out_channels),
            # RefinementStageBlock(out_channels, out_channels),#-------------Remove two refinement blocks from each refinement stage
            # RefinementStageBlock(out_channels, out_channels)
        )
        self.heatmaps = nn.Sequential(
            conv(out_channels, out_channels, kernel_size=1, padding=0, bn=False),
            conv(out_channels, num_heatmaps, kernel_size=1, padding=0, bn=False, relu=False)
        )
        self.pafs = nn.Sequential(
            conv(out_channels, out_channels, kernel_size=1, padding=0, bn=False),
            conv(out_channels, num_pafs, kernel_size=1, padding=0, bn=False, relu=False)
        )

    def forward(self, x):
        trunk_features = self.trunk(x)
        heatmaps = self.heatmaps(trunk_features)
        pafs = self.pafs(trunk_features)
        return [heatmaps, pafs]
#------------------------------------------------------------------------------------------------------------------------------------------------------------


class PoseEstimationWithMobileNet(nn.Module):
    def __init__(self, num_refinement_stages=1, num_channels=128, num_heatmaps=13, num_pafs=32):#-------


        super().__init__()
        #self.model = nn.Sequential(#---------------------------------------------------MobileNet-----------------------------------first
            # conv(     3,  32, stride=2, bias=False),
            # conv_dw( 32,  64),#---------------------------------------------dw means depthwise convolution
            # conv_dw( 64, 128, stride=2),
            # conv_dw(128, 128),
            # conv_dw(128, 256, stride=2),
            # conv_dw(256, 256),
            # conv_dw(256, 512),  # Stride of this conv4_2 removed from MobileNetv1 to preserve the receptive field. conv4_2 means kernel 4 * 4, padding 2
            # conv_dw(512, 512, dilation=2, padding=2),
            # conv_dw(512, 512),
            # conv_dw(512, 512),
            # conv_dw(512, 512),
            # conv_dw(512, 512)   # conv5_5 The last layer used from MobileNet is this one.
            # )


        #----------define all layers of mobilenet individually to add feature pyramid struture in between----------------------------
        self.model = nn.Sequential(OrderedDict({    
            'conv1': conv(in_channels=3, out_channels=32, stride=2, bias=False),
            'conv2': conv_dw( 32,  64),
            'conv3': conv_dw( 64, 128, stride=2),
            'conv4': conv_dw(128, 128),
            'conv5': conv_dw(128, 128, stride=2),
            'conv6': conv_dw(128, 128),
            'conv8': conv_dw(128, 128, dilation=2, padding=2),
            'conv9': conv_dw(128, 128),
            'conv10': conv_dw(128, 128, dilation=4, padding=4),
            'mp11': nn.MaxPool2d(kernel_size= 2, stride= 2),
            'mp12': nn.MaxPool2d(kernel_size= 2, stride= 4),
            'conv13_1x1': nn.Conv2d(kernel_size= 1, in_channels= 128, out_channels= 128),
            'upsample13': nn.Upsample(size=23, mode='nearest'),
            'conv14_1x1': nn.Conv2d(kernel_size= 1, in_channels= 128, out_channels= 128),
            'upsample14': nn.Upsample(size=46, mode='nearest')
            
        }))
        '''
        self.conv1 = conv(in_channels=3, out_channels=32, stride=2, bias=False)
        self.conv2 = conv_dw( 32,  64)
        self.conv3 = conv_dw( 64, 128, stride=2)

        # #---------first skip-------------------
        self.conv4 = conv_dw(128, 128)
        self.conv5 = conv_dw(128, 128, stride=2)


        #---------second skip------------------
        self.conv6 = conv_dw(128, 128)
        self.conv8 = conv_dw(128, 128, dilation=2, padding=2)

        #---------third skip--------------------
        self.conv9 = conv_dw(128, 128)
        self.conv10 = conv_dw(128, 128, dilation=4, padding=4)
        
        self.mp11 = nn.MaxPool2d(kernel_size= 2, stride= 2)
        self.mp12 = nn.MaxPool2d(kernel_size= 2, stride= 4)
        self.conv13_1x1 = nn.Conv2d(kernel_size= 1, in_channels= 128, out_channels= 128)
        self.upsample13 = nn.Upsample(size=23, mode='nearest')
        
        self.conv14_1x1 = nn.Conv2d(kernel_size= 1, in_channels= 128, out_channels= 128)
        self.upsample14 = nn.Upsample(size=46, mode='nearest')
        
        '''
        #----------------initial stage--------------------------
        self.initial_stage = InitialStage(num_channels, num_heatmaps, num_pafs)#--------------------------------------------third


        #-------------------refinement stage---------------------------
        self.refinement_stages = nn.ModuleList()#------------------------------------------------------------------------fourth
        for idx in range(num_refinement_stages):
            self.refinement_stages.append(RefinementStage(num_channels + num_heatmaps + num_pafs, #------in-channels, concatenated the three
                                                          64,num_heatmaps, num_pafs))# the outchannels converted to 64 from the default 128 for the refinement stage

    def forward(self, x):
        #-------------------------------------Mobilenet with feature pyramid
        #print("INPUT to mobile net", x.shape)
        
        x_after_conv5 = self.model.conv5(self.model.conv4(self.model.conv3(self.model.conv2(self.model.conv1(x)))))
        #print("conv5", x_after_conv5.shape)
        

        x_after_conv8 = self.model.conv8(self.model.conv6(x_after_conv5))#-------------------------
        #print("conv8", x.shape)

        input_to_conv_9 = torch.add(x_after_conv5, x_after_conv8)
        cv10 = self.model.conv10(self.model.conv9(input_to_conv_9))
        
        #print("cv10",cv10.shape)
        p1 = self.model.mp11(cv10)
        #print("p1",p1.shape)
        p2 = self.model.mp12(cv10)
        #print("p2",p2.shape)
        p1 = torch.add(p1, self.model.upsample13(self.model.conv13_1x1(p2)))
        cv10 = torch.add(cv10, self.model.upsample14(self.model.conv14_1x1(p1)))
        backbone_features = torch.add(cv10, x_after_conv8)
        #print("cv10 op bkbn", cv10.shape)
        #print("bkbn", backbone_features.shape)
        #backbone_features = torch.add(x_after_conv8,x)
        #print("conv10", x.shape)

        # x = self.conv11(x)
        # #print("conv11", x.shape)
        #
        # backbone_features = self.conv12(x)
        # print("conv12", x.shape)

        # import time
        # time.sleep(2222)



        #backbone_features = self.model(x)
        #backbone_features = self.cpm(backbone_features)#----------------------get backbone features

        stages_output = self.initial_stage(backbone_features)#----------------get PAfs and heatmaps

        for refinement_stage in self.refinement_stages:
            stages_output.extend(
                refinement_stage(torch.cat([backbone_features, stages_output[-2], stages_output[-1]], dim=1)))#----------refinement stages
                #we are concatenating the output of the CPM to the PAFs and Heatmaps
                #the dimensions of the op of CPM and heatmaps/ PAFs are equal.
                #-----torch.cat
                #Concatenates the given sequence of tensors in the given dimension.
                #All tensors must either have the same shape (except in the concatenating dimension) or be empty.

        return stages_output
