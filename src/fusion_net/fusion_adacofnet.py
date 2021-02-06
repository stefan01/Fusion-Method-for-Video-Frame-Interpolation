import torch
import src.adacof.cupy_module.adacof as adacof
import sys
from torch.nn import functional as F
from src.adacof.utility import CharbonnierFunc, moduleNormalize
import matplotlib.pyplot as plt
import numpy as np


def make_model(args):
    return AdaCoFNet(args).to(torch.device('cuda:{}'.format(args.gpu_id)))


class KernelEstimation(torch.nn.Module):
    def __init__(self, kernel_size):
        super(KernelEstimation, self).__init__()
        self.kernel_size = kernel_size

        def Basic(input_channel, output_channel):
            return torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=input_channel, out_channels=output_channel, kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(inplace=False),
                torch.nn.Conv2d(in_channels=output_channel, out_channels=output_channel, kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(inplace=False),
                torch.nn.Conv2d(in_channels=output_channel, out_channels=output_channel, kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(inplace=False)
            )

        def Upsample(channel):
            return torch.nn.Sequential(
                torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                torch.nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(inplace=False)
            )

        def Subnet_offset(ks):
            return torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(inplace=False),
                torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(inplace=False),
                torch.nn.Conv2d(in_channels=64, out_channels=ks, kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(inplace=False),
                torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                torch.nn.Conv2d(in_channels=ks, out_channels=ks, kernel_size=3, stride=1, padding=1)
            )

        def Subnet_weight(ks):
            return torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(inplace=False),
                torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(inplace=False),
                torch.nn.Conv2d(in_channels=64, out_channels=ks, kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(inplace=False),
                torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                torch.nn.Conv2d(in_channels=ks, out_channels=ks, kernel_size=3, stride=1, padding=1),
                torch.nn.Softmax(dim=1)
            )

        def Subnet_occlusion():
            return torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(inplace=False),
                torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(inplace=False),
                torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(inplace=False),
                torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                torch.nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3, stride=1, padding=1),
                torch.nn.Sigmoid()
            )

        self.moduleConv1 = Basic(6, 32)
        self.modulePool1 = torch.nn.AvgPool2d(kernel_size=2, stride=2)

        self.moduleConv2 = Basic(32, 64)
        self.modulePool2 = torch.nn.AvgPool2d(kernel_size=2, stride=2)

        self.moduleConv3 = Basic(64, 128)
        self.modulePool3 = torch.nn.AvgPool2d(kernel_size=2, stride=2)

        self.moduleConv4 = Basic(128, 256)
        self.modulePool4 = torch.nn.AvgPool2d(kernel_size=2, stride=2)

        self.moduleConv5 = Basic(256, 512)
        self.modulePool5 = torch.nn.AvgPool2d(kernel_size=2, stride=2)

        self.moduleDeconv5 = Basic(512, 512)
        self.moduleUpsample5 = Upsample(512)

        self.moduleDeconv4 = Basic(512, 256)
        self.moduleUpsample4 = Upsample(256)

        self.moduleDeconv3 = Basic(256, 128)
        self.moduleUpsample3 = Upsample(128)

        self.moduleDeconv2 = Basic(128, 64)
        self.moduleUpsample2 = Upsample(64)

        self.moduleWeight1 = Subnet_weight(self.kernel_size ** 2)
        self.moduleAlpha1 = Subnet_offset(self.kernel_size ** 2)
        self.moduleBeta1 = Subnet_offset(self.kernel_size ** 2)
        self.moduleWeight2 = Subnet_weight(self.kernel_size ** 2)
        self.moduleAlpha2 = Subnet_offset(self.kernel_size ** 2)
        self.moduleBeta2 = Subnet_offset(self.kernel_size ** 2)
        self.moduleOcclusion = Subnet_occlusion()

    def forward(self, rfield0, rfield2):
        tensorJoin = torch.cat([rfield0, rfield2], 1)

        tensorConv1 = self.moduleConv1(tensorJoin)
        tensorPool1 = self.modulePool1(tensorConv1)

        tensorConv2 = self.moduleConv2(tensorPool1)
        tensorPool2 = self.modulePool2(tensorConv2)

        tensorConv3 = self.moduleConv3(tensorPool2)
        tensorPool3 = self.modulePool3(tensorConv3)

        tensorConv4 = self.moduleConv4(tensorPool3)
        tensorPool4 = self.modulePool4(tensorConv4)

        tensorConv5 = self.moduleConv5(tensorPool4)
        tensorPool5 = self.modulePool5(tensorConv5)

        tensorDeconv5 = self.moduleDeconv5(tensorPool5)
        tensorUpsample5 = self.moduleUpsample5(tensorDeconv5)

        tensorCombine = tensorUpsample5 + tensorConv5

        tensorDeconv4 = self.moduleDeconv4(tensorCombine)
        tensorUpsample4 = self.moduleUpsample4(tensorDeconv4)

        tensorCombine = tensorUpsample4 + tensorConv4

        tensorDeconv3 = self.moduleDeconv3(tensorCombine)
        tensorUpsample3 = self.moduleUpsample3(tensorDeconv3)

        tensorCombine = tensorUpsample3 + tensorConv3

        tensorDeconv2 = self.moduleDeconv2(tensorCombine)
        tensorUpsample2 = self.moduleUpsample2(tensorDeconv2)

        tensorCombine = tensorUpsample2 + tensorConv2

        Weight1 = self.moduleWeight1(tensorCombine)
        Alpha1 = self.moduleAlpha1(tensorCombine)
        Beta1 = self.moduleBeta1(tensorCombine)
        Weight2 = self.moduleWeight2(tensorCombine)
        Alpha2 = self.moduleAlpha2(tensorCombine)
        Beta2 = self.moduleBeta2(tensorCombine)
        Occlusion = self.moduleOcclusion(tensorCombine)

        return Weight1, Alpha1, Beta1, Weight2, Alpha2, Beta2, Occlusion


class AdaCoFNet(torch.nn.Module):
    def __init__(self, args):
        super(AdaCoFNet, self).__init__()
        self.args = args
        self.kernel_size = args.kernel_size
        self.kernel_pad = int(((args.kernel_size - 1) * args.dilation) / 2.0)
        self.dilation = args.dilation

        self.get_kernel = KernelEstimation(self.kernel_size)

        self.modulePad = torch.nn.ReplicationPad2d([self.kernel_pad, self.kernel_pad, self.kernel_pad, self.kernel_pad])

        self.moduleAdaCoF = adacof.FunctionAdaCoF.apply

    def forward(self, frame0, frame2):
        h0 = int(list(frame0.size())[2])
        w0 = int(list(frame0.size())[3])
        h2 = int(list(frame2.size())[2])
        w2 = int(list(frame2.size())[3])
        if h0 != h2 or w0 != w2:
            sys.exit('Frame sizes do not match')

        h_padded = False
        w_padded = False
        if h0 % 32 != 0:
            pad_h = 32 - (h0 % 32)
            frame0 = F.pad(frame0, (0, 0, 0, pad_h), mode='reflect')
            frame2 = F.pad(frame2, (0, 0, 0, pad_h), mode='reflect')
            h_padded = True

        if w0 % 32 != 0:
            pad_w = 32 - (w0 % 32)
            frame0 = F.pad(frame0, (0, pad_w, 0, 0), mode='reflect')
            frame2 = F.pad(frame2, (0, pad_w, 0, 0), mode='reflect')
            w_padded = True
        Weight1, Alpha1, Beta1, Weight2, Alpha2, Beta2, Occlusion = self.get_kernel(moduleNormalize(frame0), moduleNormalize(frame2))

        tensorAdaCoF1 = self.moduleAdaCoF(self.modulePad(frame0), Weight1, Alpha1, Beta1, self.dilation)
        tensorAdaCoF2 = self.moduleAdaCoF(self.modulePad(frame2), Weight2, Alpha2, Beta2, self.dilation)

        frame1 = Occlusion * tensorAdaCoF1 + (1 - Occlusion) * tensorAdaCoF2
        if h_padded:
            tensorAdaCoF1 = tensorAdaCoF1[:, :, 0:h0, :]
            tensorAdaCoF2 = tensorAdaCoF2[:, :, 0:h0, :]
            frame1 = frame1[:, :, 0:h0, :]
        if w_padded:
            tensorAdaCoF1 = tensorAdaCoF2[:, :, :, 0:w0]
            tensorAdaCoF2 = tensorAdaCoF2[:, :, :, 0:w0]
            frame1 = frame1[:, :, :, 0:w0]
            
        # Mean and Variance Flow Map
        DeltaP1 = torch.stack([Alpha1, Beta1], 0)
        DeltaP2 = torch.stack([Alpha2, Beta2], 0)

        MeanFlowMap1 = (Weight1*DeltaP1).sum(-3)
        MeanFlowMap2 = (Weight2*DeltaP2).sum(-3)
        
        VarFlowMap1 = (Weight1 * ((MeanFlowMap1 - DeltaP1.permute(2, 0, 1, 3, 4))**2).permute(1, 2, 0, 3, 4)).sum(-3)
        VarFlowMap2 = (Weight2 * ((MeanFlowMap2 - DeltaP2.permute(2, 0, 1, 3, 4))**2).permute(1, 2, 0, 3, 4)).sum(-3)

        # Calculate Uncetainty Mask
        UncertaintyMask = torch.max(VarFlowMap1.sum(0), VarFlowMap2.sum(0))
        UncertaintyMask = torch.clip(UncertaintyMask, 0, 20)/20
        #UncertaintyMask = UncertaintyMask.permute(1,2,0).detach().cpu().numpy()

        return tensorAdaCoF1, tensorAdaCoF2, frame1