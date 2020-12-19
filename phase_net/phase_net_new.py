import torch
from torch._C import has_openmp
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import namedtuple

DecompValues = namedtuple(
    'values',
    'high_level, '
    'phase, '
    'amplitude, '
    'low_level'
)

class PhaseNet(nn.Module):
    def __init__(self, pyr, device, optimizer=torch.optim.Adam, batch_size=32, lR=0.001,
                 pic_func=torch.nn.L1Loss, face_loss=None):
        super(PhaseNet, self).__init__()
        self.height = pyr.height
        self.optimizer = optimizer
        self.batch = batch_size
        self.lR = lR
        self.pic_func = pic_func
        self.face_loss = face_loss
        self.device = device
        self.layers = self.create_architecture()

    def create_architecture(self):
        return [
            PhaseNetBlock(2, 64, 1, (1, 1), self.device),
            PhaseNetBlock(64 + 1 + 16, 64, 8, (1, 1), self.device),
            PhaseNetBlock(64 + 8 + 16, 64, 8, (1, 1), self.device),
            *[PhaseNetBlock(64 + 8 + 16, 64, 8, (3 ,3), self.device) for _ in range(8)]
        ]

    def forward(self, vals, vals2):
        #(1, C_in, H, W)

        # Phase Net -> 2 image Values
        # high level [256, 256, 1]
        # low level [8, 8, 1]
        # phase 10
        # 0 [256, 256, 2, 4]
        # 1 [182, 182, 2, 4]
        # 2 [128, 128, 2, 4]
        # 3 [90, 90, 2, 4]
        # 4 [64, 64, 2, 4]
        # 5 [46, 46, 2, 4]
        # 6 [32, 32, 2, 4]
        # 7 [22, 22, 2, 4]
        # 8 [16, 16, 2, 4]
        # 9 [12, 12, 2, 12] -> [12, 12, 2, 4] -> [12, 12, 8]
        # amplitude
        # 0 [256, 256, 2, 4]
        # 1 [182, 182, 2, 4]
        # 2 [128, 128, 2, 4]
        # 3 [90, 90, 2, 4]
        # 4 [64, 64, 2, 4]
        # 5 [46, 46, 2, 4]
        # 6 [32, 32, 2, 4]
        # 7 [22, 22, 2, 4]
        # 8 [16, 16, 2, 4]

        img = torch.stack((vals.low_level[:, :, 0], vals2.low_level[:, :, 0]), 0).unsqueeze(0)

        feature, prediction = self.layers[0](img)

        phase_dim = vals.phase[::-1]
        phase_dim2 = vals2.phase[::-1]

        high_level = prediction
        phase = []

        for idx in range(1, len(self.layers)-1):
            res = phase_dim[idx-1].shape[0]
            resized_feature = torch.nn.Upsample((res, res), mode='bilinear')(feature)
            resized_prediction = torch.nn.Upsample((res, res), mode='bilinear')(prediction)

            img1 = phase_dim[idx-1][:,:,:, :4].reshape(res, res, -1, 1).permute(3, 2, 0, 1)
            img2 = phase_dim2[idx-1][:,:,:, :4].reshape(res, res, -1, 1).permute(3, 2, 0, 1)
            concat = torch.cat((resized_feature, img1, img2, resized_prediction), 1)

            feature, prediction = self.layers[idx](concat)

            print(prediction.shape)

            permuted_prediction = prediction.squeeze(0).permute(1, 2, 0)
            phase.append(permuted_prediction.reshape((permuted_prediction.shape[0],
                                                     permuted_prediction.shape[1],
                                                     2, 4)))

            print(phase[-1].shape)
            #+exit()

        print(feature.shape, prediction.shape)
        low_level = []
        amplitude = []

        values = DecompValues(
            high_level=high_level,
            low_level=low_level,
            phase=phase,
            amplitude=amplitude
        )

        return values



class PhaseNetBlock(nn.Module):
    """
    PhaseNet Block
    """
    def __init__(self, c_in, c_out, pred_out, kernel_size, device):
        super(PhaseNetBlock, self).__init__()

        padding = 0
        if kernel_size == (3, 3):
            padding = 1

        self.feature_map = nn.Sequential(
            nn.Conv2d(c_in, c_out, kernel_size, padding=padding, padding_mode='reflect'),
            nn.ReLU(),
            nn.BatchNorm2d(c_out),
            nn.Conv2d(c_out, c_out, kernel_size, padding=padding, padding_mode='reflect'),
            nn.ReLU()
        )
        self.prediction_map = nn.Sequential(
            nn.Conv2d(c_out, pred_out, (1, 1)),
            nn.Tanh()
        )
        self.to(device)

    def forward(self, x):
       f = self.feature_map(x)
       c = self.prediction_map(f)

       return f,c
