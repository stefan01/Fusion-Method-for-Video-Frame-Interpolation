import torch
import torch.nn as nn

class PhaseNetBlock(nn.Module):
    """
    PhaseNet Block: The basic building blocks of PhaseNet
    """
    def __init__(self, c_in, c_out, pred_out, kernel_size, device, dropout=0.5):
        super(PhaseNetBlock, self).__init__()

        padding = 0
        if kernel_size == (3, 3):
            padding = 1

        self.feature_map = nn.Sequential(
            nn.Conv2d(c_in, c_out, kernel_size, padding=padding, padding_mode='reflect'),
            nn.BatchNorm2d(c_out),
            nn.ELU(),
            nn.Conv2d(c_out, c_out, kernel_size, padding=padding, padding_mode='reflect'),
            nn.ELU(),
        )
        self.prediction_map = nn.Sequential(
            nn.Conv2d(c_out, pred_out, (1, 1), padding_mode='reflect'),
            nn.Tanh()
        )
        self.to(device)

    def forward(self, x):
        f = self.feature_map(x)
        c = self.prediction_map(f)

        return f, c
