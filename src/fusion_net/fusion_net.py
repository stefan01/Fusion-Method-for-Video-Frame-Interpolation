import torch
from src.adacof.utility import moduleNormalize
from torch.nn import functional as F

class FusionNet(torch.nn.Module):

    def __init__():
        super(FusionNet, self).__init__()

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

        self.moduleOcclusion = Subnet_occlusion()

    def forward(self, adacof, phase):
        h0 = int(list(adacof.size())[2])
        w0 = int(list(adacof.size())[3])
        h2 = int(list(phase.size())[2])
        w2 = int(list(phase.size())[3])
        if h0 != h2 or w0 != w2:
            sys.exit('Frame sizes do not match')

        h_padded = False
        w_padded = False
        if h0 % 32 != 0:
            pad_h = 32 - (h0 % 32)
            adacof = F.pad(adacof, (0, 0, 0, pad_h), mode='reflect')
            phase = F.pad(phase, (0, 0, 0, pad_h), mode='reflect')
            h_padded = True

        if w0 % 32 != 0:
            pad_w = 32 - (w0 % 32)
            adacof = F.pad(adacof, (0, pad_w, 0, 0), mode='reflect')
            phase = F.pad(phase, (0, pad_w, 0, 0), mode='reflect')
            w_padded = True

        combined = torch.cat([moduleNormalize(adacof), moduleNormalize(phase)], 1)

        Occlusion = self.moduleOcclusion(combined)

        result_frame = Occlusion * adacof + (1 - Occlusion) * phase

        if h_padded:
            result_frame = result_frame[:, :, 0:h0, :]
        if w_padded:
            result_frame = result_frame[:, :, :, 0:w0]

        print(result_frame.shape)

        return result_frame