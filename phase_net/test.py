from numpy.lib.twodim_base import tri
from pyramid import Pyramid
from datareader import DBreader_Vimeo90k
from torch.utils.data import DataLoader, dataloader
import numpy as np
import torch
from collections import namedtuple
from skimage import io
import matplotlib.pyplot as plt

from phase_net_new import PhaseNet

from steerable.SCFpyr_PyTorch import SCFpyr_PyTorch
import steerable.utils as utils
import warnings
warnings.filterwarnings("ignore")


device = utils.get_device()

pyr = Pyramid(
    height=12,
    nbands=4,
    scale_factor=np.sqrt(2),
    device=device,
)

# Set batch size to default value 32
batch_size = 32
dataset = DBreader_Vimeo90k('./Trainset/vimeo/vimeo_triplet', random_crop=(256, 256))
train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=0)

#for triple in train_loader:
#    print(len(triple))
#    print(triple[0].shape) # [5, 3, 256, 256]
                           # [5*3, 1, 256, 256]

print(len(dataset[0])) #3
print(dataset[0][0].shape) # [3, 256, 256]
print(len(dataset)) # 73191

img = dataset[0][0].to(device).unsqueeze(1)
img2 = dataset[0][1].to(device).unsqueeze(1)

plt.subplot(1, 2, 1)
plt.imshow(img.cpu().squeeze(1).permute(1, 2, 0).numpy())

# Psi
vals = pyr.filter(img)
vals2 = pyr.filter(img2)

phase_net = PhaseNet(pyr, device)
# TODO vals concatenation
# vals = (values1, value2)

def get_concat_layers(pyr, vals, vals2):
    concat_layers = [torch.cat((vals.high_level, vals2.high_level))]

    for idx in range(len(pyr.phase)-2):
        phase = torch.cat((vals.phase[idx], vals2.phase[idx]))

        amplitude = torch.cat((vals.amplitude[idx], vals2.ampliture[idx]))

        concat_layers.append(torch.cat(phase, amplitude))

    concat_layers.append(torch.cat((vals.low_level, vals2.low_level)))

    return concat_layers

concat_layers = get_concat_layers(pyr, vals, vals2)

vals_r = phase_net(vals, vals2)

# Training
model = PhaseNet(pyr, device)
epoch = 10
learning_rate = 0.001
weight_decay = 0.0001
optimizer = optimizer = torch.optim.Adam(model.parameters(), 
                                        lr=learning_rate, 
                                        weight_decay=weight_decay)
loss_function = torch.nn.MSELoss()
loss_history = []

train = False
if train:
    for i in range(epoch):

        for triple in train_loader:
            
            # define heigth and width of the training image
            heigth = triple[0].shape[2]
            width = triple[0].shape[3]

            # get input data of first and second frame and the corresponding target image
            frame1 = triple[0].reshape((batch_size*3, 1, heigth, width)), 
            target = triple[1].reshape((batch_size*3, 1, heigth, width)), 
            frame2 = triple[2].reshape((batch_size*3, 1, heigth, width))

            # predicted intersected image of fram1 and frame2
            output = model(frame1, frame2)

            loss = loss_function(output, target)  # input, target

            # === The backpropagation
            # Reset the gradients
            optimizer.zero_grad()
            # Calculate new gradients with backpropagation
            loss.backward()
            # Tune weights accoring to optimizer (it has the learnrate and weight decay as defined above)
            optimizer.step()

            # To do output stuff with loss and image, we have to detach() and convert back to numpy.
            loss = loss.detach().numpy()
            # Append to loss history
            loss_history.append(loss)
    

# Phase Net -> 2 image Values
# high level [256, 256, 3]
# low level [8, 8, 3] -> [8, 8, 1], [8, 8, 1], [8, 8, 1]
# phase 10
    # 0 [256, 256, 2, 12]
    # 1 [182, 182, 2, 12]
    # 2 [128, 128, 2, 12]
    # 3 [90, 90, 2, 12]
    # 4 [64, 64, 2, 12]
    # 5 [46, 46, 2, 12]
    # 6 [32, 32, 2, 12]
    # 7 [22, 22, 2, 12]
    # 8 [16, 16, 2, 12]
    # 9 [12, 12, 2, 12]
# amplitude


# Psi^(-1)
img_r = pyr.inv_filter(vals_r)

plt.subplot(1, 2, 2)
plt.imshow(img_r.cpu().squeeze(1).permute(1, 2, 0).numpy())
plt.show()
