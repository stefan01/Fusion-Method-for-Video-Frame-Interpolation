import torch
import torch.nn as nn


def get_loss(vals_o, vals_t, output, target, pyr, weighting_factor=0.002):
    """ PhaseNet special loss. """
    phase_loss = 0
    l1loss = nn.L1Loss()

    for idx, (phase_r, phase_g) in enumerate(zip(vals_o.phase, vals_t.phase)):
        phase_r_2 = phase_r.reshape(-1, pyr.nbands, phase_r.shape[2], phase_r.shape[3]).permute(1, 0, 2, 3)
        phase_g_2 = phase_g.reshape(-1, pyr.nbands, phase_r.shape[2], phase_r.shape[3]).permute(1, 0, 2, 3)

        for (orientation_r, orientation_g) in zip(phase_r_2, phase_g_2):
            delta_psi = torch.atan2(torch.sin(orientation_g - orientation_r), torch.cos(orientation_g - orientation_r))
            phase_loss += torch.mean(torch.abs(delta_psi.reshape(-1)), 0)
            #l1loss(delta_psi, torch.zeros(delta_psi.shape, device=delta_psi.device))

    #low_loss = l1loss(vals_o.low_level, vals_t.low_level)
    l_1 = l1loss(output, target)

    total_loss = l_1 + weighting_factor*phase_loss
    l_1_p = 100*l_1.detach() / total_loss
    phase_loss_p = 100*weighting_factor*phase_loss.detach() / total_loss

    return total_loss, l_1_p, phase_loss_p