import copy

import torch


def fed_avg(w):
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[k] += w[i][k]
        w_avg[k] = torch.div(w_avg[k], len(w))
    return w_avg


def self_pruning_regularisation(lora_layers, gamma):
    reg_loss = 0
    for layer in lora_layers:
        current_rank = layer.A.shape[1]
        prune_rank = int(gamma * current_rank)

        A_prune_norm = torch.norm(layer.A[:, prune_rank:])
        B_prune_norm = torch.norm(layer.B[prune_rank:, :])

        reg_loss += A_prune_norm * B_prune_norm

    return reg_loss
