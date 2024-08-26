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


def sparsity_weighted_aggregation(w_local_clients, w_global_client):
    w_locals_client_lora = []
    client_norms = []

    # Collect LoRA weights and calculate norms
    for w_client in w_local_clients:
        client_temp_dict = {}
        client_norm = 0
        for key, value in w_client.items():
            if key.endswith("lora_A") or key.endswith("lora_B"):
                client_temp_dict[key] = value
                if key.endswith("lora_A"):
                    A = value
                elif key.endswith("lora_B"):
                    B = value
                    # Calculate norm for this LoRA pair
                    client_norm += torch.norm(A @ B).item()
        w_locals_client_lora.append(copy.deepcopy(client_temp_dict))
        client_norms.append(client_norm)

    # Calculate sparsity weights
    total_norm = sum(client_norms)
    client_weights = [norm / total_norm for norm in client_norms]

    # Sparsity-weighted aggregation
    w_glob_client_lora = {}
    for idx, w_client in enumerate(w_locals_client_lora):
        for key, value in w_client.items():
            if key in w_glob_client_lora:
                w_glob_client_lora[key] += client_weights[idx] * value
            else:
                w_glob_client_lora[key] = client_weights[idx] * value

    w_glob_client_lora_new = {}
    for key, value in w_glob_client_lora.items():
        new_key = "client_transformer." + key
        w_glob_client_lora_new[new_key] = value

    for key, _ in w_global_client.items():
        if key.endswith("lora_A") or key.endswith("lora_B"):
            w_global_client[key] = w_glob_client_lora_new[key]

    return w_global_client
