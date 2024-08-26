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


def get_client_ranks(w_locals_client):
    client_ranks = []
    for w_client in w_locals_client:
        for key, value in w_client.items():
            if key.endswith("lora_A"):
                client_ranks.append(value.shape[1])
                break
    return client_ranks


def pad_lora_weights(w_locals_client_lora):
    max_rank_A = max(w["lora_A"].shape[1] for w in w_locals_client_lora)
    max_rank_B = max(w["lora_B"].shape[0] for w in w_locals_client_lora)

    padded_weights = []
    for w in w_locals_client_lora:
        padded_w = {}
        for key, value in w.items():
            if key.endswith("lora_A"):
                padded_w[key] = torch.nn.functional.pad(
                    value, (0, max_rank_A - value.shape[1])
                )
            elif key.endswith("lora_B"):
                padded_w[key] = torch.nn.functional.pad(
                    value, (0, 0, 0, max_rank_B - value.shape[0])
                )
            else:
                padded_w[key] = value
        padded_weights.append(padded_w)
    return padded_weights


def slice_lora_for_clients(global_weights, client_ranks):
    sliced_weights = []
    for rank in client_ranks:
        client_weights = copy.deepcopy(global_weights)
        for key, value in client_weights.items():
            if key.endswith("lora_A"):
                client_weights[key] = value[:, :rank]
            elif key.endswith("lora_B"):
                client_weights[key] = value[:rank, :]
        sliced_weights.append(client_weights)
    return sliced_weights


def aggregate(
    w_locals_client,
    global_client_weight,
):
    w_locals_client_lora = []
    client_norms = []

    # Collect LoRA weights and calculate norms
    for w_client in w_locals_client:
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
        w_locals_client_lora.append(client_temp_dict)
        client_norms.append(client_norm)

    # Pad LoRA weights
    padded_w_locals_client_lora = pad_lora_weights(w_locals_client_lora)

    # Calculate sparsity weights
    total_norm = sum(client_norms)
    client_weights = [norm / total_norm for norm in client_norms]

    # Sparsity-weighted aggregation
    w_glob_client_lora = {}
    for idx, w_client in enumerate(padded_w_locals_client_lora):
        for key, value in w_client.items():
            if key in w_glob_client_lora:
                w_glob_client_lora[key] += client_weights[idx] * value
            else:
                w_glob_client_lora[key] = client_weights[idx] * value

    # Prepare global weights
    w_glob_client_lora_new = {}
    for key, value in w_glob_client_lora.items():
        new_key = "client_transformer." + key
        w_glob_client_lora_new[new_key] = value

    # Update global weights
    for key, _ in global_client_weight.items():
        if key.endswith("lora_A") or key.endswith("lora_B"):
            global_client_weight[key] = w_glob_client_lora_new[key]

    # Get client ranks
    client_ranks = get_client_ranks(w_locals_client)

    # Slice LoRA matrices for each client based on their rank
    sliced_client_weights = slice_lora_for_clients(global_client_weight, client_ranks)

    return sliced_client_weights, global_client_weight
