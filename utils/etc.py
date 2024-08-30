import copy

import torch


def fed_avg(w):
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[k] += w[i][k]
        w_avg[k] = torch.div(w_avg[k], len(w))
    return w_avg


def self_pruning_regularisation(client_model, gamma):
    reg_loss = 0
    for key, value in client_model.named_parameters():
        if key.endswith("lora_A") or key.endswith("lora_B"):
            if key.endswith("lora_A"):
                current_rank = value.size()[0]
                prune_rank = int(gamma * current_rank) - (
                    current_rank - int(gamma * current_rank)
                )
                A_prune_norm = torch.norm(value[prune_rank:, :])
            elif key.endswith("lora_B"):
                current_rank = value.size()[1]
                prune_rank = int(gamma * current_rank)
                B_prune_norm = torch.norm(value[:, prune_rank:])
                reg_loss += A_prune_norm * B_prune_norm
    return reg_loss


def get_client_ranks(client_models):
    client_ranks = []
    for client in client_models:
        for key, value in client.state_dict().items():
            if key.endswith("lora_B"):
                client_ranks.append(value.shape[1])
                break
    return client_ranks


def pad_lora_weights(w_locals_client_lora, global_client_rank=16):
    padded_weights = []
    for w in w_locals_client_lora:
        padded_w = {}
        for key, value in w.items():
            if key.endswith("lora_A"):
                padded_w[key] = torch.nn.functional.pad(
                    value, (0, 0, 0, global_client_rank * 2 - value.shape[0])
                )
            elif key.endswith("lora_B"):
                padded_w[key] = torch.nn.functional.pad(
                    value, (0, global_client_rank - value.shape[1])
                )
            else:
                padded_w[key] = value
        padded_weights.append(padded_w)
    return padded_weights


def slice_lora_for_clients(global_client_weight, aggregated_lora_weight, client_ranks):
    readied_aggregated_lora_weight = {}
    for key, value in aggregated_lora_weight.items():
        new_key = "client_transformer." + key
        readied_aggregated_lora_weight[new_key] = value

    sliced_weights = []
    for rank in client_ranks:
        client_weights = copy.deepcopy(global_client_weight)
        for key, _ in client_weights.items():
            if key.endswith("lora_A"):
                client_weights[key] = readied_aggregated_lora_weight[key][
                    : int(rank * 2), :
                ]
            elif key.endswith("lora_B"):
                client_weights[key] = readied_aggregated_lora_weight[key][:, :rank]
        sliced_weights.append(client_weights)

    return sliced_weights


def aggregate(w_local_clients, num_layers=3, global_client_rank=16):
    aggregated_client_lora = {}

    for layer in range(num_layers):
        w_locals_layer_lora = []
        layer_norms = []

        for w_client in w_local_clients:
            layer_temp_dict = {}
            layer_norm = 0
            for key, value in w_client.items():
                if key.startswith(f"h.{layer}.") and (
                    key.endswith("lora_A") or key.endswith("lora_B")
                ):
                    layer_temp_dict[key] = value
                    if key.endswith("lora_A"):
                        A = value
                    elif key.endswith("lora_B"):
                        B = value
                        layer_norm += torch.norm(
                            B @ A.reshape(int(A.size(0) / 2), -1)
                        ).item()
            if layer_temp_dict:
                w_locals_layer_lora.append(layer_temp_dict)
                layer_norms.append(layer_norm)

        if not w_locals_layer_lora:
            continue

        padded_w_locals_layer_lora = pad_lora_weights(
            w_locals_layer_lora, global_client_rank
        )

        total_layer_norm = sum(layer_norms)
        layer_weights = (
            [norm / total_layer_norm for norm in layer_norms]
            if total_layer_norm != 0
            else [1.0 / len(layer_norms)] * len(layer_norms)
        )

        for idx, w_client_layer in enumerate(padded_w_locals_layer_lora):
            for key, value in w_client_layer.items():
                if key in aggregated_client_lora:
                    aggregated_client_lora[key] += layer_weights[idx] * value
                else:
                    aggregated_client_lora[key] = layer_weights[idx] * value

    return aggregated_client_lora


def distribute(global_client_weight, client_models, aggregated_client_lora):
    client_ranks = get_client_ranks(client_models)
    sliced_client_weights = slice_lora_for_clients(
        global_client_weight, aggregated_client_lora, client_ranks
    )
    return sliced_client_weights
