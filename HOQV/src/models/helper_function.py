import torch
from torch.utils.data import DataLoader, TensorDataset, Dataset
import matplotlib.pyplot as plt
from sklearn import metrics
from copy import deepcopy
import numpy as np
from .GDD import GroupDirichlet, find_partition
def confidence_interval(accuracies):
    accuracies = np.array(accuracies)
    stds = np.std(accuracies, 0)
    ci95 = 1.96 * stds / np.sqrt(len(accuracies))
    return stds, ci95


def one_hot_embedding(labels, num_classes=10, device='cpu'):
    y = torch.eye(num_classes, device=device)
    return y[labels]


def multi_hot_embedding(label, R, num_classes=10, device='cpu'):
    y = torch.eye(num_classes, device=device)
    composite_labels = R[label]
    return y[composite_labels].sum(dim=0)


def multi_hot_embedding_batch(labels, R, num_single, device='cpu'):
    res = []
    if labels.dim() == 0:
        labels = labels.unsqueeze(dim=0) # compatible with batch_size=1
    for label in labels:
        res.append(multi_hot_embedding(label, R, num_single, device))
    return torch.stack(res, dim=0)


def complete_comp_labels(input_list, set_list):
    output = []
    for item in input_list:
        found_set = False
        for set_comp in set_list:
            if item in set_comp:
                output.append(set_comp)
                found_set = True
                break
        if not found_set:
            output.append([item])
    return output


def pseudo_target(pseudo_comp, R, num_single, device='cpu'):
    res = []
    comp_set = R[num_single:]
    for pseudo in pseudo_comp:
        if pseudo in comp_set:
            res.append(comp_set.index(pseudo) + num_single)
        else:
            res.append(pseudo[0])
    return torch.tensor(res, dtype=torch.long, device=device)

def kl_divergence(alpha, num_classes, device=None):
    ones = torch.ones([1, num_classes], dtype=torch.float32, device=device)
    sum_alpha = torch.sum(alpha, dim=1, keepdim=True)
    first_term = (
        torch.lgamma(sum_alpha)
        - torch.lgamma(alpha).sum(dim=1, keepdim=True)
        + torch.lgamma(ones).sum(dim=1, keepdim=True)
        - torch.lgamma(ones.sum(dim=1, keepdim=True))
    )
    second_term = (
        (alpha - ones)
        .mul(torch.digamma(alpha) - torch.digamma(sum_alpha))
        .sum(dim=1, keepdim=True)
    )
    kl = first_term + second_term
    return kl


def kl_GDD(alpha, evidence_comps, num_single, R, targets, device=None):
    one_hot_embed = one_hot_embedding(targets, num_classes=len(R), device=device)
    one_hot_embed_cut = one_hot_embed[:, :num_single]
    alpha = (alpha - 1) * (1 - one_hot_embed_cut) + 1
    one_hot_comp_cut = one_hot_embed[:, num_single:]
    evidence_comps = evidence_comps * (1 - one_hot_comp_cut)
    unique_comp_sets = R[num_single:]
    uniques_elements_comp = sum(unique_comp_sets, [])
    comp_rest_labels = list(set(range(num_single)) - set(uniques_elements_comp))  # [0,2,4,5]
    unique_comp_sets.append(comp_rest_labels)
    pading = torch.zeros(len(targets), 1, device=device, requires_grad=True)
    evidence_comps_custom = torch.cat([evidence_comps, pading], dim=1)
    one_GDD = GroupDirichlet(alpha, evidence_comps_custom, unique_comp_sets)

    alpha_one = torch.ones_like(alpha)
    evidence_comps_one = torch.zeros_like(evidence_comps_custom)
    GDD_tmp = GroupDirichlet(alpha_one, evidence_comps_one, unique_comp_sets)
    log_Cg_tmp = GDD_tmp.log_normalized_constant()

    log_Cg = one_GDD.log_normalized_constant()
    num_singles = one_GDD.concentration_a.size(-1)
    a0 = one_GDD.concentration_a.sum(-1)
    term2 = ((one_GDD.concentration_a - 1.0) * (torch.digamma(one_GDD.concentration_a))).sum(-1)

    beta = []
    partition_indicator = []
    for j in range(one_GDD.n_partition):
        part_idx_curr = one_GDD.partition_list[j]
        alpha_j = torch.sum(one_GDD.concentration_a[:, part_idx_curr], dim=1)
        beta_j = alpha_j + one_GDD.concentration_b[:, j]
        beta.append(beta_j)
        diff = torch.digamma(beta_j) - torch.digamma(alpha_j)
        partition_indicator.append(diff)
    beta_tensor = torch.stack(beta, dim=1)
    beta0 = beta_tensor.sum(-1)
    term3 = (a0 - num_singles) * torch.digamma(beta0)

    term4 = 0.
    for k in range(num_singles):
        partition_id = find_partition(one_GDD.partition_list, k)
        term4 += partition_indicator[partition_id] * (one_GDD.concentration_a[:, k] - 1)

    term5 = torch.sum(one_GDD.concentration_b * torch.digamma(beta_tensor), dim=1)
    term6 = torch.digamma(beta0) * torch.sum(one_GDD.concentration_b, dim=1)

    kl_gdd = log_Cg - log_Cg_tmp - term2 + term3 - term4 - term5 + term6
    return -kl_gdd.mean()