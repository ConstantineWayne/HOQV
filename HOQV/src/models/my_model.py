import random
import torch
import torch.nn as nn
from sklearn.metrics.pairwise import cosine_similarity
from .bert import BertEncoder,BertClf
from .image import ImageEncoder,ImageClf
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from collections import defaultdict
from .helper_function import pseudo_target,complete_comp_labels,multi_hot_embedding_batch,multi_hot_embedding,kl_divergence,kl_GDD,one_hot_embedding
from .GDD import GroupDirichlet
from torch.distributions.dirichlet import Dirichlet
from sklearn.metrics.pairwise import cosine_similarity
import torch.nn.functional as F
def build_composite_sets_from_features(
        features: torch.Tensor,
        labels: torch.Tensor,
        num_classes: int,
        num_composites: int = 2,
        sim_threshold: float = 0.5,
):

    features_np = features.detach().cpu().numpy()
    if labels is not None:
        labels_np = labels.detach().cpu().numpy()
        class_centers = []
        for c in range(num_classes):
            cls_feats = features_np[labels_np == c]
            if len(cls_feats) == 0:
                center = np.zeros(features_np.shape[1])
            else:
                center = np.mean(cls_feats, axis=0)
            class_centers.append(center)
        class_centers = np.stack(class_centers, axis=0)  # [C, D]

        sim_matrix = cosine_similarity(class_centers)  # [C, C]
        np.fill_diagonal(sim_matrix, 0)
        sim_matrix_filtered = np.where(sim_matrix >= sim_threshold, sim_matrix, 0.0)

        dist_matrix = 1.0 - sim_matrix_filtered
        clustering = AgglomerativeClustering(
            affinity='precomputed',
            linkage='average'
        )
        cluster_labels = clustering.fit_predict(dist_matrix)  # [C]

        from collections import defaultdict
        composite_sets_dict = defaultdict(list)
        for class_id, comp_id in enumerate(cluster_labels):
            composite_sets_dict[comp_id].append(class_id)
        composite_sets = [v for v in composite_sets_dict.values() if len(v) > 1]

        if len(composite_sets) == 0:
            all_classes = list(range(num_classes))
            random.shuffle(all_classes)
            chunk_size = max(2, num_classes // num_composites)
            for i in range(0, num_classes, chunk_size):
                subset = all_classes[i:i + chunk_size]
                if len(subset) >= 2:
                    composite_sets.append(subset)

        if len(composite_sets) < num_composites:
            all_classes = [cls for subset in composite_sets for cls in subset]
            random.shuffle(all_classes)
            chunk_size = max(1, len(all_classes) // num_composites)

            new_composite_sets = []
            for i in range(0, len(all_classes), chunk_size):
                subset = all_classes[i:i + chunk_size]
                if len(subset) > 0:
                    new_composite_sets.append(subset)

            if len(new_composite_sets) > num_composites:
                new_composite_sets = new_composite_sets[:num_composites]
            elif len(new_composite_sets) < num_composites:
                while len(new_composite_sets) < num_composites:
                    new_composite_sets[-1].append(all_classes.pop())

            composite_sets = new_composite_sets
    else:
        composite_sets = None
    return composite_sets

def build_R_from_composites(composite_sets, num_single):
    R = [[i] for i in range(num_single)]
    R.extend(composite_sets)
    return R
class HOQV(nn.Module):
    def __init__(self, args):
        super(HOQV, self).__init__()
        self.args = args

        self.txtclf = BertClf(args)
        self.imgclf= ImageClf(args)
        self.txt_linear = nn.Linear(self.args.hidden_sz,self.args.n_classes + 1)
        self.txt_linear.apply(self.txtclf.enc.bert.init_bert_weights)

        self.img_linear = nn.Linear(self.args.img_hidden_sz * self.args.num_image_embeds, args.n_classes+1)

    def forward(self, txt, mask, segment, img, tgt=None):
        txt_feature,_ = self.txtclf(txt, mask, segment)
        img_feature,_ = self.imgclf(img)
        if tgt is not None:
            txt_comp = build_composite_sets_from_features(txt_feature,tgt,self.args.n_classes,num_composites=1)
            img_comp = build_composite_sets_from_features(img_feature,tgt,self.args.n_classes,num_composites=1)
        else:
            txt_comp = None
            img_comp = None
        txt_logits = self.txt_linear(txt_feature)
        img_logits = self.img_linear(img_feature)
        e_txt = nn.Softplus()(txt_logits)
        e_img = nn.Softplus()(img_logits)
        return txt_feature,e_txt,img_feature,e_img,txt_comp,img_comp

    def get_weight(self):
        txt_clf_w = self.txtclf.clf.weight
        img_clf_w = self.imgclf.clf.weight
        return txt_clf_w,img_clf_w

def get_GDD_distribution(evidence,n_classes):
    e_s = evidence[:,:n_classes]
    e_c = evidence[:,n_classes:]
    alpha_s = e_s + 1
    concentration = torch.cat([alpha_s,e_c],dim=1)
    beta = torch.sum(concentration,dim=1,keepdim=True)
    belief = evidence / (beta.expand(evidence.shape))
    uncertainty = evidence.size(1) / beta
    return beta,belief

def GDD_fusion_proj(belief,evidence, n_classes, composite):
    b_s = belief[:, :n_classes].clone()
    b_c = belief[:, n_classes:]
    for i, comp in enumerate(composite):
        b_comp = b_c[:, i].unsqueeze(1) / len(comp)
        for c in comp:
            b_s[:, c] += b_comp.squeeze(1)

    e_s = evidence[:, :n_classes].clone()
    e_c = evidence[:, n_classes:]

    for i, comp in enumerate(composite):
        e_comp = e_c[:, i].unsqueeze(1) / len(comp)
        for c in comp:
            e_s[:, c] += e_comp.squeeze(1)

    alpha = e_s + 1
    s = torch.sum(alpha,dim=1,keepdim=True)
    uncertainty = alpha.size(1) / s
    return b_s,e_s,alpha,uncertainty


def edl_singl_comp_loss(
        func,
        targets,
        R,
        alpha,
        evidence_comps,
        num_single,
        kl_reg=True,
        device=None
):
    if targets.dim() == 0:
        targets = targets.unsqueeze(dim=0)
    concentration = torch.cat([alpha, evidence_comps], dim=1)
    beta_sum = torch.sum(concentration, dim=1, keepdim=True)

    multi_hot_embed = multi_hot_embedding_batch(targets, R, num_single, device=device)
    padding = torch.zeros(len(targets), len(R) - num_single, device=device, requires_grad=True)
    multi_hot_pad_zero = torch.cat([multi_hot_embed, padding], dim=1)
    one_hot_embed = one_hot_embedding(targets, num_classes=len(R), device=device)
    scalar_indicator_comp = targets >= num_single
    mixed_hot_embed = multi_hot_pad_zero + one_hot_embed * scalar_indicator_comp[:, None]
    beta_gt_sum = torch.sum(mixed_hot_embed * concentration, dim=1, keepdim=True)
    uce_comp = func(beta_sum) - func(beta_gt_sum)

    alpha_gt_sum = torch.sum(multi_hot_pad_zero * concentration, dim=1, keepdim=True)
    uce_term21 = func(beta_sum) - func(alpha_gt_sum)
    R_comp = complete_comp_labels(targets, R[num_single:])
    targets_pseudo = pseudo_target(R_comp, R, num_single, device=device)
    mul_hot_embed_comp = multi_hot_embedding_batch(targets_pseudo, R, num_single, device=device)
    mul_hot_comp_pad_zero = torch.cat([mul_hot_embed_comp, padding], dim=1)
    one_hot_pseudo = one_hot_embedding(targets_pseudo, num_classes=len(R), device=device)
    scalar_indicator_comp_pseudo = targets_pseudo >= num_single
    mixed_hot_comp_pseudo = mul_hot_comp_pad_zero + one_hot_pseudo * scalar_indicator_comp_pseudo[:, None]
    beta_gt_sum_pseudo = torch.sum(mixed_hot_comp_pseudo * concentration, dim=1, keepdim=True)
    alpha_gt_sum_pseudo = torch.sum(mul_hot_comp_pad_zero * concentration, dim=1, keepdim=True)
    uce_term22 = func(beta_gt_sum_pseudo) - func(alpha_gt_sum_pseudo)
    uce_single = uce_term21 - uce_term22

    scalar_indicator_singl = targets < num_single
    uce_batch = uce_comp * scalar_indicator_comp[:, None] + uce_single * scalar_indicator_singl[:, None]
    uce_mean = torch.mean(uce_batch)

    if not kl_reg:
        return uce_mean, torch.tensor(0.).cuda()
    kl_alpha = alpha
    kl_term = kl_divergence(kl_alpha, num_single, device=device)
    kl_mean = torch.mean(kl_term)
    return uce_mean, kl_mean


def ce_loss(p, alpha, c, global_step, annealing_step):
    S = torch.sum(alpha, dim=1, keepdim=True)
    E = alpha - 1

    label = F.one_hot(p, num_classes=c)

    A = torch.sum(label * (torch.digamma(S) - torch.digamma(alpha)), dim=1, keepdim=True)

    annealing_coef = min(1, global_step / annealing_step)
    alp = E * (1 - label) + 1
    B = annealing_coef * KL(alp, c)
    return torch.mean((A + B))

def unified_UCE_loss(
        evidence,
        targets,
        comps,
        num_single,
        kl_lam_GDD,
        entropy_lam_Dir,
        entropy_lam_GDD,
        anneal=False,
        kl_reg=False,
        device=None
):
    if not anneal:
        assert anneal == False

    if targets.dim() == 0:
        targets = targets.unsqueeze(dim=0)  # compatible with batch_size=1

    if evidence.dim() == 1:
        evidence = evidence.unsqueeze(dim=0)
    R = build_R_from_composites(comps, num_single)
    evidence_single = evidence[:, :num_single]
    alpha = evidence_single + 1
    evidence_comps = evidence[:, num_single:]

    uce_mean, kl_mean = edl_singl_comp_loss(
        torch.digamma,
        targets,
        R,
        alpha,
        evidence_comps,
        num_single,
        kl_reg=kl_reg,
        device=device
    )

    if entropy_lam_Dir:
        entropy = Dirichlet(evidence + 1).entropy().mean()
    else:
        entropy = torch.tensor(0.).cuda()

    # Entropy of GDD
    if entropy_lam_GDD:
        unique_comp_sets = R[num_single:]
        uniques_elements_comp = sum(unique_comp_sets, [])
        comp_rest_labels = list(set(range(num_single)) - set(uniques_elements_comp))
        unique_comp_sets.append(comp_rest_labels)
        pading = torch.zeros(len(targets), 1, device=device, requires_grad=True)
        evidence_comps_custom = torch.cat([evidence_comps, pading], dim=1)
        entropy_GDD = GroupDirichlet(alpha, evidence_comps_custom, unique_comp_sets).entropy().mean()
    else:
        entropy_GDD = torch.tensor(0.).cuda()

    if kl_lam_GDD:
        kl_gdd = kl_GDD(alpha, evidence_comps, num_single, R, targets, device=device)
    else:
        kl_gdd = torch.tensor(0.).cuda()

    loss = uce_mean - entropy_lam_Dir * entropy - entropy_lam_GDD * entropy_GDD + kl_lam_GDD * kl_gdd

    return loss,entropy_GDD

def get_projection_distribution(a1,a2,n_classes):
    s1 = torch.sum(a1,dim=1,keepdim=True)
    s2 = torch.sum(a2,dim=1,keepdim=True)
    e1 = a1 - 1
    e2 = a2-1

    b1 = e1 / (s1.expand(e1.shape))
    b2 = e2 / (s2.expand(e2.shape))
    u1 = n_classes / s1
    u2 = n_classes / s2
    b = [b1,b2]
    u = [u1,u2]
    bb = torch.bmm(b[0].view(-1, n_classes, 1), b[1].view(-1, 1, n_classes))
    # b^0 * u^1
    uv1_expand = u[1].expand(b[0].shape)
    bu = torch.mul(b[0], uv1_expand)
    # b^1 * u^0
    uv_expand = u[0].expand(b[0].shape)
    ub = torch.mul(b[1], uv_expand)
    # calculate K
    bb_sum = torch.sum(bb, dim=(1, 2), out=None)
    bb_diag = torch.diagonal(bb, dim1=-2, dim2=-1).sum(-1)
    # bb_diag1 = torch.diag(torch.mm(b[v], torch.transpose(b[v+1], 0, 1)))
    K = bb_sum - bb_diag

    # calculate b^a
    b_a = (torch.mul(b[0], b[1]) + bu + ub) / ((1 - K).view(-1, 1).expand(b[0].shape))
    # calculate u^a
    u_a = torch.mul(u[0], u[1]) / ((1 - K).view(-1, 1).expand(u[0].shape))
    # test = torch.sum(b_a, dim = 1, keepdim = True) + u_a #Verify programming errors

    # calculate new S
    S_a = n_classes / u_a
    # calculate new e_k
    e_a = torch.mul(b_a, S_a.expand(b_a.shape))
    alpha_a = e_a + 1
    return alpha_a,b_a,u_a,S_a

def get_prediction_projection(belief):
    return torch.argmax(belief)

def KL(alpha,c):
    beta = torch.ones((1, c)).cuda()
    S_alpha = torch.sum(alpha, dim=1, keepdim=True)
    S_beta = torch.sum(beta, dim=1, keepdim=True)
    lnB = torch.lgamma(S_alpha) - torch.sum(torch.lgamma(alpha), dim=1, keepdim=True)
    lnB_uni = torch.sum(torch.lgamma(beta), dim=1, keepdim=True) - torch.lgamma(S_beta)
    dg0 = torch.digamma(S_alpha)
    dg1 = torch.digamma(alpha)
    kl = torch.sum((alpha - beta) * (dg1 - dg0), dim=1, keepdim=True) + lnB + lnB_uni
    return kl












