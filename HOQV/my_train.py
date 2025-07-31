#!/usr/bin/env python3
#
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#


import argparse
from sklearn.metrics import f1_score, accuracy_score
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from pytorch_pretrained_bert import BertAdam
import torch.nn.functional as F

from src.data.helpers import get_data_loaders
from src.models import get_model,ce_loss,get_projection_distribution, unified_UCE_loss,GDD_fusion_proj,get_GDD_distribution
from src.utils.logger import create_logger
from src.utils.utils import *


torch.cuda.empty_cache()
def get_args(parser):
    parser.add_argument("--batch_sz", type=int, default=128)
    parser.add_argument("--bert_model", type=str, default="/path/to/prebert")#, choices=["bert-base-uncased", "bert-large-uncased"])
    parser.add_argument("--data_path", type=str, default="/path/to/dataset")
    parser.add_argument("--drop_img_percent", type=float, default=0.0)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--embed_sz", type=int, default=300)
    parser.add_argument("--freeze_img", type=int, default=0)
    parser.add_argument("--freeze_txt", type=int, default=0)
    parser.add_argument("--glove_path", type=str, default="./datasets/glove_embeds/glove.840B.300d.txt")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=24)
    parser.add_argument("--hidden", nargs="*", type=int, default=[])
    parser.add_argument("--hidden_sz", type=int, default=768)
    parser.add_argument("--img_embed_pool_type", type=str, default="avg", choices=["max", "avg"])
    parser.add_argument("--img_hidden_sz", type=int, default=2048)
    parser.add_argument("--include_bn", type=int, default=True)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lr_factor", type=float, default=0.5)
    parser.add_argument("--lr_patience", type=int, default=2)
    parser.add_argument("--max_epochs", type=int, default=100)
    parser.add_argument("--max_seq_len", type=int, default=512)
    parser.add_argument("--model", type=str, default="HOQV")
    parser.add_argument("--n_workers", type=int, default=0)
    parser.add_argument("--name", type=str, default="nameless")
    parser.add_argument("--num_image_embeds", type=int, default=1)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--savedir", type=str, default="/path/to/save_dir/")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--task", type=str, default="mmimdb", choices=["mmimdb", "vsnli", "food101","MVSA_Single"])
    parser.add_argument("--task_type", type=str, default="multilabel", choices=["multilabel", "classification"])
    parser.add_argument("--warmup", type=float, default=0.1)
    parser.add_argument("--weight_classes", type=int, default=1)
    parser.add_argument("--df", type=bool, default=False)
    parser.add_argument("--noise", type=float, default=0.0)


def ensure_one_hot(tgt, num_classes):
    tgt = tgt.cuda()

    if tgt.dim() == 2 and tgt.size(1) == num_classes and torch.all((tgt.sum(dim=1) == 1)):
        return tgt
    if tgt.dim() == 1:
        return F.one_hot(tgt, num_classes=num_classes).float()

    raise ValueError("Unexpected tgt shape or format.")


def get_criterion(args):
    if args.task_type == "multilabel":
        if args.weight_classes:
            freqs = [args.label_freqs[l] for l in args.labels]
            label_weights = (torch.FloatTensor(freqs) / args.train_data_len) ** -1
            criterion = nn.BCEWithLogitsLoss(pos_weight=label_weights.cuda())
        else:
            criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = nn.CrossEntropyLoss()

    return criterion


def get_optimizer(model, args):
    if args.model in ["bert", "concatbert", "mmbt"]:
        total_steps = (
            args.train_data_len
            / args.batch_sz
            / args.gradient_accumulation_steps
            * args.max_epochs
        )
        param_optimizer = list(model.named_parameters())
        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {"params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], "weight_decay": 0.01},
            {"params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], "weight_decay": 0.0,},
        ]
        optimizer = BertAdam(
            optimizer_grouped_parameters,
            lr=args.lr,
            warmup=args.warmup,
            t_total=total_steps,
        )
    else:
        optimizer = optim.Adam(model.parameters(), lr=args.lr)

    return optimizer


def get_scheduler(optimizer, args):
    return optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, "max", patience=args.lr_patience, verbose=True, factor=args.lr_factor
    )

def model_eval(i_epoch, data, model, args, criterion, store_preds=False):
    with torch.no_grad():
        losses, preds, tgts = [], [], []
        i = 0
        for batch in data:
            out,out_t,out_i,tgt,loss = model_forward(i_epoch, model, args, criterion, batch,mode='eval')
            losses.append(loss)
            if args.task_type == "multilabel":
                pred = torch.sigmoid(out).cpu().detach().numpy() > 0.5
            else:
                pred = out.argmax(dim=1).cpu().detach().numpy()

            preds.append(pred)
            tgt = tgt.cpu().detach().numpy()
            tgts.append(tgt)
            i+=1
    metrics = {"loss": np.mean(losses)}
    if args.task_type == "multilabel":
        tgts = np.vstack(tgts)
        preds = np.vstack(preds)
        metrics["macro_f1"] = f1_score(tgts, preds, average="macro")
        metrics["micro_f1"] = f1_score(tgts, preds, average="micro")
    else:
        tgts = [l for sl in tgts for l in sl]
        preds = [l for sl in preds for l in sl]
        metrics["acc"] = accuracy_score(tgts, preds)

    if store_preds:
        store_preds_to_disk(tgts, preds, args)

    return metrics



def model_forward(i_epoch, model, args, criterion, batch,mode='train'):
    txt, segment, mask, img, tgt,idx = batch
    freeze_img = i_epoch < args.freeze_img
    freeze_txt = i_epoch < args.freeze_txt

    if args.model == "bow":
        txt = txt.cuda()
        out = model(txt)
    elif args.model == "img":
        img = img.cuda()
        out = model(img)
    elif args.model == "concatbow":
        txt, img = txt.cuda(), img.cuda()
        out = model(txt, img)
    elif args.model == "bert":
        txt, mask, segment = txt.cuda(), mask.cuda(), segment.cuda()
        out = model(txt, mask, segment)
    elif args.model == "concatbert":
        txt, img = txt.cuda(), img.cuda()
        mask, segment = mask.cuda(), segment.cuda()
        out = model(txt, mask, segment, img)

    elif args.model == "latefusion":

        txt, img = txt.cuda(), img.cuda()
        mask, segment = mask.cuda(), segment.cuda()
        out, txt_logits, img_logits, txt_conf, img_conf = model(txt, mask, segment, img)
    elif args.model == "tmc":
        txt, img = txt.cuda(), img.cuda()
        mask, segment = mask.cuda(), segment.cuda()
        txt_alpha, img_alpha, txt_img_alpha = model(txt, mask, segment, img)
    elif args.model == 'HOQV':
        txt,img = txt.cuda(),img.cuda()
        mask,segment = mask.cuda(),segment.cuda()
        if mode == 'train':
            txt_feature,evidence_txt,img_feature,evidence_img,txt_comp,img_comp = model(txt,mask,segment,img,tgt.cuda())
        else:
            txt_feature, evidence_txt, img_feature, evidence_img, txt_comp, img_comp = model(txt, mask, segment, img)

    else:
        assert args.model == "mmbt"
        for param in model.enc.img_encoder.parameters():
            param.requires_grad = not freeze_img
        for param in model.enc.encoder.parameters():
            param.requires_grad = not freeze_txt
        txt, img = txt.cuda(), img.cuda()
        mask, segment = mask.cuda(), segment.cuda()
        out = model(txt, mask, segment, img)

    tgt = tgt.cuda()
    if mode == 'train':
        beta_txt,belief_txt = get_GDD_distribution(evidence_txt,args.n_classes)
        beta_img,belief_img = get_GDD_distribution(evidence_img,args.n_classes)

        t_b,t_e,t_a,t_u = GDD_fusion_proj(belief_txt,evidence_txt,args.n_classes,txt_comp)
        i_b,i_e,i_a,i_u = GDD_fusion_proj(belief_img,evidence_img,args.n_classes,img_comp)
        f_a,f_b,f_u,f_s = get_projection_distribution(t_e+1,i_e+1,args.n_classes)

        loss_fuse = ce_loss(tgt,f_a,args.n_classes,i_epoch,args.annealing_epoch)


        loss_gdd_txt,entro_txt = unified_UCE_loss(evidence_txt,tgt,txt_comp,args.n_classes,kl_lam_GDD=1,entropy_lam_Dir=0,entropy_lam_GDD=1,device=tgt.device)
        loss_gdd_img,entro_img = unified_UCE_loss(evidence_img, tgt, img_comp,args.n_classes, kl_lam_GDD=1, entropy_lam_Dir=0, entropy_lam_GDD=1,
                                        device=tgt.device)

        entro_txt = entro_txt.detach()
        entro_img = entro_img.detach()




        pred = f_b
        w_txt = 0.5*torch.exp(-entro_txt)
        w_img = 0.5*torch.exp(-entro_img)

        loss = loss_gdd_txt + loss_gdd_img + loss_fuse
        return loss, pred, tgt, t_b, i_b, w_txt, w_img
    else:
        loss = 0
        evidence_txt = evidence_txt[:,:args.n_classes]
        evidence_img = evidence_img[:,:args.n_classes]
        f_a,f_b,f_u,f_s = get_projection_distribution(evidence_txt+1,evidence_img+1,args.n_classes)
        pred = f_a
        return pred,evidence_txt,evidence_img,tgt,loss


def train(args):

    set_seed(args.seed)

    args.savedir = os.path.join(args.savedir, args.name)

    os.makedirs(args.savedir, exist_ok=True)

    train_loader, val_loader, test_loaders = get_data_loaders(args)


    model = get_model(args)

    criterion = get_criterion(args)
    optimizer = get_optimizer(model, args)
    scheduler = get_scheduler(optimizer, args)

    logger = create_logger("%s/logfile.log" % args.savedir, args)
    logger.info(model)
    model.cuda()


    torch.save(args, os.path.join(args.savedir, "args.pt"))

    start_epoch, global_step, n_no_improve, best_metric = 0, 0, 0, -np.inf



    logger.info("Training..")

    for i_epoch in range(start_epoch, args.max_epochs):
        train_losses = []
        model.train()
        optimizer.zero_grad()

        for batch in tqdm(train_loader, total=len(train_loader)):
            loss, _, _,belief_txt,belief_img,e_txt,e_img = model_forward(i_epoch, model, args, criterion, batch)
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            train_losses.append(loss.item())
            loss.backward()
            global_step += 1
            if i_epoch < 300:
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        belief_txt = torch.mean(belief_txt,dim=0)
                        belief_img = torch.mean(belief_img,dim=0)
                        if 'txt_linear' in name:
                            if len(param.grad.size()) > 1:
                                param.grad *=5 * belief_txt.unsqueeze(-1)
                            else:
                                param.grad *= 5 * belief_txt
                        elif 'txtclf' in name:
                            param.grad *= e_txt /2
                        elif 'img_linear' in name:
                            if len(param.grad.size()) > 1:
                                param.grad *= 5 * belief_img.unsqueeze(-1)
                            else:
                                param.grad *= 5 * belief_img
                        elif 'imgclf' in name:
                            param.grad *= e_img /2
            if global_step % args.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

        model.eval()

        metrics = model_eval(i_epoch, val_loader, model, args, criterion)
        logger.info("Train Loss: {:.4f}".format(np.mean(train_losses)))
        log_metrics("Val", metrics, args, logger)

        tuning_metric = (
            metrics["micro_f1"] if args.task_type == "multilabel" else metrics["acc"]
        )
        scheduler.step(tuning_metric)
        is_improvement = tuning_metric > best_metric
        if is_improvement:
            best_metric = tuning_metric
            n_no_improve = 0
        else:
            n_no_improve += 1

        save_checkpoint(
            {
                "epoch": i_epoch + 1,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "n_no_improve": n_no_improve,
                "best_metric": best_metric,
            },
            is_improvement,
            args.savedir,
        )

        if n_no_improve >= args.patience:
            logger.info("No improvement. Breaking out of loop.")
            break

    load_checkpoint(model, os.path.join(args.savedir, "model_best.pt"))
    model.eval()
    for test_name, test_loader in test_loaders.items():
        test_metrics = model_eval(
            np.inf, test_loader, model, args, criterion, store_preds=True
        )
        log_metrics(f"Test - {test_name}", test_metrics, args, logger)


def cli_main():
    parser = argparse.ArgumentParser(description="Train Models")
    get_args(parser)

    args, remaining_args = parser.parse_known_args()


    args.annealing_epoch=10
    if args.task=='food101':
        args.n_classes=101
    elif args.task=='mmimdb':
        args.n_classes=23
    elif args.task=="MVSA_Single":
        args.n_classes = 3
    else:
        print('no implemented yet')
        exit(1)
    assert remaining_args == [], remaining_args
    train(args)


if __name__ == "__main__":
    import warnings

    warnings.filterwarnings("ignore")

    cli_main()
