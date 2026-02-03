import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.functional import retrieval_average_precision
import pytorch_lightning as pl

from src.clip import clip
from experiments.options import opts

def freeze_model(m):
    m.requires_grad_(False)

def freeze_all_but_bn(m):
    if not isinstance(m, torch.nn.LayerNorm):
        if hasattr(m, 'weight') and m.weight is not None:
            m.weight.requires_grad_(False)
        if hasattr(m, 'bias') and m.bias is not None:
            m.bias.requires_grad_(False)

class Model(pl.LightningModule):
    def __init__(self):
        super().__init__()

        self.opts = opts
        self.clip, _ = clip.load('ViT-B/32', device=self.device)
        self.clip.apply(freeze_all_but_bn)

        # Prompt Engineering
        self.sk_prompt = nn.Parameter(torch.randn(self.opts.n_prompts, self.opts.prompt_dim))
        self.img_prompt = nn.Parameter(torch.randn(self.opts.n_prompts, self.opts.prompt_dim))

        self.distance_fn = lambda x, y: 1.0 - F.cosine_similarity(x, y)
        self.loss_fn = nn.TripletMarginWithDistanceLoss(
            distance_function=self.distance_fn, margin=0.2)

        # Optional auxiliary classification loss.
        self.cls_loss_weight = float(getattr(self.opts, 'cls_loss_weight', 0.0))
        self.ce_loss = nn.CrossEntropyLoss()

        embed_dim = getattr(getattr(self.clip, 'visual', None), 'output_dim', None)
        if embed_dim is None:
            embed_dim = int(getattr(getattr(self.clip, 'text_projection', None), 'shape', [None, 0])[1])

        # nclass is often not set correctly via CLI; keep a safe fallback.
        if not hasattr(self.opts, 'nclass') or int(getattr(self.opts, 'nclass', 0)) <= 0:
            self.opts.nclass = 1

        self.cls_head = nn.Linear(embed_dim, int(self.opts.nclass))

        self.best_metric = -1e3

    def configure_optimizers(self):
        optimizer = torch.optim.Adam([
            {'params': self.clip.parameters(), 'lr': self.opts.clip_LN_lr},
            {'params': [self.sk_prompt] + [self.img_prompt], 'lr': self.opts.prompt_lr},
            {'params': self.cls_head.parameters(), 'lr': self.opts.linear_lr}])
        return optimizer

    def _maybe_get_label_idx(self, batch):
        # Dataset appends label_idx at the end when cls_loss_weight > 0.
        if len(batch) < 6:
            return None
        label_idx = batch[-1]
        if isinstance(label_idx, (list, tuple)):
            label_idx = torch.as_tensor(label_idx, device=self.device)
        if isinstance(label_idx, torch.Tensor):
            return label_idx.long().to(self.device)
        try:
            return torch.as_tensor(label_idx, device=self.device).long()
        except Exception:
            return None

    def _classification_loss(self, feat: torch.Tensor, label_idx: torch.Tensor):
        logits = self.cls_head(feat.float())
        return self.ce_loss(logits, label_idx)

    def forward(self, data, dtype='image'):
        if dtype == 'image':
            feat = self.clip.encode_image(
                data, self.img_prompt.expand(data.shape[0], -1, -1))
        else:
            feat = self.clip.encode_image(
                data, self.sk_prompt.expand(data.shape[0], -1, -1))
        return feat

    def training_step(self, batch, batch_idx):
        sk_tensor, img_tensor, neg_tensor, category = batch[:4]
        img_feat = self.forward(img_tensor, dtype='image')
        sk_feat = self.forward(sk_tensor, dtype='sketch')
        neg_feat = self.forward(neg_tensor, dtype='image')

        triplet_loss = self.loss_fn(sk_feat, img_feat, neg_feat)
        loss = triplet_loss

        if self.cls_loss_weight > 0.0:
            label_idx = self._maybe_get_label_idx(batch)
            if label_idx is not None:
                cls_loss = 0.5 * (
                    self._classification_loss(sk_feat, label_idx)
                    + self._classification_loss(img_feat, label_idx)
                )
                loss = loss + self.cls_loss_weight * cls_loss
                self.log('train_cls_loss', cls_loss, prog_bar=True)

        self.log('train_triplet_loss', triplet_loss)
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        sk_tensor, img_tensor, neg_tensor, category = batch[:4]
        img_feat = self.forward(img_tensor, dtype='image')
        sk_feat = self.forward(sk_tensor, dtype='sketch')
        neg_feat = self.forward(neg_tensor, dtype='image')

        triplet_loss = self.loss_fn(sk_feat, img_feat, neg_feat)
        loss = triplet_loss

        if self.cls_loss_weight > 0.0:
            label_idx = self._maybe_get_label_idx(batch)
            if label_idx is not None:
                cls_loss = 0.5 * (
                    self._classification_loss(sk_feat, label_idx)
                    + self._classification_loss(img_feat, label_idx)
                )
                loss = loss + self.cls_loss_weight * cls_loss
                self.log('val_cls_loss', cls_loss)

        self.log('val_triplet_loss', triplet_loss)
        self.log('val_loss', loss)
        return sk_feat, img_feat, category

    def validation_epoch_end(self, val_step_outputs):
        Len = len(val_step_outputs)
        if Len == 0:
            return
        query_feat_all = torch.cat([val_step_outputs[i][0] for i in range(Len)])
        gallery_feat_all = torch.cat([val_step_outputs[i][1] for i in range(Len)])
        all_category = np.array(sum([list(val_step_outputs[i][2]) for i in range(Len)], []))


        ## mAP category-level SBIR Metrics
        gallery = gallery_feat_all
        ap = torch.zeros(len(query_feat_all))
        for idx, sk_feat in enumerate(query_feat_all):
            category = all_category[idx]
            distance = -1*self.distance_fn(sk_feat.unsqueeze(0), gallery)
            target = torch.zeros(len(gallery), dtype=torch.bool)
            target[np.where(all_category == category)] = True
            ap[idx] = retrieval_average_precision(distance.cpu(), target.cpu())
        
        mAP = torch.mean(ap)
        self.log('mAP', mAP)
        if self.global_step > 0:
            self.best_metric = self.best_metric if  (self.best_metric > mAP.item()) else mAP.item()
        print ('mAP: {}, Best mAP: {}'.format(mAP.item(), self.best_metric))
