#more imports
from trainers.graph_utils import (
    build_class_graph,
    graph_label_smooth,
    graph_laplacian_loss
)
from trainers.gcn import ClassGCN, normalize_adj
import torch.nn.functional as F

import os.path as osp

import torch
import torch.nn as nn
from torch.nn import functional as F

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.metrics import compute_accuracy
from dassl.utils import load_pretrained_weights, load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

_tokenizer = _Tokenizer()


def load_clip_to_cpu(cfg):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(state_dict or model.state_dict())

    return model


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x


class PromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = cfg.TRAINER.COOP.N_CTX
        ctx_init = cfg.TRAINER.COOP.CTX_INIT
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.INPUT.SIZE[0]
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        if ctx_init:
            # use given words to initialize context vectors
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1 : 1 + n_ctx, :]
            prompt_prefix = ctx_init

        else:
            # random initialization
            if cfg.TRAINER.COOP.CSC:
                print("Initializing class-specific contexts")
                ctx_vectors = torch.empty(n_cls, n_ctx, ctx_dim, dtype=dtype)
            else:
                print("Initializing a generic context")
                ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")

        self.ctx = nn.Parameter(ctx_vectors)  # to be optimized

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx :, :])  # CLS, EOS

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens
        self.class_token_position = cfg.TRAINER.COOP.CLASS_TOKEN_POSITION

    def forward(self):
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        prefix = self.token_prefix
        suffix = self.token_suffix

        if self.class_token_position == "end":
            prompts = torch.cat(
                [
                    prefix,  # (n_cls, 1, dim)
                    ctx,     # (n_cls, n_ctx, dim)
                    suffix,  # (n_cls, *, dim)
                ],
                dim=1,
            )

        elif self.class_token_position == "middle":
            half_n_ctx = self.n_ctx // 2
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i_half1 = ctx[i : i + 1, :half_n_ctx, :]
                ctx_i_half2 = ctx[i : i + 1, half_n_ctx:, :]
                prompt = torch.cat(
                    [
                        prefix_i,     # (1, 1, dim)
                        ctx_i_half1,  # (1, n_ctx//2, dim)
                        class_i,      # (1, name_len, dim)
                        ctx_i_half2,  # (1, n_ctx//2, dim)
                        suffix_i,     # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        elif self.class_token_position == "front":
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i = ctx[i : i + 1, :, :]
                prompt = torch.cat(
                    [
                        prefix_i,  # (1, 1, dim)
                        class_i,   # (1, name_len, dim)
                        ctx_i,     # (1, n_ctx, dim)
                        suffix_i,  # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        else:
            raise ValueError

        return prompts


class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.prompt_learner = PromptLearner(cfg, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        self.num_classes = len(classnames)

    def forward(self, image):
        image_features = self.image_encoder(image.type(self.dtype))

        prompts = self.prompt_learner()
        tokenized_prompts = self.tokenized_prompts
        text_features = self.text_encoder(prompts, tokenized_prompts)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()

        return logits, text_features

@TRAINER_REGISTRY.register()
class CoOpGraph(TrainerX):                  # rename the class

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model, _ = clip.load(cfg.MODEL.BACKBONE.NAME, device=self.device)
        clip_model = clip_model.float()

        # ── GRAPH SETUP ───────────────────────────────────────────────────────
        print("Building graph from class names...")
        self.A, self.A_norm, self.L = build_class_graph(
            classnames,
            clip_model,
            self.device,
            threshold=cfg.TRAINER.COOP.GRAPH_THRESHOLD
        )
        print(f"Graph built: {self.A.sum().item():.0f} edges "
              f"among {len(classnames)} classes")

        # ── ADD GCN HERE, right after graph is built ──────────────────────────
        self.A_hat = normalize_adj(self.A)              # precompute normalized adjacency

        feat_dim = clip_model.text_projection.shape[1]  # 512 for RN50
        self.use_gcn = cfg.TRAINER.COOP.USE_GCN
        self.gcn = ClassGCN(feat_dim=feat_dim, hidden_dim=256).to(self.device)
        # ── END GCN BLOCK ─────────────────────────────────────────────────────

        print("Building custom CLIP model")
        self.model = CustomCLIP(cfg, classnames, clip_model)

        print("Turning off gradients in model except prompt learner")
        for name, param in self.model.named_parameters():
            if "prompt_learner" not in name:
                param.requires_grad_(False)

        self.model.to(self.device)

        # Hyperparameters
        self.lambda_lap   = cfg.TRAINER.COOP.LAMBDA_LAP
        self.alpha_smooth = cfg.TRAINER.COOP.ALPHA_SMOOTH

        # ── CHANGED: optimizer now covers both prompt learner AND GCN ─────────
        trainable_params = list(self.model.prompt_learner.parameters())
        if self.use_gcn:
            trainable_params += list(self.gcn.parameters())

        self.optim = build_optimizer(trainable_params, cfg.OPTIM)
        # ── END CHANGE ────────────────────────────────────────────────────────

        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model(
            "prompt_learner", self.model.prompt_learner,
            self.optim, self.sched
        )

    def parse_batch_train(self, batch):
        input = batch["img"]
        label = batch["label"]
        input = input.to(self.device)
        label = label.to(self.device)
        return input, label

    def forward_backward(self, batch):
        image, label = self.parse_batch_train(batch)

        # ── STEP 1: Image features ────────────────────────────────────────────────
        image_features = self.model.image_encoder(image.type(self.model.dtype))
        image_features = F.normalize(image_features.float(), dim=-1)  # (B, D)

        # ── STEP 2: Text features from CoOp prompt learner ────────────────────────
        prompts = self.model.prompt_learner()
        text_features = self.model.text_encoder(
            prompts, self.model.tokenized_prompts
        )
        text_features = F.normalize(text_features.float(), dim=-1)    # (K, D)

        # ── STEP 3: GCN refinement (Idea 3) ──────────────────────────────────────
        # Runs only if USE_GCN: True in config
        # Refines text_features by aggregating info from neighboring classes
        if self.use_gcn:
            text_features = self.gcn(text_features, self.A_hat)       # (K, D)
            # gcn() already applies F.normalize internally via residual + normalize

        # ── STEP 4: Compute logits ────────────────────────────────────────────────
        logit_scale = self.model.logit_scale.exp()
        logits = logit_scale * (image_features @ text_features.T)     # (B, K)

        # ── STEP 5: Graph Label Smoothing loss (Idea 2) ───────────────────────────
        # Replaces standard CE with soft-target CE
        # alpha_smooth=0.0 makes this identical to standard cross-entropy
        if self.alpha_smooth > 0:
            soft_targets = graph_label_smooth(
                label,
                self.A_norm,
                num_classes=self.model.num_classes,
                alpha=self.alpha_smooth
            )                                                          # (B, K)
            log_probs = F.log_softmax(logits, dim=-1)                 # (B, K)
            loss_ce = -(soft_targets * log_probs).sum(dim=-1).mean()  # scalar
        else:
            loss_ce = F.cross_entropy(logits, label)                  # scalar

        # ── STEP 6: Graph Laplacian regularization (Idea 1) ──────────────────────
        # Penalizes semantically similar classes having distant weight vectors
        # lambda_lap=0.0 disables this term entirely
        if self.lambda_lap > 0:
            loss_lap = graph_laplacian_loss(text_features, self.L)    # scalar
        else:
            loss_lap = torch.tensor(0.0, device=image.device)

        # ── STEP 7: Total loss ────────────────────────────────────────────────────
        loss = loss_ce + self.lambda_lap * loss_lap

        # ── STEP 8: Backward pass ─────────────────────────────────────────────────
        self.model_backward_and_update(loss)

        # ── STEP 9: Logging ───────────────────────────────────────────────────────
        loss_summary = {
            "loss_total": loss.item(),
            "loss_ce":    loss_ce.item(),
            "loss_lap":   loss_lap.item(),
            "acc":        compute_accuracy(logits, label)[0].item(),
        }

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary