#!/usr/bin/env python3
import torch
from types import SimpleNamespace

from trainers.coop_graph import PromptLearner

class FakeVisual:
    input_resolution = 224

class FakeClipModel:
    def __init__(self):
        self.dtype = torch.float32
        self.ln_final = SimpleNamespace(weight=torch.nn.Parameter(torch.ones(512)))
        self.visual = FakeVisual()
        # Use a large vocab size to accommodate tokenizer indices
        self.token_embedding = torch.nn.Embedding(50000, 512)

# build minimal cfg
cfg = SimpleNamespace()
cfg.TRAINER = SimpleNamespace()
cfg.TRAINER.COOP = SimpleNamespace()
cfg.TRAINER.COOP.N_CTX = 16
cfg.TRAINER.COOP.CTX_INIT = ""  # empty -> random init path
cfg.TRAINER.COOP.CSC = False
cfg.TRAINER.COOP.CLASS_TOKEN_POSITION = "end"
cfg.INPUT = SimpleNamespace()
cfg.INPUT.SIZE = [224]

classnames = ["cat", "dog", "plane"]

clip_model = FakeClipModel()

pl = PromptLearner(cfg, classnames, clip_model)
print("PromptLearner created successfully.")
print("tokenized_prompts device:", pl.tokenized_prompts.device)
print("token_prefix device:", pl.token_prefix.device)
print("token_suffix device:", pl.token_suffix.device)
