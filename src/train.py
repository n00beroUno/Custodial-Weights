"""Trainer: walk curriculum, format-tuning, and gate-confirmed contrast training.

The three curricula map directly onto the thesis:
  - step        : LEFT-to-right language modelling over walked triples (composition).
  - step_sft    : teach the exact eval form `S p` -> `O` (the probe boundary).
  - step_contrast: unlikelihood on Gate-REJECTED negatives, so the model learns
                  to not assign probability to lies — with ground truth still
                  owned by the Gate, never the model itself.
"""
import math
import torch, torch.nn as nn
from model import Config, TinyGPT


class Trainer:
    def __init__(self, cfg, tok, seed=0, unfreeze=True, device=None,
                 lr=3e-4, weight_decay=0.1, warmup_steps=0, cos_steps=0,
                 decay_embeddings=False):
        torch.manual_seed(seed)
        self.model = TinyGPT(cfg)
        self.tok = tok
        self.device = device if device is not None else (
            "cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.opt = torch.optim.AdamW(
            self._param_groups(weight_decay, decay_embeddings), lr=lr)
        self.cfg = cfg
        self.lr = lr
        self.warmup_steps = warmup_steps
        self.cos_steps = cos_steps
        self._step = 0
        self.opt.zero_grad()

    def _param_groups(self, weight_decay, decay_embeddings):
        """Standard practice: no weight decay on the embedding (or tied head)."""
        exclude = {"tok.weight"} if not decay_embeddings else set()
        decay, no_decay = [], []
        for name, p in self.model.named_parameters():
            (no_decay if name in exclude or p.ndim < 2 else decay).append(p)
        return [{"params": decay, "weight_decay": weight_decay},
                {"params": no_decay, "weight_decay": 0.0}]

    def set_lr(self, lr):
        """Change the learning rate (e.g. a lower refresh/fine-tune rate)."""
        self.lr = lr
        self.warmup_steps = 0
        self.cos_steps = 0

    def _current_lr(self):
        t = self._step
        if self.cos_steps > 0:
            # linear warmup then cosine decay to 10% of peak
            if t < self.warmup_steps:
                return self.lr * (t + 1) / max(1, self.warmup_steps)
            p = (t - self.warmup_steps) / max(1, self.cos_steps - self.warmup_steps)
            p = min(1.0, p)
            return self.lr * (0.1 + 0.9 * (0.5 * (1 + math.cos(math.pi * p))))
        return self.lr

    def _apply_lr(self):
        for g in self.opt.param_groups:
            g["lr"] = self._current_lr()

    def _batches(self, seqs, batch_size):
        for i in range(0, len(seqs), batch_size):
            chunk = seqs[i:i+batch_size]
            max_len = max(len(s) for s in chunk)
            pad = self.tok.stoi["<pad>"]
            x = [s + [pad]*(max_len-len(s)) for s in chunk]
            x = torch.tensor(x, device=self.device)
            yield x[:, :-1], x[:, 1:]

    def step(self, seqs, batch_size=16, steps=50, clip=1.0, label_smoothing=0.0):
        seqs = [s for s in (self._trim(s) for s in seqs) if len(s) >= 2]
        self.model.train()
        loss_history = []
        for _ in range(steps):
            self.opt.zero_grad()
            total, nb = 0.0, 0
            for xb, yb in self._batches(seqs, batch_size):
                logits = self.model(xb)
                loss = nn.functional.cross_entropy(
                    logits.reshape(-1, self.cfg.vocab_size), yb.reshape(-1),
                    label_smoothing=label_smoothing)
                total += loss.item(); nb += 1
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), clip)
                self._apply_lr()
                self.opt.step()
                self._step += 1
            loss_history.append(total / nb)
        return loss_history

    def _trim(self, seq):
        return seq[: self.cfg.block_size]

    def step_sft(self, items, batch_size=16, steps=50, clip=1.0,
                 label_smoothing=0.0):
        """Format-tuning: teach the exact eval form `S p` -> `O`.

        items: list of (prompt_ids, answer_ids) pairs. Loss counts ONLY on
        positions that produce an answer token (the first is produced from the
        last prompt position — the exact probe boundary). Nothing is taught
        about how to continue the prompt prefix, so composition built by the
        walk curriculum is never overwritten — only the response form is.
        """
        self.model.train()
        loss_history = []
        pad = self.tok.stoi["<pad>"]
        for _ in range(steps):
            self.opt.zero_grad()
            total, nb = 0.0, 0
            for i in range(0, len(items), batch_size):
                chunk = items[i:i+batch_size]
                seqs, n_prompt = [], []
                for p, a in chunk:
                    seqs.append(p + a)
                    n_prompt.append(len(p))
                max_len = max(len(s) for s in seqs)
                wm = torch.zeros((len(seqs), max_len), dtype=torch.bool,
                                 device=self.device)
                yb = torch.full((len(seqs), max_len), -100, dtype=torch.long,
                                device=self.device)
                for r, (s, np_) in enumerate(zip(seqs, n_prompt)):
                    t = torch.tensor(s, device=self.device)
                    L = len(s)
                    yb[r, :L-1] = t[1:]                    # next-token targets
                    wm[r, np_-1:L-1] = True                # positions that produce answer tokens
                xb = torch.tensor(
                    [s + [pad]*(max_len-len(s)) for s in seqs],
                    device=self.device)
                logits = self.model(xb)                    # [B, L, V]
                ce = nn.functional.cross_entropy(
                    logits.reshape(-1, self.cfg.vocab_size), yb.reshape(-1),
                    reduction="none", label_smoothing=label_smoothing)
                ce = ce.view_as(yb)
                loss = (ce * wm).sum() / wm.sum().clamp(min=1).float()
                total += loss.item(); nb += 1
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), clip)
                self._apply_lr()
                self.opt.step()
                self._step += 1
            loss_history.append(total / nb)
        return loss_history

    def step_contrast(self, items, batch_size=16, steps=50, clip=1.0,
                      alpha=0.5, label_smoothing=0.0):
        """Contrastive pair training in the exact eval probe form.

        items: list of (prompt_ids, pos_ids, neg_ids_list)
          prompt = ``s p`` (subject + relation), pos = the true object name,
          neg_list = same-family objects the GATE confirmed are not the object.
        The model is taught to continue ``prompt`` with the positive answer
        (masked NLL) and **not** with each negative answer (unlikelihood
        -log(1-p) at the same answer positions). Negatives are gate-rejected
        lies, so ground truth still lives with the gate.
        """
        self.model.train()
        pad = self.tok.stoi["<pad>"]
        loss_history = []
        for _ in range(steps):
            self.opt.zero_grad()
            total, nb = 0.0, 0
            for i in range(0, len(items), batch_size):
                chunk = items[i:i+batch_size]
                # positives: seq = prompt+pos, loss only on answer tokens
                seqs, n_prompt = [], []
                for pr, po, _n in chunk:
                    seqs.append(pr + po)
                    n_prompt.append(len(pr))
                max_len = max(len(s) for s in seqs)
                wm = torch.zeros((len(seqs), max_len), dtype=torch.bool,
                                 device=self.device)
                yb = torch.full((len(seqs), max_len), -100, dtype=torch.long,
                                device=self.device)
                for r, (s, np_) in enumerate(zip(seqs, n_prompt)):
                    t = torch.tensor(s, device=self.device)
                    L = len(s)
                    yb[r, :L-1] = t[1:]
                    wm[r, np_-1:L-1] = True
                xb = torch.tensor(
                    [s + [pad]*(max_len-len(s)) for s in seqs],
                    device=self.device)
                logits = self.model(xb)
                ce = nn.functional.cross_entropy(
                    logits.reshape(-1, self.cfg.vocab_size), yb.reshape(-1),
                    reduction="none", label_smoothing=label_smoothing).view_as(yb)
                loss_pos = (ce * wm).sum() / wm.sum().clamp(min=1).float()

                # negatives: unlikelihood -log(1-p) at answer positions
                loss_neg = torch.tensor(0.0, device=self.device)
                cnt_neg = 0
                n_neg_max = max(len(n) for _, _, n in chunk)
                for k in range(n_neg_max):
                    neg_seqs, neg_np = [], []
                    for pr, _po, n in chunk:
                        if k < len(n):
                            neg_seqs.append(pr + n[k])
                            neg_np.append(len(pr))
                    if not neg_seqs:
                        continue
                    mlen = max(len(s) for s in neg_seqs)
                    wm2 = torch.zeros((len(neg_seqs), mlen), dtype=torch.bool,
                                      device=self.device)
                    yb2 = torch.full((len(neg_seqs), mlen), -100,
                                     dtype=torch.long, device=self.device)
                    for r, (s, np_) in enumerate(zip(neg_seqs, neg_np)):
                        t = torch.tensor(s, device=self.device)
                        L = len(s)
                        yb2[r, :L-1] = t[1:]
                        wm2[r, np_-1:L-1] = True
                    xb2 = torch.tensor(
                        [s + [pad]*(mlen-len(s)) for s in neg_seqs],
                        device=self.device)
                    logits2 = self.model(xb2)
                    ce2 = nn.functional.cross_entropy(
                        logits2.reshape(-1, self.cfg.vocab_size),
                        yb2.reshape(-1), reduction="none").view_as(yb2)
                    ce2 = ce2 * wm2                      # -log p at answer tokens
                    p = torch.exp(-ce2).clamp(max=1 - 1e-6)
                    ul = -torch.log1p(-p) * wm2          # -log(1-p)
                    loss_neg = loss_neg + ul.sum() / wm2.sum().clamp(min=1).float()
                    cnt_neg += 1
                loss = loss_pos if not cnt_neg else loss_pos + alpha * (loss_neg / cnt_neg)
                total += loss.item(); nb += 1
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), clip)
                self._apply_lr()
                self.opt.step()
                self._step += 1
            loss_history.append(total / nb)
        return loss_history

    def save(self, path):
        import os
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save({"model": self.model.state_dict()}, path)

    def load(self, path):
        self.model.load_state_dict(torch.load(path, weights_only=True)["model"])
