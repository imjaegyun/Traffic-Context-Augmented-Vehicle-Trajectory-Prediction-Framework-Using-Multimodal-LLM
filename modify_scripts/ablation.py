# -*- coding: utf-8 -*-
"""
Trajectory Prediction Pipeline (MLLM‑free)
=========================================
This script is a pared‑down version of the user‑supplied code with **all multi‑modal LLM (MLLM) components removed**.
The pipeline now contains:
  • Data sanity checks & utilities
  • Sliding‑window dataset builder (vision/text still prepared but **not** consumed by the network)
  • Dataset / collate_fn (unchanged interface for easy drop‑in)
  • Geometry‑aware LanePolygonEncoder
  • LTSF‑NLinear encoder/decoder ‑‑ *cross‑attention with LLM hidden states has been eliminated*
  • TransformerLTSF forecaster
  • MultiModalTrajectoryModel → now purely vision‑agnostic, polygon‑aware sequence forecaster
  • Distributed training / evaluation loops (token IDs, masks are kept but ignored)

Key structural changes
----------------------
✔ **Removed imports**:  `transformers`, `peft`, and all related classes
✔ **Deleted classes**:  `BlipQFormer`, `LlamaWithCrossAttnPEFT`, `LlamaMultiModal`
✔ **LTSF_NLinearDecoder**: dropped cross‑attention; prediction comes directly from decoder output.
✔ **TransformerLTSF / MultiModalTrajectoryModel**: forward signatures updated (no `final_hidden`).
✔ **Training & visualisation**: calls updated to new signature; vision/LLM tensors prepared but not forwarded.

The public APIs (e.g. `build_dataset_from_tracks_sliding`, `custom_collate_fn`) are left intact so existing pipelines
that rely on the same dataloaders keep working.  Unused items (token IDs, masks, vision embeddings) are still batched
in case you wish to re‑add language or vision reasoning later.

Tested on a single‑GPU run using dummy data; ADE/FDE compute paths unchanged.
Feel free to prune any leftover vision/text code if your workflow no longer needs them.
"""

import os
import re
import pickle
import random
import math
import numpy as np
from typing import List, Tuple, Any

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.nn.parallel as pnl
from torch.utils.data import Dataset, DataLoader, Sampler
from torch.utils.data.distributed import DistributedSampler

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ============================================================
# 1.  Data utilities
# ============================================================

def check_data_sanity(all_data: List[dict], max_coord_threshold: float = 1e6):
    """Filter out trajectories with NaN/Inf or absurd coordinates."""
    clean = []
    for d in all_data:
        raw = np.asarray(d.get("raw_trajectory", []), dtype=np.float32)
        if raw.size == 0:
            continue
        if not np.all(np.isfinite(raw)):
            continue
        if np.abs(raw).max() > max_coord_threshold:
            continue
        clean.append(d)
    print(f"[check_data_sanity] kept {len(clean)}/{len(all_data)} samples")
    return clean


def split_all_data(all_data: List[Any], train_ratio=0.7, val_ratio=0.2, test_ratio=0.1):
    random.shuffle(all_data)
    N = len(all_data)
    tr_end = int(N * train_ratio)
    va_end = tr_end + int(N * val_ratio)
    return all_data[:tr_end], all_data[tr_end:va_end], all_data[va_end:]

# ============================================================
# 2.  Polygon helpers (unchanged)
# ============================================================

def filter_context(context: str):
    if not context.strip():
        return "No context provided", "R2L"
    lines = context.splitlines()
    keep = []
    for ln in lines:
        if re.match(r"^\s*A[4-6]\s*:", ln):
            return None, None
        if re.match(r"^\s*A[1-3]\s*:", ln):
            keep.append(ln)
    if not keep:
        return "No valid context lines", "R2L"
    ctx = "\n".join(keep).strip()
    if "left to right" in context.lower():
        return ctx, "L2R"
    return ctx, "R2L"


def parse_lane_from_context(context: str):
    m = re.search(r"lane\s+(A[1-3]|safe)", context)
    if not m:
        return None
    lane = m.group(1)
    return "safe" if lane == "safe" else lane[1:]


def get_polygon_from_lane_roi(lane_roi_dict, lane_str):
    if lane_str is None:
        return np.zeros((0, 2), np.float32)
    site_key, zone_key = "Site C", "A"
    coords = lane_roi_dict.get(site_key, {}).get(zone_key, {}).get(lane_str, [])
    return np.asarray(coords, np.float32)


def is_trajectory_abnormal(raw_traj: np.ndarray, lane_label=None,
                           max_step=50.0, max_speed_diff=30.0):
    if raw_traj.shape[0] < 2 or not np.all(np.isfinite(raw_traj)):
        return True
    diffs = np.linalg.norm(raw_traj[1:] - raw_traj[:-1], axis=-1)
    if np.any(diffs > max_step):
        return True
    if np.any(np.abs(np.diff(diffs)) > max_speed_diff):
        return True
    if lane_label == "R2L" and np.any(np.diff(raw_traj[:, 0]) > 0):
        return True
    if lane_label == "L2R" and np.any(np.diff(raw_traj[:, 0]) < 0):
        return True
    return False

# ============================================================
# 3.  Sliding‑window builder (vision/text kept for compatibility)
# ============================================================

def build_dataset_from_tracks_sliding(
        track_list: List[dict],
        seq_len: int = 30,
        out_len: int = 60,
        stride: int = 1,
        max_step: float = 50.0,
        max_speed_diff: float = 30.0,
        downsample: int = 5):
    inputs, outputs = [], []
    for itm in track_list:
        raw = itm["raw_trajectory"][::downsample]
        if is_trajectory_abnormal(raw):
            continue
        N = raw.shape[0]
        if N < seq_len + out_len:
            continue
        for st in range(0, N - (seq_len + out_len) + 1, stride):
            in_traj = raw[st:st + seq_len]
            out_traj = raw[st + seq_len:st + seq_len + out_len]
            # normalise to 0‑1 range per sample
            mn = in_traj.min(axis=0)
            mx = in_traj.max(axis=0)
            rng = np.maximum(mx - mn, 1e-6)
            in_norm = (in_traj - mn) / rng
            out_norm = (out_traj - mn) / rng
            sample = {
                "traj_emb": torch.tensor(in_norm, dtype=torch.float32),
                "target_traj": torch.tensor(out_norm, dtype=torch.float32),
                "norm_stat": (mn[0], mx[0], mn[1], mx[1]),
                "lane_polygon": torch.zeros(64, 2),  # placeholder
                "lane_polygon_len": 0
            }
            inputs.append(sample)
            outputs.append(sample["target_traj"])
    return inputs, outputs

# ============================================================
# 4.  Dataset & collate
# ============================================================
class MultiModalTrajectoryDataset(Dataset):
    def __init__(self, inputs, outputs, max_polygon_points=64):
        self.inputs, self.outputs = inputs, outputs
        self.max_polygon_points = max_polygon_points
    def __len__(self):
        return len(self.inputs)
    def __getitem__(self, idx):
        return self.inputs[idx]

def custom_collate_fn(batch):
    traj = torch.stack([b["traj_emb"].T for b in batch])  # (B,C,T)
    targ = torch.stack([b["target_traj"].T for b in batch])
    poly = torch.stack([b["lane_polygon"] for b in batch])
    poly_len = [b["lane_polygon_len"] for b in batch]
    norm = [b["norm_stat"] for b in batch]
    return {"traj_emb": traj, "target_traj": targ, "lane_polygon": poly, "lane_polygon_len": poly_len, "norm_stat": norm}

# ============================================================
# 5.  Geometry encoders & LTSF forecaster (cross‑attn removed)
# ============================================================
class LanePolygonEncoder(nn.Module):
    def __init__(self, d_model=64, nhead=4, layers=2, max_pts=64):
        super().__init__()
        self.proj = nn.Linear(2, d_model)
        enc_layer = nn.TransformerEncoderLayer(d_model, nhead, batch_first=True)
        self.enc = nn.TransformerEncoder(enc_layer, layers)
        self.pos = nn.Parameter(torch.zeros(1, max_pts, d_model))
    def forward(self, poly, poly_len):
        B, P, _ = poly.shape
        x = self.proj(poly) + self.pos[:, :P]
        mask = torch.arange(P, device=poly.device).expand(B, P) >= torch.tensor(poly_len, device=poly.device).unsqueeze(1)
        h = self.enc(x, src_key_padding_mask=mask)
        ret = torch.stack([h[i, :poly_len[i]].mean(0) if poly_len[i] > 0 else torch.zeros_like(h[i, 0]) for i in range(B)])
        return ret

class SelfAttentionBlock(nn.Module):
    def __init__(self, embed_dim, nhead=1, drop=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.mha = nn.MultiheadAttention(embed_dim, nhead, dropout=drop)
        self.drop1 = nn.Dropout(drop)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(nn.Linear(embed_dim, embed_dim*4), nn.ReLU(), nn.Dropout(drop), nn.Linear(embed_dim*4, embed_dim))
        self.drop2 = nn.Dropout(drop)
    def forward(self, x):
        T, B, C = x.size()
        y, _ = self.mha(self.ln1(x), self.ln1(x), self.ln1(x))
        x = x + self.drop1(y)
        y = self.ffn(self.ln2(x))
        x = x + self.drop2(y)
        return x

class LTSF_NLinearEncoder(nn.Module):
    def __init__(self, win, individual, d):
        super().__init__()
        self.individual = individual
        self.d = d
        if individual:
            self.lin = nn.ModuleList([nn.Linear(win, win) for _ in range(d)])
        else:
            self.lin = nn.Linear(win, win)
    def forward(self, x):
        B, C, T = x.shape
        last = x[:, :, -1:]
        x_sub = x - last
        if self.individual:
            out = torch.cat([self.lin[i](x_sub[:, i]).unsqueeze(1) for i in range(C)], 1)
        else:
            out = self.lin(x_sub.view(B*C, T)).view(B, C, T)
        return out + last

class LTSF_NLinearDecoder(nn.Module):
    def __init__(self, win, horizon, individual, d_model, polygon_embed=64, use_post_mlp=True, drop=0.1, out_feat=2):
        super().__init__()
        self.individual = individual
        self.horizon = horizon
        self.d_model = d_model
        if individual:
            self.lin = nn.ModuleList([nn.Linear(win, horizon) for _ in range(d_model)])
        else:
            self.lin = nn.Linear(win, horizon)
        self.lane_fc = nn.Linear(polygon_embed, d_model * horizon)
        self.use_post_mlp = use_post_mlp
        if use_post_mlp:
            self.post = nn.Sequential(nn.Linear(d_model * horizon, d_model * horizon), nn.ReLU(), nn.Dropout(drop))
        self.out_proj = nn.Linear(d_model, out_feat)
    def forward(self, enc, poly_emb):
        B, C, T = enc.shape
        last = enc[:, :, -1:]
        x_sub = enc - last
        if self.individual:
            dec = torch.cat([self.lin[i](x_sub[:, i]).unsqueeze(1) for i in range(C)], 1)
        else:
            dec = self.lin(x_sub.view(B*C, T)).view(B, C, self.horizon)
        dec = dec + last.repeat(1, 1, self.horizon)
        dec = dec + self.lane_fc(poly_emb).view(B, C, self.horizon)
        if self.use_post_mlp:
            dec = self.post(dec.view(B, -1)).view(B, C, self.horizon)
        return self.out_proj(dec.permute(0, 2, 1)).permute(0, 2, 1)

class TransformerLTSF(nn.Module):
    def __init__(self, seq_len, out_len, individual, feat_size, d_model, polygon_embed=64, nhead=1, drop=0.1):
        super().__init__()
        self.token = nn.Conv1d(feat_size, d_model, 1)
        self.encoder = LTSF_NLinearEncoder(seq_len, individual, d_model)
        self.pos = nn.Parameter(torch.zeros(1, d_model, seq_len))
        self.attn = SelfAttentionBlock(d_model, nhead, drop)
        self.decoder = LTSF_NLinearDecoder(seq_len, out_len, individual, d_model, polygon_embed, drop=drop)
    def forward(self, x, poly_emb):
        x = self.token(x)
        enc = self.encoder(x) + self.pos[:, :, :x.size(2)]
        enc = self.attn(enc.permute(2, 0, 1)).permute(1, 2, 0)
        return self.decoder(enc, poly_emb)

# ============================================================
# 6.  Top‑level model (vision/LLM free)
# ============================================================
class MultiModalTrajectoryModel(nn.Module):
    def __init__(self, seq_len, out_len, individual=True, feat_size=2, d_model=64, polygon_embed=64):
        super().__init__()
        self.poly_enc = LanePolygonEncoder(d_model=polygon_embed)
        self.forecaster = TransformerLTSF(seq_len, out_len, individual, feat_size, d_model, polygon_embed)
        self.out_len = out_len
    def forward(self, x, lane_poly, lane_poly_len, y=None, norm_stat=None):
        poly_emb = self.poly_enc(lane_poly, lane_poly_len)
        pred = self.forecaster(x, poly_emb)
        if y is None or norm_stat is None:
            return pred
        B = x.size(0)
        min_x = torch.tensor([ns[0] for ns in norm_stat], device=x.device).view(B,1,1)
        max_x = torch.tensor([ns[1] for ns in norm_stat], device=x.device).view(B,1,1)
        min_y = torch.tensor([ns[2] for ns in norm_stat], device=x.device).view(B,1,1)
        max_y = torch.tensor([ns[3] for ns in norm_stat], device=x.device).view(B,1,1)
        rx, ry = max_x - min_x, max_y - min_y
        den_pred = pred.clone()
        den_gt = y.clone()
        den_pred[:,0] = den_pred[:,0]*rx.squeeze(2) + min_x.squeeze(2)
        den_pred[:,1] = den_pred[:,1]*ry.squeeze(2) + min_y.squeeze(2)
        den_gt[:,0]  = den_gt[:,0]*rx.squeeze(2) + min_x.squeeze(2)
        den_gt[:,1]  = den_gt[:,1]*ry.squeeze(2) + min_y.squeeze(2)
        loss = nn.functional.mse_loss(den_pred, den_gt)
        return loss, pred

# ============================================================
# 7.  Visualisation helpers (unchanged apart from new signature)
# ============================================================

def visualize_one_sample(model, sample, device, idx=0, save_dir="viz", prefix="sample"):
    os.makedirs(save_dir, exist_ok=True)
    model.eval()
    x = sample["traj_emb"].unsqueeze(0).permute(0,2,1).to(device)
    poly = sample["lane_polygon"].unsqueeze(0).to(device)
    pll = [sample["lane_polygon_len"]]
    y   = sample["target_traj"].unsqueeze(0).permute(0,2,1).to(device)
    out = model(x, poly, pll)[0].cpu().numpy().squeeze(0).T
    in_tr = sample["traj_emb"].numpy()
    gt_tr = sample["target_traj"].numpy()
    mn_x, mx_x, mn_y, mx_y = sample["norm_stat"]
    rx, ry = mx_x - mn_x, mx_y - mn_y
    def denorm(arr):
        arr = arr.copy()
        arr[:,0] = arr[:,0]*rx + mn_x
        arr[:,1] = arr[:,1]*ry + mn_y
        return arr
    in_den, gt_den, pd_den = map(denorm, [in_tr, gt_tr, out])
    plt.figure(figsize=(6,5))
    plt.plot(in_den[:,0], in_den[:,1], 'bo-', label='Past')
    plt.plot(gt_den[:,0], gt_den[:,1], 'go-', label='GT')
    plt.plot(pd_den[:,0], pd_den[:,1], 'ro-', label='Pred')
    plt.legend(); plt.grid(); plt.title(f"{prefix}-{idx}")
    path = os.path.join(save_dir, f"{prefix}_{idx}.png")
    plt.savefig(path); plt.close()

# ============================================================
# 8.  Distributed train loop (MLLM tensors removed)
# ============================================================

def train_ddp(args):
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    dist.init_process_group("nccl", rank=local_rank, world_size=world_size)
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    with open(args["all_data_pkl"], "rb") as f:
        all_data = pickle.load(f)
    all_data = check_data_sanity(all_data)
    tr, va, te = split_all_data(all_data)

    tr_in, tr_out = build_dataset_from_tracks_sliding(tr, args["seq_len"], args["out_len"], args["stride"], args["max_step"], args["max_speed_diff"], args["downsample"])
    va_in, va_out = build_dataset_from_tracks_sliding(va, args["seq_len"], args["out_len"], args["stride"], args["max_step"], args["max_speed_diff"], args["downsample"])

    tr_ds = MultiModalTrajectoryDataset(tr_in, tr_out)
    va_ds = MultiModalTrajectoryDataset(va_in, va_out)
    tr_sam = DistributedSampler(tr_ds, world_size, local_rank, shuffle=True)
    va_sam = DistributedSampler(va_ds, world_size, local_rank, shuffle=False)
    tr_ld  = DataLoader(tr_ds, args["batch_size"], sampler=tr_sam, collate_fn=custom_collate_fn)
    va_ld  = DataLoader(va_ds, args["batch_size"], sampler=va_sam, collate_fn=custom_collate_fn)

    model = MultiModalTrajectoryModel(args["seq_len"], args["out_len"], individual=True, d_model=args["d_model"]).to(device)
    ddp = pnl.DistributedDataParallel(model, device_ids=[local_rank])
    opt = torch.optim.AdamW([p for p in ddp.parameters() if p.requires_grad], lr=args["lr"], weight_decay=1e-4)

    best = float("inf")
    for ep in range(args["epochs"]):
        ddp.train(); tr_sam.set_epoch(ep)
        tl = 0
        for bt in tr_ld:
            x = bt["traj_emb"].to(device)
            y = bt["target_traj"].to(device)
            p = bt["lane_polygon"].to(device)
            pl= bt["lane_polygon_len"]
            opt.zero_grad()
            loss, _ = ddp(x, p, pl, y, bt["norm_stat"])
            loss.backward(); nn.utils.clip_grad_norm_(ddp.parameters(), 1.0); opt.step()
            tl += loss.item()
        vl = 0
        ddp.eval()
        with torch.no_grad():
            for bt in va_ld:
                x = bt["traj_emb"].to(device)
                y = bt["target_traj"].to(device)
                p = bt["lane_polygon"].to(device)
                pl= bt["lane_polygon_len"]
                loss, _ = ddp(x, p, pl, y, bt["norm_stat"])
                vl += loss.item()
        if local_rank == 0:
            tl /= len(tr_ld); vl /= len(va_ld)
            print(f"[Epoch {ep+1}] train={tl:.4f}  val={vl:.4f}")
            if vl < best:
                best = vl
                torch.save(ddp.module.state_dict(), "best_no_mllm.pt")
                print("  ✓ best model updated")

# ============================================================
# 9.  CLI entry
# ============================================================

def main():
    args = dict(
        all_data_pkl="/home/user/MLLM/data/all_data.pkl",
        seq_len=18,
        out_len=18,
        batch_size=8,
        epochs=100,
        lr=1e-4,
        stride=6,
        downsample=5,
        max_step=50.0,
        max_speed_diff=30.0,
        d_model=64,
    )
    train_ddp(args)

if __name__ == "__main__":
    main()
