# import os
# import re
# import pickle
# import random
# import math
# import sys
# import numpy as np

# import torch
# import torch.nn as nn
# from torch.utils.data import Dataset, DataLoader
# from torch.nn.utils.rnn import pad_sequence

# import matplotlib
# matplotlib.use('Agg')
# import matplotlib.pyplot as plt


# # -----------------------------------------
# # 추가된 함수: 데이터 내 NaN/Inf, 극단값 간단 검사
# # -----------------------------------------
# def check_data_sanity(all_data, max_coord_threshold=1e6):
#     clean_data = []
#     for d in all_data:
#         raw_traj = d.get("raw_trajectory", None)
#         if raw_traj is None:
#             continue
#         raw_traj = np.array(raw_traj, dtype=np.float32)
#         if not np.all(np.isfinite(raw_traj)):
#             continue
#         if np.abs(raw_traj).max() > max_coord_threshold:
#             continue
#         clean_data.append(d)
#     print(f"[check_data_sanity] clean_data: {len(clean_data)} / {len(all_data)}")
#     return clean_data


# # NCCL 포트 기본값
# if 'MASTER_PORT' not in os.environ:
#     os.environ['MASTER_PORT'] = '29502'


# ########################################
# # Data Splitting Utility
# ########################################
# def split_all_data(all_data, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1):
#     random.shuffle(all_data)
#     N = len(all_data)
#     train_end = int(N * train_ratio)
#     val_end = train_end + int(N * val_ratio)
#     return all_data[:train_end], all_data[train_end:val_end], all_data[val_end:]


# ########################################
# # Polygon Lane ROI 전처리
# ########################################
# def filter_context(context: str):
#     if not context.strip():
#         return "No context provided", "R2L"
#     lines = context.splitlines()
#     filtered = []
#     for line in lines:
#         if re.match(r'^\s*A[4-6]\s*:', line):
#             return None, None
#         if re.match(r'^\s*A[1-3]\s*:', line):
#             filtered.append(line)
#     if not filtered:
#         return "No valid context lines", "R2L"
#     ctx = "\n".join(filtered).strip()
#     low = context.lower()
#     direction = "L2R" if "left to right" in low else "R2L"
#     return ctx, direction


# def parse_lane_from_context(context: str):
#     m = re.search(r"lane\s+(A[1-3]|safe)", context)
#     if not m: return None
#     s = m.group(1)
#     return "safe" if s == "safe" else s[1:]


# def get_polygon_from_lane_roi(lane_roi_dict, lane_str):
#     if lane_str is None: return np.zeros((0,2), dtype=np.float32)
#     sub = lane_roi_dict.get("Site C", {}).get("A", {})
#     coords = sub.get(lane_str, [])
#     return np.array(coords, dtype=np.float32) if coords else np.zeros((0,2), dtype=np.float32)


# def is_trajectory_abnormal(raw_traj, lane_label=None,
#                            max_step=50.0, max_speed_diff=30.0):
#     if raw_traj.shape[0] < 2: return False
#     if not np.all(np.isfinite(raw_traj)): return True
#     diffs = np.sqrt(((raw_traj[1:] - raw_traj[:-1])**2).sum(-1))
#     if np.any(diffs > max_step): return True
#     sd = np.abs(diffs[1:] - diffs[:-1])
#     if np.any(sd > max_speed_diff): return True
#     if lane_label:
#         x = raw_traj[:,0]
#         if lane_label=="R2L" and np.any(x[1:]>x[:-1]): return True
#         if lane_label=="L2R" and np.any(x[1:]<x[:-1]): return True
#     return False


# ########################################
# # build_dataset_from_tracks_sliding
# ########################################
# def build_dataset_from_tracks_sliding(track_list,
#                                       seq_len=30, out_len=60, stride=1,
#                                       max_step=50.0, max_speed_diff=30.0,
#                                       downsample=5):
#     inputs, outputs = [], []
#     for item in track_list:
#         raw = item["raw_trajectory"][::downsample]
#         vision = item.get("vision_embeddings", None)
#         if isinstance(vision, torch.Tensor): vision = vision.cpu().numpy()
#         if vision is not None:
#             vision = vision[::downsample]
#             if not np.all(np.isfinite(vision)): continue

#         ctx_str = item.get("context_str","")
#         lane_roi = item.get("lane_roi", None)
#         if lane_roi is None: continue
#         filt, direction = filter_context(ctx_str)
#         if filt is None: continue
#         lane = parse_lane_from_context(ctx_str)
#         if lane is None: continue
#         poly = get_polygon_from_lane_roi(lane_roi, lane)
#         if is_trajectory_abnormal(raw, direction, max_step, max_speed_diff): continue

#         N = raw.shape[0]
#         if N < seq_len+out_len: continue
#         track_id = item.get("track_id","unknown")

#         for st in range(0, N - (seq_len+out_len) + 1, stride):
#             seg = raw[st:st+seq_len+out_len]
#             in_traj, out_traj = seg[:seq_len], seg[seq_len:]
#             xs, ys = seg[:,0], seg[:,1]
#             minx, maxx = xs.min(), xs.max()
#             miny, maxy = ys.min(), ys.max()
#             rx, ry = maxx-minx, maxy-miny
#             if rx<1e-6 or ry<1e-6: continue

#             in_norm = (in_traj - [minx,miny]) / [rx,ry]
#             out_norm = (out_traj - [minx,miny]) / [rx,ry]
#             in_t = torch.tensor(in_norm, dtype=torch.float32).transpose(0,1).unsqueeze(0)
#             out_t = torch.tensor(out_norm, dtype=torch.float32).transpose(0,1).unsqueeze(0)

#             inputs.append({
#                 "traj": in_t, "vision": vision, "poly": poly,
#                 "poly_len": poly.shape[0], "norm": (minx,maxx,miny,maxy),
#                 "track_id": track_id
#             })
#             outputs.append(out_t)
#     return inputs, outputs


# ########################################
# # Dataset & Collate
# ########################################
# class MultiModalTrajectoryDataset(Dataset):
#     def __init__(self, inputs, outputs, max_poly=64):
#         self.inp, self.out = inputs, outputs
#         self.max_poly = max_poly

#     def __len__(self):
#         return len(self.inp)

#     def __getitem__(self, idx):
#         d = self.inp[idx]
#         traj, vision = d["traj"], d["vision"]
#         poly = d["poly"]
#         nl = min(poly.shape[0], self.max_poly)
#         pad = np.zeros((self.max_poly,2),dtype=np.float32)
#         pad[:nl] = poly[:nl]
#         return {
#             "traj": traj.squeeze(0),  # [C,T]
#             "vision": torch.tensor(vision, dtype=torch.float32) if vision is not None else torch.zeros(seq_len,1),
#             "poly": torch.tensor(pad, dtype=torch.float32),
#             "poly_len": nl,
#             "norm": d["norm"],
#             "track_id": d["track_id"],
#             "target": self.out[idx].squeeze(0)
#         }


# def collate_fn(batch):
#     trajs = torch.stack([b["traj"] for b in batch])
#     targets = torch.stack([b["target"] for b in batch])
#     visions = torch.stack([b["vision"] for b in batch])
#     polys   = torch.stack([b["poly"]   for b in batch])
#     poly_l  = [b["poly_len"] for b in batch]
#     norms   = [b["norm"] for b in batch]
#     return {
#         "traj": trajs, "target": targets, "vision": visions,
#         "poly": polys, "poly_len": poly_l, "norm": norms
#     }


# ########################################
# # LanePolygonEncoder
# ########################################
# class LanePolygonEncoder(nn.Module):
#     def __init__(self, d_model=64, nhead=4, layers=2, max_points=64):
#         super().__init__()
#         self.proj = nn.Linear(2, d_model)
#         enc_layer = nn.TransformerEncoderLayer(d_model, nhead, batch_first=True)
#         self.enc = nn.TransformerEncoder(enc_layer, layers)
#         self.pos = nn.Parameter(torch.zeros(1,max_points,d_model))

#     def forward(self, poly, lengths):
#         # poly: [B,P,2]
#         x = self.proj(poly) + self.pos[:,:poly.size(1)]
#         mask = torch.zeros(poly.size(0), poly.size(1), dtype=torch.bool, device=x.device)
#         for i,l in enumerate(lengths):
#             if l<poly.size(1): mask[i,l:] = True
#         out = self.enc(x, src_key_padding_mask=mask)
#         embs = []
#         for i,l in enumerate(lengths):
#             if l>0: embs.append(out[i,:l].mean(0))
#             else:   embs.append(torch.zeros(x.size(-1),device=x.device))
#         return torch.stack(embs,0)


# ########################################
# # SelfAttentionBlock, LTSF Components
# ########################################
# class SelfAttentionBlock(nn.Module):
#     def __init__(self, d_model, nhead=1, dropout=0.1):
#         super().__init__()
#         self.norm1 = nn.LayerNorm(d_model)
#         self.attn  = nn.MultiheadAttention(d_model,nhead,dropout, batch_first=False)
#         self.norm2 = nn.LayerNorm(d_model)
#         self.ffn   = nn.Sequential(nn.Linear(d_model,d_model*4), nn.ReLU(),
#                                    nn.Dropout(dropout), nn.Linear(d_model*4,d_model))
#     def forward(self,x):
#         B,C,T = x.shape
#         x0 = x.permute(2,0,1)
#         r1,_ = self.attn(self.norm1(x0),self.norm1(x0),self.norm1(x0))
#         r1 = r1 + x0
#         r2 = self.ffn(self.norm2(r1)) + r1
#         return r2.permute(1,2,0)


# class LTSF_NLinearEncoder(nn.Module):
#     def __init__(self, window, individual, d_model):
#         super().__init__()
#         self.ind = individual
#         if individual:
#             self.mods = nn.ModuleList([nn.Linear(window,window) for _ in range(d_model)])
#         else:
#             self.mod = nn.Linear(window,window)
#     def forward(self,x):
#         B,C,T = x.shape
#         last = x[:,:, -1:].clone()
#         sub = x - last
#         if self.ind:
#             out = torch.stack([self.mods[i](sub[:,i]) for i in range(C)],1)
#         else:
#             out = self.mod(sub.view(B*C,T)).view(B,C,T)
#         return out + last


# class LTSF_NLinearDecoder(nn.Module):
#     def __init__(self, window, out_sz, individual, d_model,
#                  polygon_embed_dim=64, use_post_mlp=True,
#                  post_mlp_h=64, dropout=0.1,
#                  cross_dim=None, cross_nhead=1, output_dim=2):
#         super().__init__()
#         self.ind = individual
#         if individual:
#             self.mods = nn.ModuleList([nn.Linear(window,out_sz) for _ in range(d_model)])
#         else:
#             self.mod = nn.Linear(window,out_sz)
#         self.lane_fc = nn.Linear(polygon_embed_dim, d_model*out_sz)
#         self.use_mlp = use_post_mlp
#         if use_post_mlp:
#             self.mlp = nn.Sequential(nn.Linear(d_model*out_sz,post_mlp_h),
#                                      nn.ReLU(), nn.Dropout(dropout),
#                                      nn.Linear(post_mlp_h,d_model*out_sz))
#         self.cross_attn = SelfAttentionBlock(d_model,cross_nhead,dropout)
#         self.outp = nn.Linear(d_model,output_dim)
#         self.out_sz = out_sz

#     def forward(self, enc, lane_emb, final_hidden=None):
#         B,C,T = enc.shape
#         last = enc[:,:, -1:].clone()
#         sub = enc - last
#         if self.ind:
#             dec = torch.stack([self.mods[i](sub[:,i]) for i in range(C)],1)
#         else:
#             dec = self.mod(sub.view(B*C,T)).view(B,C,self.out_sz)
#         dec = dec + last.repeat(1,1,self.out_sz)
#         lane_adj = self.lane_fc(lane_emb).view(B,C,self.out_sz)
#         dec = dec + lane_adj
#         if self.use_mlp:
#             dec = self.mlp(dec.view(B,-1)).view(B,C,self.out_sz)
#         # cross-attn via SelfAttentionBlock
#         x = self.cross_attn(dec)
#         return self.outp(x.permute(1,2,0)).permute(0,2,1)


# class TransformerLTSF(nn.Module):
#     def __init__(self, seq_len, out_len, individual, feature_size,
#                  d_model, polygon_embed_dim=64,
#                  use_post_mlp=True, post_mlp_h=64,
#                  nhead=1, dropout=0.1,
#                  cross_dim=None, cross_nhead=1, output_dim=2):
#         super().__init__()
#         self.proj = nn.Conv1d(feature_size, d_model,1)
#         self.enc  = LTSF_NLinearEncoder(seq_len, individual, d_model)
#         self.pos  = nn.Parameter(torch.zeros(1,d_model,seq_len))
#         self.attn = SelfAttentionBlock(d_model,nhead,dropout)
#         self.dec  = LTSF_NLinearDecoder(seq_len,out_len,individual,d_model,
#                                         polygon_embed_dim,use_post_mlp,
#                                         post_mlp_h,dropout,
#                                         cross_dim or d_model, cross_nhead, output_dim)
#     def forward(self,x, lane_emb, final_hidden=None):
#         x = self.proj(x)
#         x = self.enc(x) + self.pos[:,:,:x.size(2)]
#         x = self.attn(x)
#         return self.dec(x, lane_emb, final_hidden)


# ########################################
# # PureTrajectoryModel (MLLM 제외)
# ########################################
# class PureTrajectoryModel(nn.Module):
#     def __init__(self,
#                  seq_len, out_len, individual,
#                  feature_size=2, d_model=64,
#                  polygon_embed_dim=64, use_post_mlp=True,
#                  post_mlp_h=64, ltsf_nhead=1, dropout=0.1):
#         super().__init__()
#         self.lane_enc = LanePolygonEncoder(polygon_embed_dim,4,2,64)
#         self.ltsf = TransformerLTSF(seq_len,out_len,individual,
#                                     feature_size,d_model,
#                                     polygon_embed_dim,use_post_mlp,
#                                     post_mlp_h,ltsf_nhead,dropout,
#                                     d_model,1,feature_size)
#         self.out_len = out_len

#     def forward(self, x, poly, poly_len, y=None, norm=None):
#         lane_emb = self.lane_enc(poly, poly_len)
#         dec = self.ltsf(x, lane_emb)
#         last = x[:,:, -1:].clone()
#         dec = dec + last.repeat(1,1,self.out_len)
#         if y is not None and norm:
#             minx,maxx,miny,maxy = zip(*norm)
#             minx = torch.tensor(minx,device=x.device).view(-1,1,1)
#             maxx = torch.tensor(maxx,device=x.device).view(-1,1,1)
#             miny = torch.tensor(miny,device=x.device).view(-1,1,1)
#             maxy = torch.tensor(maxy,device=x.device).view(-1,1,1)
#             rx, ry = maxx-minx, maxy-miny
#             den = dec.clone(); gt = y.clone()
#             den[:,0] = den[:,0]*rx.squeeze()+minx.squeeze()
#             den[:,1] = den[:,1]*ry.squeeze()+miny.squeeze()
#             gt[:,0]  = gt[:,0]*rx.squeeze()+minx.squeeze()
#             gt[:,1]  = gt[:,1]*ry.squeeze()+miny.squeeze()
#             loss = nn.MSELoss()(den,gt)
#             return loss, dec
#         return dec


# ########################################
# # Entrypoint: 파라미터 개수 확인
# ########################################
# def main():
#     args = dict(
#         seq_len=18, out_len=18, individual=True,
#         feature_size=2, d_model=64,
#         polygon_embed_dim=64, use_post_mlp=True,
#         post_mlp_h=64, ltsf_nhead=2, dropout=0.1
#     )
#     device = torch.device('cpu')
#     model = PureTrajectoryModel(**args).to(device)

#     total = 0
#     print("=== Trainable Parameters (no MLLM) ===")
#     for n,p in model.named_parameters():
#         if p.requires_grad:
#             cnt = p.numel()
#             total += cnt
#             print(f"{n}: {cnt}")
#     print(f"Total: {total}")
#     sys.exit(0)


# if __name__ == "__main__":
#     main()

import os
import re
import pickle
import random
import math
import sys
import numpy as np

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.nn.parallel as pnl
from torch.utils.data import Dataset, DataLoader, Sampler
from torch.utils.data.distributed import DistributedSampler

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, TaskType, get_peft_model


# -----------------------------------------
# 추가된 함수: 데이터 내 NaN/Inf, 극단값 간단 검사
# -----------------------------------------
def check_data_sanity(all_data, max_coord_threshold=1e6):
    clean_data = []
    for d in all_data:
        raw_traj = d.get("raw_trajectory", None)
        if raw_traj is None:
            continue
        raw_traj = np.array(raw_traj, dtype=np.float32)
        if not np.all(np.isfinite(raw_traj)):
            continue
        if np.abs(raw_traj).max() > max_coord_threshold:
            continue
        clean_data.append(d)
    print(f"[check_data_sanity] clean_data: {len(clean_data)} / {len(all_data)}")
    return clean_data


# NCCL 포트 기본값
if 'MASTER_PORT' not in os.environ:
    os.environ['MASTER_PORT'] = '29502'


########################################
# Data Splitting Utility
########################################
def split_all_data(all_data, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1):
    random.shuffle(all_data)
    N = len(all_data)
    train_end = int(N * train_ratio)
    val_end = train_end + int(N * val_ratio)
    train_data = all_data[:train_end]
    val_data = all_data[train_end:val_end]
    test_data = all_data[val_end:]
    return train_data, val_data, test_data


########################################
# Polygon Lane ROI 전처리
########################################
def filter_context(context: str):
    if not context.strip():
        return "No context provided", "R2L"
    lines = context.splitlines()
    filtered_lines = []
    for line in lines:
        if re.match(r'^\s*A[4-6]\s*:', line):
            return None, None
        if re.match(r'^\s*A[1-3]\s*:', line):
            filtered_lines.append(line)
    if not filtered_lines:
        return "No valid context lines", "R2L"

    filtered_ctx = "\n".join(filtered_lines).strip()
    ctx_lower = context.lower()
    if "left to right" in ctx_lower:
        direction = "L2R"
    elif "right to left" in ctx_lower:
        direction = "R2L"
    else:
        direction = "R2L"
    return filtered_ctx, direction


def parse_lane_from_context(context_str: str):
    pattern = r"lane\s+(A[1-3]|safe)"
    match = re.search(pattern, context_str)
    if not match:
        return None
    lane_str = match.group(1)
    if lane_str == "safe":
        return "safe"
    else:
        return lane_str[1:]


def get_polygon_from_lane_roi(lane_roi_dict, lane_str):
    if lane_str is None:
        return np.zeros((0, 2), dtype=np.float32)
    site_key = "Site C"
    zone_key = "A"
    sub_dict = lane_roi_dict.get(site_key, {}).get(zone_key, {})
    if lane_str not in sub_dict:
        return np.zeros((0, 2), dtype=np.float32)
    coords_list = sub_dict[lane_str]
    return np.array(coords_list, dtype=np.float32)


def is_trajectory_abnormal(raw_traj, lane_label=None,
                           max_step=50.0, max_speed_diff=30.0):
    if raw_traj.shape[0] < 2:
        return False
    if not np.all(np.isfinite(raw_traj)):
        return True

    diffs = np.sqrt(np.sum((raw_traj[1:] - raw_traj[:-1])**2, axis=-1))
    if np.any(diffs > max_step):
        return True
    speed_diff = np.abs(diffs[1:] - diffs[:-1])
    if np.any(speed_diff > max_speed_diff):
        return True

    if lane_label is not None:
        x_vals = raw_traj[:, 0]
        if lane_label == "R2L" and np.any(x_vals[1:] > x_vals[:-1]):
            return True
        if lane_label == "L2R" and np.any(x_vals[1:] < x_vals[:-1]):
            return True
    return False


########################################
# build_dataset_from_tracks_sliding
########################################
def build_dataset_from_tracks_sliding(track_list,
                                      seq_len=30,
                                      out_len=60,
                                      stride=1,
                                      max_step=50.0,
                                      max_speed_diff=30.0,
                                      image_width=3840,
                                      image_height=2160,
                                      downsample=5,
                                      tokenizer=None,
                                      max_length=512):
    inputs_list = []
    outputs_list = []

    for item in track_list:
        raw_traj = item["raw_trajectory"]
        raw_traj = raw_traj[::downsample]
        vision_emb = item.get("vision_embeddings", None)
        if isinstance(vision_emb, torch.Tensor):
            vision_emb = vision_emb.cpu()
        if vision_emb is not None:
            vision_emb = vision_emb[::downsample]
            emb_np = vision_emb.numpy() if isinstance(vision_emb, torch.Tensor) else vision_emb
            if not np.all(np.isfinite(emb_np)):
                continue

        original_ctx = item.get("context_str", "")
        lane_roi = item.get("lane_roi", None)
        if lane_roi is None:
            continue

        filtered_ctx, lane_direction = filter_context(original_ctx)
        if filtered_ctx is None:
            continue
        lane_str = parse_lane_from_context(original_ctx)
        if lane_str is None:
            continue

        lane_polygon = get_polygon_from_lane_roi(lane_roi, lane_str)
        if is_trajectory_abnormal(raw_traj, lane_label=lane_direction,
                                  max_step=max_step, max_speed_diff=max_speed_diff):
            continue

        N = raw_traj.shape[0]
        if N < (seq_len + out_len):
            continue

        track_id = item.get("track_id", "unknown")

        for start in range(0, N - (seq_len + out_len) + 1, stride):
            sample_traj = raw_traj[start:start + seq_len + out_len]
            in_traj = sample_traj[:seq_len]
            out_traj = sample_traj[seq_len:seq_len + out_len]

            all_x, all_y = sample_traj[:, 0], sample_traj[:, 1]
            min_x_, max_x_ = float(all_x.min()), float(all_x.max())
            min_y_, max_y_ = float(all_y.min()), float(all_y.max())
            range_x_, range_y_ = max_x_ - min_x_, max_y_ - min_y_
            if range_x_ < 1e-6 or range_y_ < 1e-6:
                continue

            in_norm = np.zeros_like(in_traj, dtype=np.float32)
            out_norm = np.zeros_like(out_traj, dtype=np.float32)
            in_norm[:, 0] = (in_traj[:, 0] - min_x_) / range_x_
            in_norm[:, 1] = (in_traj[:, 1] - min_y_) / range_y_
            out_norm[:, 0] = (out_traj[:, 0] - min_x_) / range_x_
            out_norm[:, 1] = (out_traj[:, 1] - min_y_) / range_y_

            in_traj_t = torch.tensor(in_norm, dtype=torch.float32)
            out_traj_t = torch.tensor(out_norm, dtype=torch.float32)

            if vision_emb is not None:
                vision_tensor = torch.from_numpy(vision_emb) if isinstance(vision_emb, np.ndarray) else vision_emb
                in_vision = vision_tensor[start:start + seq_len]
                if in_vision.shape[0] < seq_len:
                    pad_sz = seq_len - in_vision.shape[0]
                    pad_emb = torch.zeros(pad_sz, in_vision.shape[1], dtype=in_vision.dtype)
                    in_vision = torch.cat([in_vision, pad_emb], dim=0)
                in_vision = in_vision.float()
            else:
                in_vision = torch.zeros(seq_len, 1, dtype=torch.float32)

            prompt_text = (
                "You are an expert traffic environment analyst tasked with examining "
                "comprehensive road environment data. You are provided with partial "
                "trajectory data, visual embeddings (<vision>), and contextual information "
                "about the current road scene. Using this multi-modal information, please "
                "analyze and describe the following aspects in detail:\n\n"
                "1. **Road Geometry and Infrastructure**: Describe the road structure, "
                "including the number of lanes, lane widths, curvature (straight or curved), "
                "and any visible road markings or lane boundary details.\n"
                "2. **Surrounding Environment**: Identify and characterize the type of "
                "environment (e.g., urban, suburban, rural) based on observable elements.\n"
                "3. **Vehicle Dynamics**: Analyze the motion parameters of the vehicle(s) "
                "from the trajectory data, including velocity, acceleration, and heading.\n"
                "4. **Neighboring Entities**: Provide details on the presence and behavior "
                "of nearby vehicles, pedestrians, or obstacles.\n"
                "5. **Safety and Hazard Indicators**: Evaluate any potential hazards or "
                "abnormal conditions.\n\n"
                "Provide your comprehensive answer as a natural language paragraph."
            )
            answer_text = original_ctx

            if tokenizer is not None:
                prompt_enc = tokenizer(
                    prompt_text,
                    truncation=True,
                    max_length=max_length,
                    return_tensors="pt",
                    add_special_tokens=False
                )
                answer_enc = tokenizer(
                    answer_text,
                    truncation=True,
                    max_length=max_length,
                    return_tensors="pt",
                    add_special_tokens=False
                )
                input_ids = torch.cat([prompt_enc["input_ids"], answer_enc["input_ids"]], dim=1)
                attention_mask = torch.cat([prompt_enc["attention_mask"], answer_enc["attention_mask"]], dim=1)
                labels = torch.full_like(input_ids, -100)
                prompt_len = prompt_enc["input_ids"].size(1)
                labels[:, prompt_len:] = input_ids[:, prompt_len:]
                if input_ids.size(1) > max_length:
                    input_ids = input_ids[:, :max_length]
                    attention_mask = attention_mask[:, :max_length]
                    labels = labels[:, :max_length]
            else:
                input_ids = torch.zeros(1,1,dtype=torch.long)
                attention_mask = torch.ones(1,1,dtype=torch.long)
                labels = torch.zeros(1,1,dtype=torch.long)

            sample_input = {
                "trajectory_embeddings": in_traj_t,
                "vision_embeddings": in_vision,
                "context_str": prompt_text,
                "answer_str": answer_text,
                "norm_stat": (min_x_, max_x_, min_y_, max_y_),
                "track_id": track_id,
                "lane_polygon": lane_polygon,
                "input_ids": input_ids.squeeze(0),
                "attention_mask": attention_mask.squeeze(0),
                "labels": labels.squeeze(0)
            }
            inputs_list.append(sample_input)
            outputs_list.append(out_traj_t)

    return inputs_list, outputs_list


########################################
# Dataset & Collate_fn
########################################
class MultiModalTrajectoryDataset(Dataset):
    def __init__(self, inputs_list, outputs_list, max_polygon_points=64):
        self.inputs_list = inputs_list
        self.outputs_list = outputs_list
        self.max_polygon_points = max_polygon_points
        assert len(inputs_list) == len(outputs_list)

    def __len__(self):
        return len(self.inputs_list)

    def __getitem__(self, idx):
        sample = {
            "traj_emb": self.inputs_list[idx]["trajectory_embeddings"],
            "vision_emb": self.inputs_list[idx]["vision_embeddings"],
            "context_str": self.inputs_list[idx]["context_str"],
            "answer_str": self.inputs_list[idx]["answer_str"],
            "norm_stat": self.inputs_list[idx]["norm_stat"],
            "target_traj": self.outputs_list[idx],
            "track_id": self.inputs_list[idx].get("track_id", None),
            "input_ids": self.inputs_list[idx]["input_ids"],
            "attention_mask": self.inputs_list[idx]["attention_mask"],
            "labels": self.inputs_list[idx]["labels"]
        }
        polygon = self.inputs_list[idx]["lane_polygon"]
        n_p = polygon.shape[0] if polygon is not None else 0
        if n_p > self.max_polygon_points:
            polygon = polygon[:self.max_polygon_points, :]
            poly_len = self.max_polygon_points
        else:
            poly_len = n_p if n_p > 0 else 0
        padded = np.zeros((self.max_polygon_points, 2), dtype=np.float32)
        if n_p > 0:
            padded[:poly_len, :] = polygon
        sample["lane_polygon"] = torch.tensor(padded, dtype=torch.float32)
        sample["lane_polygon_len"] = poly_len
        return sample


def custom_collate_fn(batch):
    from torch.nn.utils.rnn import pad_sequence
    traj_list = [b["traj_emb"] for b in batch]
    targ_list = [b["target_traj"] for b in batch]
    vision_list = [b["vision_emb"] for b in batch]

    x_3d = torch.stack([t.transpose(0,1) for t in traj_list], dim=0)
    y_3d = torch.stack([t.transpose(0,1) for t in targ_list], dim=0)
    vision_3d = torch.stack(vision_list, dim=0)

    poly_list = [b["lane_polygon"] for b in batch]
    lane_polygon_tensor = torch.stack(poly_list, dim=0)
    poly_len_list = [b["lane_polygon_len"] for b in batch]

    norm_stats = [b["norm_stat"] for b in batch]
    context_strs = [b["context_str"] for b in batch]
    answer_strs  = [b["answer_str"] for b in batch]
    track_ids = [b["track_id"] for b in batch]

    input_ids_list = [b["input_ids"] for b in batch]
    attn_mask_list = [b["attention_mask"] for b in batch]
    labels_list = [b["labels"] for b in batch]

    input_ids_pad = pad_sequence(input_ids_list, batch_first=True, padding_value=0)
    attn_mask_pad = pad_sequence(attn_mask_list, batch_first=True, padding_value=0)
    labels_pad    = pad_sequence(labels_list, batch_first=True, padding_value=-100)

    return {
        "traj_emb": x_3d,
        "target_traj": y_3d,
        "vision_emb": vision_3d,
        "lane_polygon": lane_polygon_tensor,
        "lane_polygon_len": poly_len_list,
        "norm_stat": norm_stats,
        "context_str": context_strs,
        "answer_str": answer_strs,
        "track_id": track_ids,
        "input_ids": input_ids_pad,
        "attention_mask": attn_mask_pad,
        "labels": labels_pad
    }


########################################
# LanePolygonEncoder
########################################
class LanePolygonEncoder(nn.Module):
    def __init__(self, d_model=64, nhead=4, num_layers=2, max_points=64):
        super().__init__()
        self.d_model = d_model
        self.max_points = max_points
        self.input_proj = nn.Linear(2, d_model)
        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.pos_embedding = nn.Parameter(torch.zeros(1, max_points, d_model))

    def forward(self, polygon_batch, poly_len_list):
        B, P, _ = polygon_batch.shape
        x = self.input_proj(polygon_batch)
        x = x + self.pos_embedding[:, :P, :]
        pad_mask = torch.zeros((B, P), dtype=torch.bool, device=x.device)
        for i in range(B):
            valid_len = poly_len_list[i]
            if valid_len < P:
                pad_mask[i, valid_len:] = True
        enc_out = self.encoder(x, src_key_padding_mask=pad_mask)
        emb_list = []
        for i in range(B):
            valid_len = poly_len_list[i]
            if valid_len > 0:
                emb_i = enc_out[i, :valid_len, :]
                emb_mean = emb_i.mean(dim=0)
            else:
                emb_mean = torch.zeros(self.d_model, device=x.device)
            emb_list.append(emb_mean)
        emb_batch = torch.stack(emb_list, dim=0)
        return emb_batch


########################################
# BLIP Q-Former
########################################
class BlipQFormer(nn.Module):
    def __init__(
        self,
        vision_dim=512,
        hidden_size=768,
        nhead=8,
        num_encoder_layers=4,
        num_decoder_layers=4,
        num_query_tokens=16
    ):
        super().__init__()
        self.num_query_tokens = num_query_tokens
        self.hidden_size = hidden_size
        self.vision_proj = nn.Linear(vision_dim, hidden_size)
        enc_layer = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=nhead, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_encoder_layers)
        self.query_tokens = nn.Parameter(torch.randn(num_query_tokens, hidden_size))
        dec_layer = nn.TransformerDecoderLayer(d_model=hidden_size, nhead=nhead, batch_first=True)
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=num_decoder_layers)

    def forward(self, vision_embs):
        B, Tv, _ = vision_embs.shape
        x = self.vision_proj(vision_embs)
        enc_out = self.encoder(x)
        query = self.query_tokens.unsqueeze(0).expand(B, -1, -1)
        dec_out = self.decoder(query, enc_out)
        return dec_out


########################################
# LlamaWithCrossAttnPEFT
########################################
class LlamaWithCrossAttnPEFT(nn.Module):
    def __init__(self,
                 base_model_name,
                 use_lora=False,
                 lora_r=8,
                 lora_alpha=32,
                 lora_dropout=0.1):
        super().__init__()
        self.llama_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            load_in_8bit=False,
            device_map=None
        )
        if use_lora:
            peft_config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                bias="none",
                task_type=TaskType.CAUSAL_LM,
                target_modules=["q_proj","v_proj"]
            )
            self.llama_model = get_peft_model(self.llama_model, peft_config)
            for name, param in self.llama_model.named_parameters():
                param.requires_grad = ("lora_" in name)

        self.config = self.llama_model.config
        self.hidden_size = self.config.hidden_size

    def forward(self, inputs_embeds, attention_mask, labels=None, output_hidden_states=False):
        outputs = self.llama_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
            output_hidden_states=output_hidden_states,
            return_dict=True
        )
        return outputs


########################################
# LlamaMultiModal
########################################
class LlamaMultiModal(nn.Module):
    def __init__(self,
                 base_model_name="meta-llama/Llama-2-7b-hf",
                 use_lora=True,
                 lora_r=8,
                 lora_alpha=32,
                 lora_dropout=0.1,
                 vision_dim=512,
                 q_hidden_size=768,
                 q_nhead=8,
                 q_enc_layers=4,
                 q_dec_layers=4,
                 q_num_query_tokens=16):
        super().__init__()
        self.qformer = BlipQFormer(
            vision_dim=vision_dim,
            hidden_size=q_hidden_size,
            nhead=q_nhead,
            num_encoder_layers=q_enc_layers,
            num_decoder_layers=q_dec_layers,
            num_query_tokens=q_num_query_tokens
        )
        self.llama_wrapper = LlamaWithCrossAttnPEFT(
            base_model_name=base_model_name,
            use_lora=use_lora,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout
        )
        self.llama_hidden_size = self.llama_wrapper.hidden_size
        if self.llama_hidden_size != q_hidden_size:
            self.q_proj = nn.Linear(q_hidden_size, self.llama_hidden_size)
        else:
            self.q_proj = nn.Identity()

        self.vision_modality_embedding = nn.Parameter(torch.randn(1, 1, self.llama_hidden_size))
        self.text_modality_embedding   = nn.Parameter(torch.randn(1, 1, self.llama_hidden_size))

        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def forward(self,
                vision_embs,
                context_str,
                input_ids=None,
                attention_mask=None,
                labels=None):
        device = vision_embs.device
        B = vision_embs.size(0)

        # Q-Former -> image tokens
        image_tokens = self.qformer(vision_embs)
        image_tokens = self.q_proj(image_tokens)
        image_tokens = image_tokens + self.vision_modality_embedding

        # 텍스트가 주어졌을 때
        if input_ids is not None and attention_mask is not None:
            text_embeds = self.llama_wrapper.llama_model.get_input_embeddings()(input_ids)
            text_embeds = text_embeds + self.text_modality_embedding
            fused_embeds = torch.cat([image_tokens, text_embeds], dim=1)
            img_mask = torch.ones((B, image_tokens.size(1)), dtype=attention_mask.dtype, device=device)
            fused_mask = torch.cat([img_mask, attention_mask], dim=1)
            if labels is not None:
                fused_labels = torch.full(
                    (B, image_tokens.size(1) + labels.size(1)),
                    -100,
                    dtype=labels.dtype,
                    device=device
                )
                fused_labels[:, image_tokens.size(1):] = labels
            else:
                fused_labels = None
            outputs = self.llama_wrapper(
                inputs_embeds=fused_embeds,
                attention_mask=fused_mask,
                labels=fused_labels,
                output_hidden_states=True
            )
            final_hidden = outputs.hidden_states[-1]
            return final_hidden, image_tokens.size(1)

        # 텍스트만 처리할 때
        text_inputs = self.tokenizer(context_str, return_tensors="pt", padding=True, truncation=True)
        text_inputs = {k: v.to(device) for k, v in text_inputs.items()}
        text_embeds = self.llama_wrapper.llama_model.get_input_embeddings()(text_inputs["input_ids"])
        text_embeds = text_embeds + self.text_modality_embedding
        fused_embeds = torch.cat([image_tokens, text_embeds], dim=1)
        img_mask = torch.ones((B, image_tokens.size(1)), dtype=text_inputs["attention_mask"].dtype, device=device)
        fused_mask = torch.cat([img_mask, text_inputs["attention_mask"]], dim=1)
        outputs = self.llama_wrapper(
            inputs_embeds=fused_embeds,
            attention_mask=fused_mask,
            labels=None,
            output_hidden_states=True
        )
        final_hidden = outputs.hidden_states[-1]
        return final_hidden, image_tokens.size(1)


########################################
# SelfAttentionBlock, LTSF Components
########################################
class SelfAttentionBlock(nn.Module):
    def __init__(self, embed_dim, nhead=1, dropout_rate=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.mha = nn.MultiheadAttention(embed_dim, num_heads=nhead, dropout=dropout_rate)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(embed_dim * 4, embed_dim)
        )
        self.dropout2 = nn.Dropout(dropout_rate)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        B, E, T = x.size()
        x_perm = x.permute(2, 0, 1)
        x_norm = self.norm1(x_perm)
        attn_out, _ = self.mha(x_norm, x_norm, x_norm)
        attn_out = self.dropout1(attn_out)
        res1 = x_norm + attn_out
        res1_norm = self.norm2(res1)
        ffn_out = self.ffn(res1_norm)
        ffn_out = self.dropout2(ffn_out)
        out = res1_norm + ffn_out
        out = out.permute(1, 2, 0)
        return out


class LTSF_NLinearEncoder(nn.Module):
    def __init__(self, window_size, individual, d_model):
        super().__init__()
        self.window_size = window_size
        self.individual = individual
        self.channels = d_model
        if self.individual:
            self.encoder_linears = nn.ModuleList(
                [nn.Linear(window_size, window_size) for _ in range(self.channels)]
            )
        else:
            self.encoder_linear = nn.Linear(window_size, window_size)

    def forward(self, x):
        B, C, T = x.shape
        seq_last = x[:, :, -1:].clone()
        x_sub = x - seq_last
        if self.individual:
            out_list = []
            for i in range(C):
                out_i = self.encoder_linears[i](x_sub[:, i, :])
                out_list.append(out_i.unsqueeze(1))
            encoded = torch.cat(out_list, dim=1)
        else:
            BC = B * C
            encoded = self.encoder_linear(x_sub.view(BC, T)).view(B, C, T)
        encoded = encoded + seq_last
        return encoded


class LTSF_NLinearDecoder(nn.Module):
    def __init__(self, window_size, forecast_size, individual, d_model,
                 polygon_embed_dim=64,
                 use_post_mlp=True,
                 post_mlp_hidden_dim=64,
                 post_mlp_output_dim=None,
                 dropout_rate=0.1,
                 cross_dim=768,
                 cross_nhead=2,
                 output_feature_dim=2):
        super().__init__()
        self.window_size = window_size
        self.forecast_size = forecast_size
        self.individual = individual
        self.channels = d_model

        if self.individual:
            self.decoder_linears = nn.ModuleList(
                [nn.Linear(window_size, forecast_size) for _ in range(self.channels)]
            )
        else:
            self.decoder_linear = nn.Linear(window_size, forecast_size)

        self.lane_fc = nn.Linear(polygon_embed_dim, self.channels * self.forecast_size)
        self.use_post_mlp = use_post_mlp
        if post_mlp_output_dim is None:
            post_mlp_output_dim = self.channels * self.forecast_size
        if self.use_post_mlp:
            self.post_mlp = nn.Sequential(
                nn.Linear(self.channels * self.forecast_size, post_mlp_hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(post_mlp_hidden_dim, post_mlp_output_dim)
            )

        self.cross_attn = nn.MultiheadAttention(embed_dim=cross_dim, num_heads=cross_nhead,
                                                dropout=dropout_rate, batch_first=False)
        self.dec_proj   = nn.Linear(d_model, cross_dim)
        self.dec_unproj = nn.Linear(cross_dim, d_model)

        self.fusion_layer = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
        self.out_proj = nn.Linear(d_model, output_feature_dim)

    def forward(self, encoded, lane_polygon_emb, final_hidden):
        B, C, T = encoded.shape
        seq_last = encoded[:, :, -1:].clone()
        x_sub = encoded - seq_last

        if self.individual:
            out_list = []
            for i in range(C):
                dec_i = self.decoder_linears[i](x_sub[:, i, :])
                out_list.append(dec_i.unsqueeze(1))
            decoded = torch.cat(out_list, dim=1)
        else:
            BC = B * C
            decoded = self.decoder_linear(x_sub.view(BC, T)).view(B, C, self.forecast_size)

        decoded = decoded + seq_last.repeat(1, 1, self.forecast_size)
        lane_adj = self.lane_fc(lane_polygon_emb).view(B, C, self.forecast_size)
        decoded = decoded + lane_adj

        if self.use_post_mlp:
            flat_dec = decoded.reshape(B, -1)
            post_out = self.post_mlp(flat_dec).view(B, C, self.forecast_size)
            decoded = post_out

        dec_t = decoded.permute(0, 2, 1)
        proj_dec = self.dec_proj(dec_t)
        query = proj_dec.transpose(0, 1)
        key = final_hidden.transpose(0, 1)
        val = final_hidden.transpose(0, 1)
        cross_out = self.cross_attn(query, key, val)[0].transpose(0, 1)
        cross_to_d = self.dec_unproj(cross_out)
        fused = dec_t + cross_to_d
        fused = self.fusion_layer(fused)
        out = fused.permute(0, 2, 1)
        out = self.out_proj(out)
        out = out.permute(0, 2, 1)
        return out


class TransformerLTSF(nn.Module):
    def __init__(self,
                 seq_len, out_len, individual, feature_size,
                 d_model,
                 polygon_embed_dim=64,
                 use_post_mlp=True,
                 post_mlp_hidden_dim=64,
                 post_mlp_output_dim=None,
                 nhead=1,
                 dropout_rate=0.1,
                 cross_dim=768,
                 cross_nhead=2,
                 output_feature_dim=2):
        super().__init__()
        self.token_proj = nn.Conv1d(feature_size, d_model, kernel_size=1)
        self.nlinear_encoder = LTSF_NLinearEncoder(seq_len, individual, d_model)
        self.pos_encoding = nn.Parameter(torch.zeros(1, d_model, seq_len))
        self.attn_block = SelfAttentionBlock(embed_dim=d_model, nhead=nhead, dropout_rate=dropout_rate)
        self.decoder = LTSF_NLinearDecoder(seq_len, out_len, individual, d_model,
                                           polygon_embed_dim=polygon_embed_dim,
                                           use_post_mlp=use_post_mlp,
                                           post_mlp_hidden_dim=post_mlp_hidden_dim,
                                           post_mlp_output_dim=d_model * out_len,
                                           dropout_rate=dropout_rate,
                                           cross_dim=cross_dim,
                                           cross_nhead=cross_nhead,
                                           output_feature_dim=output_feature_dim)

    def forward(self, x, lane_polygon_emb, final_hidden):
        x_proj = self.token_proj(x)
        enc = self.nlinear_encoder(x_proj)
        enc = enc + self.pos_encoding[:, :, :enc.size(2)]
        enc = self.attn_block(enc)
        dec = self.decoder(enc, lane_polygon_emb, final_hidden)
        return dec


########################################
# MultiModalTrajectoryModel (최종 모델)
########################################
class MultiModalTrajectoryModel(nn.Module):
    def __init__(self,
                 seq_len,
                 out_len,
                 individual,
                 feature_size=2,
                 d_model=64,
                 lane_polygon_d_model=64,
                 lane_polygon_nhead=4,
                 lane_polygon_layers=2,
                 max_polygon_points=64,
                 use_post_mlp=True,
                 post_mlp_hidden_dim=64,
                 base_model_name="meta-llama/Llama-7B",
                 use_lora=True,
                 lora_r=8,
                 lora_alpha=32,
                 lora_dropout=0.1,
                 vision_dim=512,
                 q_hidden_size=768,
                 q_nhead=8,
                 q_enc_layers=4,
                 q_dec_layers=4,
                 q_num_query_tokens=16,
                 ltsf_nhead=1,
                 ltsf_dropout=0.1):
        super().__init__()
        self.lane_polygon_encoder = LanePolygonEncoder(
            d_model=lane_polygon_d_model,
            nhead=lane_polygon_nhead,
            num_layers=lane_polygon_layers,
            max_points=max_polygon_points
        )
        self.mllm = LlamaMultiModal(
            base_model_name=base_model_name,
            use_lora=use_lora,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            vision_dim=vision_dim,
            q_hidden_size=q_hidden_size,
            q_nhead=q_nhead,
            q_enc_layers=q_enc_layers,
            q_dec_layers=q_dec_layers,
            q_num_query_tokens=q_num_query_tokens
        )
        self.llama_hidden_size = self.mllm.llama_hidden_size
        self.ltsf = TransformerLTSF(
            seq_len=seq_len,
            out_len=out_len,
            individual=individual,
            feature_size=feature_size,
            d_model=d_model,
            polygon_embed_dim=lane_polygon_d_model,
            use_post_mlp=use_post_mlp,
            post_mlp_hidden_dim=post_mlp_hidden_dim,
            post_mlp_output_dim=d_model * out_len,
            nhead=ltsf_nhead,
            dropout_rate=ltsf_dropout,
            cross_dim=self.llama_hidden_size,
            cross_nhead=2,
            output_feature_dim=feature_size
        )
        self.feature_size = feature_size
        self.out_len = out_len

    def forward(self,
                x,
                vision_embs,
                context_str,
                lane_polygon_batch,
                lane_polygon_len,
                y=None,
                norm_stat=None,
                input_ids=None,
                attention_mask=None,
                labels=None):
        device = x.device
        B = x.size(0)

        poly_emb = self.lane_polygon_encoder(lane_polygon_batch, lane_polygon_len)
        final_hidden, _ = self.mllm(
            vision_embs,
            context_str,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        decoded = self.ltsf(x, lane_polygon_emb=poly_emb, final_hidden=final_hidden)
        last_in = x[:, :, -1:].clone()
        last_exp = last_in.repeat(1, 1, self.out_len)
        decoded = decoded + last_exp

        if y is not None and norm_stat is not None:
            min_x = torch.tensor([ns[0] for ns in norm_stat], device=device).view(B,1,1)
            max_x = torch.tensor([ns[1] for ns in norm_stat], device=device).view(B,1,1)
            min_y = torch.tensor([ns[2] for ns in norm_stat], device=device).view(B,1,1)
            max_y = torch.tensor([ns[3] for ns in norm_stat], device=device).view(B,1,1)
            rx = max_x - min_x
            ry = max_y - min_y
            dec_den = decoded.clone()
            gt_den  = y.clone()
            dec_den[:,0,:] = dec_den[:,0,:]*rx.squeeze(2) + min_x.squeeze(2)
            dec_den[:,1,:] = dec_den[:,1,:]*ry.squeeze(2) + min_y.squeeze(2)
            gt_den[:,0,:]  = gt_den[:,0,:]*rx.squeeze(2) + min_x.squeeze(2)
            gt_den[:,1,:]  = gt_den[:,1,:]*ry.squeeze(2) + min_y.squeeze(2)
            loss_x = nn.MSELoss()(dec_den[:,0,:], gt_den[:,0,:])
            loss_y = nn.MSELoss()(dec_den[:,1,:], gt_den[:,1,:])
            return loss_x + loss_y, decoded
        else:
            return decoded


########################################
# Visualization Helpers
########################################
def visualize_one_sample(model, sample, device, idx=0,
                         prefix="test_sample", save_dir="visualization"):
    os.makedirs(save_dir, exist_ok=True)
    model.eval()

    in_traj = sample["traj_emb"]
    out_traj = sample["target_traj"]
    min_x, max_x, min_y, max_y = sample["norm_stat"]

    x_in = in_traj.unsqueeze(0).permute(0,2,1).to(device)
    vision= sample["vision_emb"].unsqueeze(0).to(device)
    polygon= sample["lane_polygon"].unsqueeze(0).to(device)
    poly_len= [sample["lane_polygon_len"]]
    ctx= [sample["context_str"]]

    in_ids= sample["input_ids"].unsqueeze(0).to(device)
    attn_m= sample["attention_mask"].unsqueeze(0).to(device)
    labs= sample["labels"].unsqueeze(0).to(device)

    with torch.no_grad():
        out = model(x_in, vision, ctx, polygon, poly_len,
                    y=out_traj.unsqueeze(0).permute(0,2,1).to(device),
                    norm_stat=[(min_x,max_x,min_y,max_y)],
                    input_ids=in_ids,
                    attention_mask=attn_m,
                    labels=labs)
    pred = out[1] if isinstance(out, tuple) else out

    pred_np = pred.squeeze(0).cpu().numpy().transpose(1,0)
    in_den = in_traj.cpu().numpy()
    gt_den = out_traj.cpu().numpy()

    range_x = max_x - min_x
    range_y = max_y - min_y
    in_den[:,0] = in_den[:,0]*range_x + min_x
    in_den[:,1] = in_den[:,1]*range_y + min_y
    gt_den[:,0] = gt_den[:,0]*range_x + min_x
    gt_den[:,1] = gt_den[:,1]*range_y + min_y
    pred_np[:,0] = pred_np[:,0]*range_x + min_x
    pred_np[:,1] = pred_np[:,1]*range_y + min_y

    plt.figure(figsize=(8,6))
    plt.plot(in_den[:,0], in_den[:,1], 'bo-', label='Past')
    plt.plot(gt_den[:,0], gt_den[:,1], 'go-', label='GT')
    plt.plot(pred_np[:,0], pred_np[:,1], 'ro-', label='Pred')
    plt.title(f"{prefix} idx={idx}")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid(True)
    plt.legend()
    save_path = os.path.join(save_dir, f"{prefix}_idx_{idx}.png")
    plt.savefig(save_path)
    plt.close()
    print(f"[{prefix}] Sample {idx} => Saved to {save_path}")


def visualize_test_samples(model, dataset, device, num_samples=5,
                           prefix="test_sample", save_dir="visualization"):
    os.makedirs(save_dir, exist_ok=True)
    indices= random.sample(range(len(dataset)), min(num_samples, len(dataset)))
    for idx in indices:
        visualize_one_sample(model, dataset[idx], device, idx, prefix, save_dir)


########################################
# Entrypoint: 파라미터 개수 확인 전용
########################################
def main():
    args = {
        "seq_len": 18,
        "out_len": 18,
        "feature_size": 2,
        "d_model": 64,
        "lane_polygon_d_model": 64,
        "lane_polygon_nhead": 4,
        "lane_polygon_layers": 2,
        "max_polygon_points": 64,
        "use_post_mlp": True,
        "post_mlp_hidden_dim": 64,
        "base_model_name": "meta-llama/Llama-3.2-1B",
        "use_lora": False,
        "lora_r": 8,
        "lora_alpha": 32,
        "lora_dropout": 0.1,
        "vision_dim": 512,
        "qformer_hidden_size": 768,
        "qformer_nhead": 8,
        "qformer_enc_layers": 4,
        "qformer_dec_layers": 4,
        "qformer_num_query_tokens": 16,
        "ltsf_nhead": 2,
        "ltsf_dropout": 0.1,
    }
    device = torch.device('cpu')
    model = MultiModalTrajectoryModel(
        seq_len=args["seq_len"],
        out_len=args["out_len"],
        individual=True,
        feature_size=args["feature_size"],
        d_model=args["d_model"],
        lane_polygon_d_model=args["lane_polygon_d_model"],
        lane_polygon_nhead=args["lane_polygon_nhead"],
        lane_polygon_layers=args["lane_polygon_layers"],
        max_polygon_points=args["max_polygon_points"],
        use_post_mlp=args["use_post_mlp"],
        post_mlp_hidden_dim=args["post_mlp_hidden_dim"],
        base_model_name=args["base_model_name"],
        use_lora=args["use_lora"],
        lora_r=args["lora_r"],
        lora_alpha=args["lora_alpha"],
        lora_dropout=args["lora_dropout"],
        vision_dim=args["vision_dim"],
        q_hidden_size=args["qformer_hidden_size"],
        q_nhead=args["qformer_nhead"],
        q_enc_layers=args["qformer_enc_layers"],
        q_dec_layers=args["qformer_dec_layers"],
        q_num_query_tokens=args["qformer_num_query_tokens"],
        ltsf_nhead=args["ltsf_nhead"],
        ltsf_dropout=args["ltsf_dropout"],
    ).to(device)

    total_trainable = 0
    print("=== Trainable Parameters ===")
    for name, param in model.named_parameters():
        if param.requires_grad:
            cnt = param.numel()
            total_trainable += cnt
            print(f"{name}: {cnt}")
    print(f"Total trainable parameters: {total_trainable}")

    print("Training is disabled. Exiting.")
    sys.exit(0)


if __name__ == "__main__":
    main()
