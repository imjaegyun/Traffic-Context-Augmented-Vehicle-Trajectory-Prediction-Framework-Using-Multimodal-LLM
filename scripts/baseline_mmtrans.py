import os
import re
import pickle
import random
import math
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

if 'MASTER_PORT' not in os.environ:
    os.environ['MASTER_PORT'] = '29502'

# Data Splitting Utility
def split_all_data(all_data, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1):
    random.shuffle(all_data)
    N = len(all_data)
    train_end = int(N * train_ratio)
    val_end = train_end + int(N * val_ratio)
    train_data = all_data[:train_end]
    val_data = all_data[train_end:val_end]
    test_data = all_data[val_end:]
    return train_data, val_data, test_data

# Polygon Lane ROI 전처리
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

def is_trajectory_abnormal(raw_traj, lane_label=None, max_step=50.0, max_speed_diff=30.0):
    if raw_traj.shape[0] < 2:
        return False
    diffs = np.sqrt(np.sum((raw_traj[1:] - raw_traj[:-1])**2, axis=-1))
    if np.any(diffs > max_step):
        return True
    speed_diff = np.abs(diffs[1:] - diffs[:-1])
    if np.any(speed_diff > max_speed_diff):
        return True
    if lane_label is not None:
        x_vals = raw_traj[:, 0]
        if lane_label == "R2L":
            if np.any(x_vals[1:] > x_vals[:-1]):
                return True
        elif lane_label == "L2R":
            if np.any(x_vals[1:] < x_vals[:-1]):
                return True
    return False

# Dataset 생성
def build_dataset_from_tracks_sliding(track_list, seq_len=30, out_len=60, stride=1,
                                      max_step=50.0, max_speed_diff=30.0, image_width=3840,
                                      image_height=1280, downsample=5):
    inputs_list = []
    outputs_list = []
    for track_idx, item in enumerate(track_list):
        raw_traj = item["raw_trajectory"]
        raw_traj = raw_traj[::downsample]
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
            sample_traj = raw_traj[start:start+seq_len+out_len]
            in_traj = sample_traj[:seq_len]
            out_traj = sample_traj[seq_len:seq_len+out_len]
            all_x = sample_traj[:, 0]
            all_y = sample_traj[:, 1]
            min_x_ = float(all_x.min())
            max_x_ = float(all_x.max())
            min_y_ = float(all_y.min())
            max_y_ = float(all_y.max())
            range_x_ = max_x_ - min_x_
            range_y_ = max_y_ - min_y_
            if range_x_ < 100:
                continue
            if abs(range_x_) < 1e-6:
                range_x_ = 1.0
            if abs(range_y_) < 1e-6:
                range_y_ = 1.0
            in_norm = np.zeros_like(in_traj, dtype=np.float32)
            out_norm = np.zeros_like(out_traj, dtype=np.float32)
            in_norm[:, 0] = (in_traj[:, 0] - min_x_) / range_x_
            in_norm[:, 1] = (in_traj[:, 1] - min_y_) / range_y_
            out_norm[:, 0] = (out_traj[:, 0] - min_x_) / range_x_
            out_norm[:, 1] = (out_traj[:, 1] - min_y_) / range_y_
            in_traj_t = torch.tensor(in_norm, dtype=torch.float32)
            out_traj_t = torch.tensor(out_norm, dtype=torch.float32)
            lane_norm = np.zeros_like(lane_polygon, dtype=np.float32)
            if lane_polygon.size > 0:
                lane_norm[:, 0] = (lane_polygon[:, 0] - min_x_) / range_x_
                lane_norm[:, 1] = (lane_polygon[:, 1] - min_y_) / range_y_
            lane_polygon_t = torch.tensor(lane_norm, dtype=torch.float32)
            sample_input = {
                "trajectory_embeddings": in_traj_t,
                "lane_polygon": lane_polygon_t,
                "norm_stat": (min_x_, max_x_, min_y_, max_y_),
                "track_id": track_id,
            }
            inputs_list.append(sample_input)
            outputs_list.append(out_traj_t)
    return inputs_list, outputs_list

# Dataset 클래스
class TrajectoryDataset(Dataset):
    def __init__(self, inputs_list, outputs_list):
        self.inputs_list = inputs_list
        self.outputs_list = outputs_list
        assert len(inputs_list) == len(outputs_list)

    def __len__(self):
        return len(self.inputs_list)

    def __getitem__(self, idx):
        sample = {
            "traj_emb": self.inputs_list[idx]["trajectory_embeddings"],
            "lane_polygon": self.inputs_list[idx]["lane_polygon"],
            "norm_stat": self.inputs_list[idx]["norm_stat"],
            "target_traj": self.outputs_list[idx],
            "track_id": self.inputs_list[idx].get("track_id", None),
        }
        return sample

def custom_collate_fn(batch):
    traj_list = [b["traj_emb"] for b in batch]
    lane_list = [b["lane_polygon"] for b in batch]
    targ_list = [b["target_traj"] for b in batch]
    x_3d = torch.stack([t.transpose(0, 1) for t in traj_list], dim=0)  # (B, 2, T_in)
    y_3d = torch.stack([t.transpose(0, 1) for t in targ_list], dim=0)  # (B, 2, T_out)
    max_lane_len = max(l.shape[0] for l in lane_list if l.shape[0] > 0) if any(l.shape[0] > 0 for l in lane_list) else 1
    lane_padded = []
    for lane in lane_list:
        if lane.shape[0] == 0:
            padded = torch.zeros((max_lane_len, 2), dtype=torch.float32)
        else:
            pad_size = max_lane_len - lane.shape[0]
            padded = torch.nn.functional.pad(lane, (0, 0, 0, pad_size), mode='constant', value=0)
        lane_padded.append(padded.transpose(0, 1))  # (2, T_lane)
    lane_3d = torch.stack(lane_padded, dim=0)  # (B, 2, T_lane)
    norm_stats = [b["norm_stat"] for b in batch]
    track_ids = [b["track_id"] for b in batch]
    return {
        "traj_emb": x_3d,
        "lane_polygon": lane_3d,
        "target_traj": y_3d,
        "norm_stat": norm_stats,
        "track_id": track_ids,
    }

# MMTrans 모델 정의
class MMTrans(nn.Module):
    def __init__(self, seq_len, out_len, feature_size=2, d_model=128, n_heads=8, n_layers=4, dropout=0.1, max_len=100):
        super(MMTrans, self).__init__()
        self.seq_len = seq_len
        self.out_len = out_len
        self.feature_size = feature_size
        self.d_model = d_model
        self.max_len = max_len

        # 궤적 데이터 임베딩
        self.traj_embedding = nn.Linear(feature_size, d_model)
        # lane_polygon 데이터 임베딩
        self.lane_embedding = nn.Linear(feature_size, d_model)
        # 위치 인코딩
        self.pos_encoder = self._generate_positional_encoding(max_len, d_model)
        # 트랜스포머 인코더
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        # 디코더
        self.decoder = nn.Linear(d_model * seq_len, out_len * feature_size)

    def _generate_positional_encoding(self, max_len, d_model):
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)  # (1, max_len, d_model)

    def forward(self, traj, lane, y=None, norm_stat=None):
        B, _, T_in = traj.size()  # (B, 2, T_in)
        _, _, T_lane = lane.size()  # (B, 2, T_lane)

        # 궤적 임베딩
        traj = traj.permute(0, 2, 1)  # (B, T_in, 2)
        traj_emb = self.traj_embedding(traj)  # (B, T_in, d_model)
        traj_emb = traj_emb + self.pos_encoder[:, :T_in, :].to(traj.device)

        # lane 임베딩
        lane = lane.permute(0, 2, 1)  # (B, T_lane, 2)
        lane_emb = self.lane_embedding(lane)  # (B, T_lane, d_model)
        lane_emb = lane_emb + self.pos_encoder[:, :T_lane, :].to(lane.device)

        # 모달 결합
        combined = torch.cat((traj_emb, lane_emb), dim=1)  # (B, T_in + T_lane, d_model)

        # 패딩 마스크 생성
        mask = torch.cat((torch.ones(B, T_in, device=traj.device),
                          (lane.sum(dim=-1) != 0).float()), dim=1)  # (B, T_in + T_lane)
        mask = (mask == 0)  # True는 패딩된 위치

        # 트랜스포머 인코딩
        output = self.transformer_encoder(combined, src_key_padding_mask=mask)  # (B, T_in + T_lane, d_model)
        output = output[:, :self.seq_len, :]  # 궤적 부분만 사용 (B, T_in, d_model)
        output = output.reshape(B, -1)  # (B, T_in * d_model)

        # 출력 생성
        pred = self.decoder(output)  # (B, out_len * feature_size)
        pred = pred.view(B, self.out_len, self.feature_size)  # (B, T_out, 2)

        if y is not None and norm_stat is not None:
            device = traj.device
            B = pred.size(0)
            min_x = torch.tensor([ns[0] for ns in norm_stat], device=device).view(B, 1)
            max_x = torch.tensor([ns[1] for ns in norm_stat], device=device).view(B, 1)
            min_y = torch.tensor([ns[2] for ns in norm_stat], device=device).view(B, 1)
            max_y = torch.tensor([ns[3] for ns in norm_stat], device=device).view(B, 1)
            rx = max_x - min_x
            ry = max_y - min_y
            pred_den = pred.clone()
            gt_den = y.clone()
            pred_den[:, :, 0] = pred_den[:, :, 0] * rx + min_x
            pred_den[:, :, 1] = pred_den[:, :, 1] * ry + min_y
            gt_den[:, :, 0] = gt_den[:, :, 0] * rx + min_x
            gt_den[:, :, 1] = gt_den[:, :, 1] * ry + min_y
            loss = nn.MSELoss()(pred_den, gt_den)
            return loss, pred
        return pred

# 시각화 함수
def visualize_one_sample(model, sample, device, idx=0, prefix="test_sample", save_dir="vis_mmtrans"):
    os.makedirs(save_dir, exist_ok=True)
    model.eval()
    in_traj = sample["traj_emb"]
    lane_polygon = sample["lane_polygon"]
    out_traj = sample["target_traj"]
    min_x, max_x, min_y, max_y = sample["norm_stat"]
    x_in = in_traj.unsqueeze(0).permute(0, 2, 1).to(device)
    lane_in = lane_polygon.unsqueeze(0).permute(0, 2, 1).to(device)
    y = out_traj.unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(x_in, lane_in, y=y, norm_stat=[(min_x, max_x, min_y, max_y)])
    if isinstance(out, tuple):
        pred = out[1]
    else:
        pred = out
    pred_np = pred.squeeze(0).cpu().numpy()
    in_np = in_traj.cpu().numpy()
    gt_np = out_traj.cpu().numpy()
    range_x = max_x - min_x
    range_y = max_y - min_y
    in_den = in_np.copy()
    gt_den = gt_np.copy()
    pd_den = pred_np.copy()
    in_den[:, 0] = in_den[:, 0] * range_x + min_x
    in_den[:, 1] = in_den[:, 1] * range_y + min_y
    gt_den[:, 0] = gt_den[:, 0] * range_x + min_x
    gt_den[:, 1] = gt_den[:, 1] * range_y + min_y
    pd_den[:, 0] = pd_den[:, 0] * range_x + min_x
    pd_den[:, 1] = pd_den[:, 1] * range_y + min_y
    plt.figure(figsize=(8, 6))
    plt.plot(in_den[:, 0], in_den[:, 1], 'bo-', label='Past')
    plt.plot(gt_den[:, 0], gt_den[:, 1], 'go-', label='GT')
    plt.plot(pd_den[:, 0], pd_den[:, 1], 'ro-', label='Pred')
    plt.title(f"{prefix} idx={idx} (MMTrans)")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid(True)
    plt.legend()
    filename = f"{prefix}_idx_{idx}.png"
    save_path = os.path.join(save_dir, filename)
    plt.savefig(save_path)
    plt.close()
    print(f"[{prefix}] Sample {idx} => Saved to {save_path}")

def train_ddp(args):
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    dist.init_process_group("nccl", rank=local_rank, world_size=world_size)
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    # 데이터 로드
    with open(args["all_data_pkl"], 'rb') as f:
        all_data = pickle.load(f)
    train_data, val_data, test_data = split_all_data(all_data, 0.7, 0.2, 0.1)

    train_inputs, train_outputs = build_dataset_from_tracks_sliding(
        track_list=train_data, seq_len=args["seq_len"], out_len=args["out_len"], stride=args["stride"],
        max_step=args["max_step"], max_speed_diff=args["max_speed_diff"],
        image_width=args["image_width"], image_height=args["image_height"], downsample=args["downsample"]
    )
    train_dataset = TrajectoryDataset(train_inputs, train_outputs)

    val_inputs, val_outputs = build_dataset_from_tracks_sliding(
        track_list=val_data, seq_len=args["seq_len"], out_len=args["out_len"], stride=args["stride"],
        max_step=args["max_step"], max_speed_diff=args["max_speed_diff"],
        image_width=args["image_width"], image_height=args["image_height"], downsample=args["downsample"]
    )
    val_dataset = TrajectoryDataset(val_inputs, val_outputs)

    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=local_rank, shuffle=True)
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=local_rank, shuffle=False)

    train_loader = DataLoader(train_dataset, batch_size=args["batch_size"],
                              sampler=train_sampler, collate_fn=custom_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=args["batch_size"],
                            sampler=val_sampler, collate_fn=custom_collate_fn)

    # 모델 초기화
    model = MMTrans(seq_len=args["seq_len"], out_len=args["out_len"], feature_size=2,
                    d_model=128, n_heads=8, n_layers=4, dropout=0.1, max_len=100).to(device)
    ddp_model = pnl.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)
    optimizer = torch.optim.AdamW(ddp_model.parameters(), lr=args["lr"], weight_decay=1e-4)

    epochs = args["epochs"]
    best_val_loss = float('inf')
    best_ckpt_path = "mmtrans_best_val_checkpoint.pt"

    # for epoch in range(epochs):
    #     ddp_model.train()
    #     train_sampler.set_epoch(epoch)
    #     total_loss = 0.0
    #     for batch_data in train_loader:
    #         x = batch_data["traj_emb"].to(device)
    #         lane = batch_data["lane_polygon"].to(device)
    #         y = batch_data["target_traj"].permute(0, 2, 1).to(device)
    #         ns = batch_data["norm_stat"]
    #         optimizer.zero_grad()
    #         ret = ddp_model(x, lane, y=y, norm_stat=ns)
    #         if isinstance(ret, tuple):
    #             loss, _ = ret
    #         else:
    #             loss = torch.tensor(0.0, device=device)
    #         loss.backward()
    #         optimizer.step()
    #         total_loss += loss.item()
    #     avg_loss = total_loss / len(train_loader)

    #     ddp_model.eval()
    #     val_loss = 0.0
    #     with torch.no_grad():
    #         for batch_data in val_loader:
    #             x_v = batch_data["traj_emb"].to(device)
    #             lane_v = batch_data["lane_polygon"].to(device)
    #             y_v = batch_data["target_traj"].permute(0, 2, 1).to(device)
    #             ns_ = batch_data["norm_stat"]
    #             rr = ddp_model(x_v, lane_v, y=y_v, norm_stat=ns_)
    #             if isinstance(rr, tuple):
    #                 lv, _ = rr
    #             else:
    #                 lv = torch.tensor(0.0, device=device)
    #             val_loss += lv.item()
    #     avg_val_loss = val_loss / len(val_loader)

    #     if local_rank == 0:
    #         print(f"[Epoch {epoch+1}/{epochs}] Train Loss={avg_loss:.4f}, Val Loss={avg_val_loss:.4f} (MMTrans)")
    #         if avg_val_loss < best_val_loss:
    #             best_val_loss = avg_val_loss
    #             torch.save(ddp_model.module.state_dict(), best_ckpt_path)
    #             print(f"  Best Val Loss updated => {best_val_loss:.4f}, ckpt={best_ckpt_path}")
    #         rand_idx = random.randint(0, len(val_dataset) - 1)
    #         sample_val = val_dataset[rand_idx]
    #         visualize_one_sample(ddp_model.module, sample_val, device, idx=epoch,
    #                              prefix="val_sample", save_dir="vis_mmtrans_val")

    if local_rank == 0 and os.path.exists(best_ckpt_path):
        checkpoint = torch.load(best_ckpt_path, map_location=device)
        ddp_model.module.load_state_dict(checkpoint)
        print(f"Loaded model from {best_ckpt_path} for testing")
    else:
        if local_rank == 0:
            print(f"Warning: Checkpoint {best_ckpt_path} not found. Testing with untrained model.")

    if local_rank == 0:
        test_inputs, test_outputs = build_dataset_from_tracks_sliding(
            track_list=test_data, seq_len=args["seq_len"], out_len=args["out_len"], stride=args["stride"],
            max_step=args["max_step"], max_speed_diff=args["max_speed_diff"],
            image_width=args["image_width"], image_height=args["image_height"], downsample=args["downsample"]
        )
        test_dataset = TrajectoryDataset(test_inputs, test_outputs)
        test_loader = DataLoader(test_dataset, batch_size=args["batch_size"], shuffle=False,
                                 collate_fn=custom_collate_fn)
        ddp_model.eval()
        
        num_candidates = 10
        total_min_ade = 0.0
        total_min_fde = 0.0
        total_min_rmse = 0.0
        total_samp = 0
        with torch.no_grad():
            ddp_model.train()
            for batch_data in test_loader:
                x_t = batch_data["traj_emb"].to(device)
                lane_t = batch_data["lane_polygon"].to(device)
                y_t = batch_data["target_traj"].permute(0, 2, 1).to(device)
                n_t = batch_data["norm_stat"]
                B_ = x_t.size(0)
                
                candidate_preds = []
                for i in range(num_candidates):
                    out_candidate = ddp_model(x_t, lane_t, y=None, norm_stat=None)
                    if isinstance(out_candidate, tuple):
                        candidate_preds.append(out_candidate[1])
                    else:
                        candidate_preds.append(out_candidate)
                candidate_preds = torch.stack(candidate_preds, dim=1)
                
                min_x = torch.tensor([ns[0] for ns in n_t], device=device).view(B_, 1)
                max_x = torch.tensor([ns[1] for ns in n_t], device=device).view(B_, 1)
                min_y = torch.tensor([ns[2] for ns in n_t], device=device).view(B_, 1)
                max_y = torch.tensor([ns[3] for ns in n_t], device=device).view(B_, 1)
                rx = max_x - min_x
                ry = max_y - min_y
                
                pred_denorm = candidate_preds.clone()
                pred_denorm[..., 0] = pred_denorm[..., 0] * rx.unsqueeze(1) + min_x.unsqueeze(1)
                pred_denorm[..., 1] = pred_denorm[..., 1] * ry.unsqueeze(1) + min_y.unsqueeze(1)
                
                y_denorm = y_t.clone()
                y_denorm[..., 0] = y_denorm[..., 0] * rx + min_x
                y_denorm[..., 1] = y_denorm[..., 1] * ry + min_y
                
                errors = torch.sqrt(((pred_denorm - y_denorm.unsqueeze(1)) ** 2).sum(dim=-1))
                ade_candidates = errors.mean(dim=-1)
                fde_candidates = errors[..., -1]
                rmse_candidates = torch.sqrt(torch.mean((pred_denorm - y_denorm.unsqueeze(1)) ** 2, dim=[-2, -1]))
                
                min_ade, _ = ade_candidates.min(dim=1)
                min_fde, _ = fde_candidates.min(dim=1)
                min_rmse, _ = rmse_candidates.min(dim=1)
                
                total_min_ade += min_ade.sum().item()
                total_min_fde += min_fde.sum().item()
                total_min_rmse += min_rmse.sum().item()
                total_samp += B_
            
            avg_min_ade = total_min_ade / total_samp if total_samp > 0 else 0.0
            avg_min_fde = total_min_fde / total_samp if total_samp > 0 else 0.0
            avg_min_rmse = total_min_rmse / total_samp if total_samp > 0 else 0.0
            print(f"[Test] minADE={avg_min_ade:.4f}, minFDE={avg_min_fde:.4f}, minRMSE={avg_min_rmse:.4f} (MMTrans)")
            
            ddp_model.eval()
            for i in range(min(5, len(test_dataset))):
                sample = test_dataset[i]
                visualize_one_sample(ddp_model.module, sample, device, idx=i,
                                    prefix="test_sample", save_dir="vis_mmtrans_test")

def main():
    args = {
        "all_data_pkl": "/home/user/MLLM/data/all_data.pkl",
        "seq_len": 6,
        "out_len": 30,
        "batch_size": 16,
        "epochs": 100,
        "lr": 5e-4,
        "stride": 6,
        "downsample": 5,
        "max_step": 50.0,
        "max_speed_diff": 30.0,
        "image_width": 3840,
        "image_height": 2160,
    }
    train_ddp(args)

if __name__ == "__main__":
    main()