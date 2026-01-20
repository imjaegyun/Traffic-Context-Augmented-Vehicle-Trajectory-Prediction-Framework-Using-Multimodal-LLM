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
            sample_input = {
                "trajectory_embeddings": in_traj_t,
                "norm_stat": (min_x_, max_x_, min_y_, max_y_),
                "track_id": track_id,
                "lane_polygon": lane_polygon,
            }
            inputs_list.append(sample_input)
            outputs_list.append(out_traj_t)
    return inputs_list, outputs_list

# Dataset 클래스
class TrajectoryDataset(Dataset):
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
            "norm_stat": self.inputs_list[idx]["norm_stat"],
            "target_traj": self.outputs_list[idx],
            "track_id": self.inputs_list[idx].get("track_id", None),
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
    traj_list = [b["traj_emb"] for b in batch]
    targ_list = [b["target_traj"] for b in batch]
    x_3d = torch.stack([t.transpose(0, 1) for t in traj_list], dim=0)  # (B, 2, T_in)
    y_3d = torch.stack([t.transpose(0, 1) for t in targ_list], dim=0)  # (B, 2, T_out)
    norm_stats = [b["norm_stat"] for b in batch]
    track_ids = [b["track_id"] for b in batch]
    poly_list = [b["lane_polygon"] for b in batch]
    lane_polygon_tensor = torch.stack(poly_list, dim=0)
    poly_len_list = [b["lane_polygon_len"] for b in batch]
    return {
        "traj_emb": x_3d,
        "target_traj": y_3d,
        "norm_stat": norm_stats,
        "track_id": track_ids,
        "lane_polygon": lane_polygon_tensor,
        "lane_polygon_len": poly_len_list,
    }

# Trajectron++ 모델
class TrajectronPP(nn.Module):
    def __init__(self, seq_len, out_len, feature_size=2, hidden_dim=128, latent_dim=32):
        super(TrajectronPP, self).__init__()
        self.seq_len = seq_len
        self.out_len = out_len
        self.feature_size = feature_size
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.encoder_lstm = nn.LSTM(feature_size, hidden_dim, batch_first=True)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        self.decoder_lstm = nn.LSTM(latent_dim + feature_size, hidden_dim, batch_first=True)
        self.output_layer = nn.Linear(hidden_dim, feature_size)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, y=None, norm_stat=None):
        B, C, T = x.size()
        x = x.permute(0, 2, 1)  # (B, T_in, 2)
        _, (h_n, c_n) = self.encoder_lstm(x)
        h_n = h_n.squeeze(0)  # (B, hidden_dim)
        mu = self.fc_mu(h_n)  # (B, latent_dim)
        logvar = self.fc_logvar(h_n)  # (B, latent_dim)
        z = self.reparameterize(mu, logvar)  # (B, latent_dim)
        z = z.unsqueeze(1).repeat(1, self.out_len, 1)  # (B, T_out, latent_dim)
        last_in = x[:, -1:, :]  # (B, 1, 2)
        last_in = last_in.repeat(1, self.out_len, 1)  # (B, T_out, 2)
        decoder_input = torch.cat([last_in, z], dim=-1)  # (B, T_out, 2 + latent_dim)
        out, _ = self.decoder_lstm(decoder_input)  # (B, T_out, hidden_dim)
        pred = self.output_layer(out)  # (B, T_out, 2)
        pred = pred.transpose(1, 2)  # (B, 2, T_out)
        if y is not None and norm_stat is not None:
            device = x.device
            B = pred.size(0)
            min_x = torch.tensor([ns[0] for ns in norm_stat], device=device).view(B, 1, 1)
            max_x = torch.tensor([ns[1] for ns in norm_stat], device=device).view(B, 1, 1)
            min_y = torch.tensor([ns[2] for ns in norm_stat], device=device).view(B, 1, 1)
            max_y = torch.tensor([ns[3] for ns in norm_stat], device=device).view(B, 1, 1)
            rx = max_x - min_x
            ry = max_y - min_y
            pred_den = pred.clone()
            gt_den = y.clone()
            pred_den[:, 0, :] = pred_den[:, 0, :] * rx + min_x
            pred_den[:, 1, :] = pred_den[:, 1, :] * ry + min_y
            gt_den[:, 0, :] = gt_den[:, 0, :] * rx + min_x
            gt_den[:, 1, :] = gt_den[:, 1, :] * ry + min_y
            recon_loss = nn.MSELoss()(pred_den, gt_den)
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / B
            loss = recon_loss + 0.1 * kl_loss
            return loss, pred
        return pred

# 시각화 함수 (다중 후보 예측 포함)
def visualize_one_sample_with_candidates(model, sample, device, num_candidates=5, idx=0,
                                        prefix="test_sample", save_dir="vis_trajectronpp_test"):
    os.makedirs(save_dir, exist_ok=True)
    model.eval()

    # 데이터 준비
    in_traj = sample["traj_emb"]           # (T_in, 2)
    out_traj = sample["target_traj"]       # (T_out, 2)
    min_x, max_x, min_y, max_y = sample["norm_stat"]

    # (B=1)로 확장, shape = (1, 2, T_in)
    x_in = in_traj.unsqueeze(0).permute(0, 2, 1).to(device)

    # 여러 후보 예측 생성
    candidate_preds = []
    with torch.no_grad():
        for _ in range(num_candidates):
            out = model(x_in, y=None, norm_stat=None)
            if isinstance(out, tuple):
                pred_traj = out[1]  # decoded
            else:
                pred_traj = out
            candidate_preds.append(pred_traj)

    # (num_candidates, 1, 2, T_out) -> squeeze(1) => (num_candidates, 2, T_out)
    candidate_preds = torch.stack(candidate_preds, dim=0).squeeze(1)

    # 정규화 해제
    rx = max_x - min_x
    ry = max_y - min_y

    in_np = in_traj.cpu().numpy()   # (T_in, 2)
    gt_np = out_traj.cpu().numpy()  # (T_out, 2)

    # 입력/GT
    in_den = in_np.copy()
    gt_den = gt_np.copy()
    in_den[:, 0] = in_den[:, 0] * rx + min_x
    in_den[:, 1] = in_den[:, 1] * ry + min_y
    gt_den[:, 0] = gt_den[:, 0] * rx + min_x
    gt_den[:, 1] = gt_den[:, 1] * ry + min_y

    # 후보 예측
    candidate_den = candidate_preds.cpu().numpy()  # (num_candidates, 2, T_out)
    for i in range(candidate_den.shape[0]):
        candidate_den[i, 0, :] = candidate_den[i, 0, :] * rx + min_x
        candidate_den[i, 1, :] = candidate_den[i, 1, :] * ry + min_y

    # 시각화
    plt.figure(figsize=(8, 6))
    plt.plot(in_den[:, 0], in_den[:, 1], 'bo-', label='Input Trajectory')
    plt.plot(gt_den[:, 0], gt_den[:, 1], 'go-', label='GT Trajectory')
    for i in range(candidate_den.shape[0]):
        pred_x = candidate_den[i, 0, :]
        pred_y = candidate_den[i, 1, :]
        if i == 0:
            plt.plot(pred_x, pred_y, 'r--', label='Predicted Candidates')
        else:
            plt.plot(pred_x, pred_y, 'r--')

    plt.title(f"{prefix} idx={idx} (num_candidates={num_candidates})")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid(True)
    plt.legend()

    filename = f"{prefix}_idx_{idx}.png"
    save_path = os.path.join(save_dir, filename)
    plt.savefig(save_path)
    plt.close()
    print(f"[{prefix}] Sample {idx} with {num_candidates} candidate predictions => Saved to {save_path}")

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
    model = TrajectronPP(seq_len=args["seq_len"], out_len=args["out_len"], feature_size=2,
                         hidden_dim=128, latent_dim=32).to(device)
    ddp_model = pnl.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)
    optimizer = torch.optim.AdamW(ddp_model.parameters(), lr=args["lr"], weight_decay=1e-4)

    # 학습 루프 (주석 처리됨)
    epochs = args["epochs"]
    best_val_loss = float('inf')
    best_ckpt_path = "trajectronpp_best_val_checkpoint.pt"

    # 학습된 모델 로드
    if local_rank == 0 and os.path.exists(best_ckpt_path):
        checkpoint = torch.load(best_ckpt_path, map_location=device)
        ddp_model.module.load_state_dict(checkpoint)
        print(f"Loaded model from {best_ckpt_path} for testing")
    else:
        if local_rank == 0:
            print(f"Warning: Checkpoint {best_ckpt_path} not found. Testing with untrained model.")

    # 테스트 (train.py의 테스트 로직 통합)
    if local_rank == 0:
        test_inputs, test_outputs = build_dataset_from_tracks_sliding(
            track_list=test_data, seq_len=args["seq_len"], out_len=args["out_len"], stride=args["stride"],
            max_step=args["max_step"], max_speed_diff=args["max_speed_diff"],
            image_width=args["image_width"], image_height=args["image_height"], downsample=args["downsample"]
        )
        test_dataset = TrajectoryDataset(test_inputs, test_outputs)
        test_sampler = DistributedSampler(test_dataset, num_replicas=world_size, rank=local_rank, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=args["batch_size"],
                                 sampler=test_sampler, collate_fn=custom_collate_fn)
        ddp_model.eval()

        # 후보 예측 개수 설정
        num_candidates = 10

        total_min_ade = 0.0
        total_min_fde = 0.0
        total_min_rmse = 0.0
        total_samp = 0

        with torch.no_grad():
            ddp_model.train()  # 다중 후보 생성을 위해 train 모드 (VAE의 randomness 활용)
            for batch_data in test_loader:
                x_t = batch_data["traj_emb"].to(device)       # (B, 2, T_in)
                y_t = batch_data["target_traj"].to(device)    # (B, 2, T_out)
                n_t = batch_data["norm_stat"]
                B_ = x_t.size(0)
                candidate_preds = []

                # num_candidates 번 반복
                for _ in range(num_candidates):
                    out_candidate = ddp_model(x_t, y=None, norm_stat=None)
                    if isinstance(out_candidate, tuple):
                        candidate_preds.append(out_candidate[1])
                    else:
                        candidate_preds.append(out_candidate)

                # shape: (B, num_candidates, 2, T_out)
                candidate_preds = torch.stack(candidate_preds, dim=1)

                # 정규화 해제
                min_x = torch.tensor([ns[0] for ns in n_t], device=device).view(B_, 1, 1)
                max_x = torch.tensor([ns[1] for ns in n_t], device=device).view(B_, 1, 1)
                min_y = torch.tensor([ns[2] for ns in n_t], device=device).view(B_, 1, 1)
                max_y = torch.tensor([ns[3] for ns in n_t], device=device).view(B_, 1, 1)
                rx = max_x - min_x
                ry = max_y - min_y

                pred_denorm = candidate_preds.clone()
                pred_denorm[:, :, 0, :] = pred_denorm[:, :, 0, :] * rx + min_x
                pred_denorm[:, :, 1, :] = pred_denorm[:, :, 1, :] * ry + min_y

                y_denorm = y_t.clone().unsqueeze(1)  # (B, 1, 2, T_out)
                y_denorm[:, :, 0, :] = y_denorm[:, :, 0, :] * rx + min_x
                y_denorm[:, :, 1, :] = y_denorm[:, :, 1, :] * ry + min_y

                # 오차 계산
                errors = torch.sqrt(((pred_denorm - y_denorm) ** 2).sum(dim=2))  # (B, num_candidates, T_out)
                ade_candidates = errors.mean(dim=-1)  # (B, num_candidates)
                fde_candidates = errors[..., -1]      # (B, num_candidates)
                rmse_candidates = torch.sqrt(torch.mean((pred_denorm - y_denorm) ** 2, dim=[2, 3]))  # (B, num_candidates)

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
            print(f"[Test] minADE={avg_min_ade:.4f}, minFDE={avg_min_fde:.4f}, minRMSE={avg_min_rmse:.4f} (Trajectron++)")

            # 시각화
            if len(test_dataset) > 0:
                sample_idx = random.randint(0, len(test_dataset) - 1)
                sample_test = test_dataset[sample_idx]
                visualize_one_sample_with_candidates(
                    ddp_model.module,
                    sample_test,
                    device,
                    num_candidates=num_candidates,
                    idx=sample_idx,
                    prefix="test_sample",
                    save_dir="vis_trajectronpp_test"
                )

def main():
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    args = {
        "all_data_pkl": "/home/user/MLLM/data/all_data.pkl",
        "seq_len": 6,
        "out_len": 30,
        "batch_size": 16,
        "epochs": 200,
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