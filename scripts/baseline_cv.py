import os
import re
import pickle
import random
import math
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# 데이터 분할 유틸리티
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

# 데이터셋 생성
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

# 데이터셋 클래스
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
            "norm_stat": self.inputs_list[idx]["norm_stat"],
            "target_traj": self.outputs_list[idx],
            "track_id": self.inputs_list[idx].get("track_id", None),
        }
        return sample

def custom_collate_fn(batch):
    traj_list = [b["traj_emb"] for b in batch]
    targ_list = [b["target_traj"] for b in batch]
    x_3d = torch.stack([t.transpose(0, 1) for t in traj_list], dim=0)  # (B, 2, T_in)
    y_3d = torch.stack([t.transpose(0, 1) for t in targ_list], dim=0)  # (B, 2, T_out)
    norm_stats = [b["norm_stat"] for b in batch]
    track_ids = [b["track_id"] for b in batch]
    return {
        "traj_emb": x_3d,
        "target_traj": y_3d,
        "norm_stat": norm_stats,
        "track_id": track_ids,
    }

# 수정된 상수 속도 모델 (다중 예측 지원)
class ConstantVelocityPredictor(nn.Module):
    def __init__(self, seq_len, out_len, feature_size=2):
        super(ConstantVelocityPredictor, self).__init__()
        self.seq_len = seq_len
        self.out_len = out_len
        self.feature_size = feature_size

    def forward(self, x, y=None, norm_stat=None, num_candidates=1, noise_scale=0.1):
        """
        x: 입력 궤적 (B, 2, T_in)
        y: 목표 궤적 (B, 2, T_out), 손실 계산 시 사용
        norm_stat: 정규화 통계, 손실 계산 시 사용
        num_candidates: 생성할 예측 후보 수
        noise_scale: 속도에 추가할 노이즈 크기
        """
        B, C, T = x.size()  # (B, 2, T_in)
        last_pos = x[:, :, -1]  # (B, 2)
        prev_pos = x[:, :, -2]  # (B, 2)
        velocity = last_pos - prev_pos  # (B, 2)

        # 다중 예측 후보 생성
        pred = torch.zeros(B, num_candidates, self.feature_size, self.out_len, device=x.device)
        for i in range(num_candidates):
            # 속도에 노이즈 추가
            noise = torch.randn(B, self.feature_size, device=x.device) * noise_scale
            noisy_velocity = velocity + noise
            for t in range(self.out_len):
                pred[:, i, :, t] = last_pos + noisy_velocity * (t + 1)

        pred = pred.permute(0, 1, 3, 2)  # (B, num_candidates, T_out, 2)

        if y is not None and norm_stat is not None:
            # 손실 계산을 위한 역정규화
            device = x.device
            B = pred.size(0)
            min_x = torch.tensor([ns[0] for ns in norm_stat], device=device).view(B, 1)
            max_x = torch.tensor([ns[1] for ns in norm_stat], device=device).view(B, 1)
            min_y = torch.tensor([ns[2] for ns in norm_stat], device=device).view(B, 1)
            max_y = torch.tensor([ns[3] for ns in norm_stat], device=device).view(B, 1)
            rx = max_x - min_x
            ry = max_y - min_y

            pred_den = pred.clone()  # (B, num_candidates, T_out, 2)
            gt_den = y.clone().permute(0, 2, 1)  # (B, T_out, 2)
            pred_den[..., 0] = pred_den[..., 0] * rx.unsqueeze(1) + min_x.unsqueeze(1)
            pred_den[..., 1] = pred_den[..., 1] * ry.unsqueeze(1) + min_y.unsqueeze(1)
            gt_den[:, :, 0] = gt_den[:, :, 0] * rx + min_x
            gt_den[:, :, 1] = gt_den[:, :, 1] * ry + min_y

            # 평균 제곱 오차 손실 계산 (최소 오차 후보 선택은 evaluate_cv에서)
            loss = nn.MSELoss(reduction='none')(pred_den, gt_den.unsqueeze(1)).mean(dim=[2, 3])  # (B, num_candidates)
            return loss, pred  # loss: (B, num_candidates), pred: (B, num_candidates, T_out, 2)
        return pred  # (B, num_candidates, T_out, 2)

# 시각화 함수 (첫 번째 후보만 시각화)
def visualize_one_sample(model, sample, device, idx=0, prefix="test_sample", save_dir="vis_cv_trajectory"):
    os.makedirs(save_dir, exist_ok=True)
    model.eval()
    in_traj = sample["traj_emb"]
    out_traj = sample["target_traj"]
    min_x, max_x, min_y, max_y = sample["norm_stat"]
    x_in = in_traj.unsqueeze(0).permute(0, 2, 1).to(device)
    with torch.no_grad():
        pred = model(x_in, num_candidates=10, noise_scale=0.1)  # (1, num_candidates, T_out, 2)
    pred_np = pred[0, 0].cpu().numpy()  # 첫 번째 후보만 시각화
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
    plt.title(f"{prefix} idx={idx} (ConstantVelocityPredictor)")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid(True)
    plt.legend()
    filename = f"{prefix}_idx_{idx}.png"
    save_path = os.path.join(save_dir, filename)
    plt.savefig(save_path)
    plt.close()
    print(f"[{prefix}] Sample {idx} => Saved to {save_path}")

# 평가 함수
def evaluate_cv(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 데이터 로드
    try:
        with open(args["all_data_pkl"], 'rb') as f:
            all_data = pickle.load(f)
    except FileNotFoundError:
        raise Exception(f"데이터 파일 {args['all_data_pkl']}을 찾을 수 없습니다")

    train_data, val_data, test_data = split_all_data(all_data, 0.7, 0.2, 0.1)

    # 테스트 데이터셋 생성
    test_inputs, test_outputs = build_dataset_from_tracks_sliding(
        track_list=test_data, seq_len=args["seq_len"], out_len=args["out_len"], stride=args["stride"],
        max_step=args["max_step"], max_speed_diff=args["max_speed_diff"],
        image_width=args["image_width"], image_height=args["image_height"], downsample=args["downsample"]
    )
    test_dataset = TrajectoryDataset(test_inputs, test_outputs)
    test_loader = DataLoader(test_dataset, batch_size=args["batch_size"], shuffle=False,
                             collate_fn=custom_collate_fn)

    # 모델 초기화
    model = ConstantVelocityPredictor(
        seq_len=args["seq_len"],
        out_len=args["out_len"],
        feature_size=2
    ).to(device)
    model.eval()

    # 평가: minADE, minFDE, minRMSE 계산
    num_candidates = 10  # 예측 후보 수
    total_min_ade = 0.0
    total_min_fde = 0.0
    total_min_rmse = 0.0
    total_samp = 0
    with torch.no_grad():
        for batch_data in test_loader:
            x_t = batch_data["traj_emb"].to(device)  # (B, 2, T_in)
            y_t = batch_data["target_traj"].to(device)  # (B, 2, T_out)
            n_t = batch_data["norm_stat"]
            B_ = x_t.size(0)

            # 다중 예측 생성
            pred = model(x_t, num_candidates=num_candidates, noise_scale=0.1)  # (B, num_candidates, T_out, 2)

            # 정규화 해제
            min_x = torch.tensor([ns[0] for ns in n_t], device=device).view(B_, 1)
            max_x = torch.tensor([ns[1] for ns in n_t], device=device).view(B_, 1)
            min_y = torch.tensor([ns[2] for ns in n_t], device=device).view(B_, 1)
            max_y = torch.tensor([ns[3] for ns in n_t], device=device).view(B_, 1)
            rx = max_x - min_x
            ry = max_y - min_y

            pred_denorm = pred.clone()  # (B, num_candidates, T_out, 2)
            y_denorm = y_t.clone().permute(0, 2, 1)  # (B, T_out, 2)
            pred_denorm[..., 0] = pred_denorm[..., 0] * rx.unsqueeze(1) + min_x.unsqueeze(1)
            pred_denorm[..., 1] = pred_denorm[..., 1] * ry.unsqueeze(1) + min_y.unsqueeze(1)
            y_denorm[..., 0] = y_denorm[..., 0] * rx + min_x
            y_denorm[..., 1] = y_denorm[..., 1] * ry + min_y

            # 오차 계산
            errors = torch.sqrt(((pred_denorm - y_denorm.unsqueeze(1)) ** 2).sum(dim=-1))  # (B, num_candidates, T_out)
            ade_candidates = errors.mean(dim=-1)  # (B, num_candidates)
            fde_candidates = errors[..., -1]  # (B, num_candidates)
            rmse_candidates = torch.sqrt(torch.mean((pred_denorm - y_denorm.unsqueeze(1)) ** 2, dim=[-2, -1]))  # (B, num_candidates)

            # 최소 오차 선택
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
        print(f"[Test] minADE={avg_min_ade:.4f}, minFDE={avg_min_fde:.4f}, minRMSE={avg_min_rmse:.4f} (ConstantVelocityPredictor)")

        # 시각화 (최대 5개 샘플)
        for i in range(min(5, len(test_dataset))):
            sample = test_dataset[i]
            visualize_one_sample(model, sample, device, idx=i,
                                prefix="test_sample", save_dir="vis_cv_trajectory_test")

def main():
    args = {
        "all_data_pkl": "/home/user/MLLM/data/all_data.pkl",
        "seq_len": 6,
        "out_len": 30,
        "batch_size": 16,
        "stride": 6,
        "downsample": 5,
        "max_step": 50.0,
        "max_speed_diff": 30.0,
        "image_width": 3840,
        "image_height": 2160,
    }
    evaluate_cv(args)

if __name__ == "__main__":
    main()