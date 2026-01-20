import os
# PYTHONHASHSEED를 고정합니다.
os.environ["PYTHONHASHSEED"] = "42"

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

# 시드 고정 (42)
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
# 결정론적 연산을 위한 CuDNN 설정
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Transformers & PEFT
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, TaskType, get_peft_model

if 'MASTER_PORT' not in os.environ:
    os.environ['MASTER_PORT'] = '29502'

########################################
# Data Splitting Utility
########################################
def split_all_data(all_data, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1):
    """
    전체 데이터를 무작위로 섞은 후, 7:2:1 비율로 분할합니다.
    """
    random.shuffle(all_data)
    N = len(all_data)
    train_end = int(N * train_ratio)
    val_end = train_end + int(N * val_ratio)
    train_data = all_data[:train_end]
    val_data = all_data[train_end:val_end]
    test_data = all_data[val_end:]
    return train_data, val_data, test_data

########################################
# 0. Polygon Lane ROI 전처리용 (코드 1의 전처리 사용)
########################################
def filter_context(context: str):
    """
    A4~A6이 등장하면 무효화. A1~A3만 필터.
    left to right -> L2R, right to left -> R2L. 없으면 R2L로 디폴트.
    """
    if not context.strip():
        return "No context provided", "R2R"  # 기본값: R2R

    lines = context.splitlines()
    filtered_lines = []
    for line in lines:
        if re.match(r'^\s*A[4-6]\s*:', line):
            return None, None
        if re.match(r'^\s*A[1-3]\s*:', line):
            filtered_lines.append(line)

    if not filtered_lines:
        return "No valid context lines", "R2R"

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
    """
    lane A1, A2, A3, safe 만 인식
    (코드 1의 방식)
    """
    pattern = r"lane\s+(A[1-3]|safe)"
    match = re.search(pattern, context_str)
    if not match:
        return None
    lane_str = match.group(1)
    if lane_str == "safe":
        return "safe"
    else:
        return lane_str[1:]  # 'A1' -> '1'

def get_polygon_from_lane_roi(lane_roi_dict, lane_str):
    """
    lane_str이 '1','2','3','safe' 등일 때 해당 폴리곤을 반환
    (N,2) float array
    """
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
    """
    비정상 궤적 필터링
    """
    if raw_traj.shape[0] < 2:
        return False
    diffs = np.sqrt(np.sum((raw_traj[1:] - raw_traj[:-1])**2, axis=-1))
    if np.any(diffs > max_step):
        return True
    speed_diff = np.abs(diffs[1:] - diffs[:-1])
    if np.any(speed_diff > max_speed_diff):
        return True

    # 방향 체크
    if lane_label is not None:
        x_vals = raw_traj[:, 0]
        if lane_label == "R2L":
            if np.any(x_vals[1:] > x_vals[:-1]):
                return True
        elif lane_label == "L2R":
            if np.any(x_vals[1:] < x_vals[:-1]):
                return True

    return False

########################################
# 0. build_dataset_from_tracks_sliding (코드 1의 전처리 방식 적용)
########################################
def build_dataset_from_tracks_sliding(track_list,
                                      seq_len=30,
                                      out_len=60,
                                      stride=1,
                                      max_step=50.0,
                                      max_speed_diff=30.0,
                                      image_width=3840,
                                      image_height=1280,
                                      downsample=5):
    inputs_list = []
    outputs_list = []

    for track_idx, item in enumerate(track_list):
        raw_traj = item["raw_trajectory"]
        raw_traj = raw_traj[::downsample]

        vision_emb = item.get("vision_embeddings", None)
        if vision_emb is not None:
            vision_emb = vision_emb[::downsample]

        original_ctx = item.get("context_str", "")
        lane_roi = item.get("lane_roi", None)  # polygon dict
        if lane_roi is None:
            continue

        # 필터링
        filtered_ctx, lane_direction = filter_context(original_ctx)
        if filtered_ctx is None:
            continue
        # 코드 1의 방식: 원본 컨텍스트에서 lane 정보를 파싱
        lane_str = parse_lane_from_context(original_ctx)
        if lane_str is None:
            continue

        lane_polygon = get_polygon_from_lane_roi(lane_roi, lane_str)  # (N,2)

        # 이상치 궤적 필터
        if is_trajectory_abnormal(raw_traj, lane_label=lane_direction,
                                  max_step=max_step, max_speed_diff=max_speed_diff):
            continue

        N = raw_traj.shape[0]
        if N < (seq_len + out_len):
            continue

        track_id = item.get("id", track_idx)

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
            # x 좌표 이동 값이 800 미만이면 해당 샘플 스킵 (코드 1 기준)
            if range_x_ < 800:
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

            if vision_emb is not None:
                in_vision = vision_emb[start:start+seq_len]
                if in_vision.shape[0] < seq_len:
                    pad_sz = seq_len - in_vision.shape[0]
                    pad_emb = torch.zeros(pad_sz, in_vision.shape[1], dtype=in_vision.dtype)
                    in_vision = torch.cat([in_vision, pad_emb], dim=0)
                in_vision = in_vision.float()
            else:
                in_vision = torch.zeros(seq_len, 1, dtype=torch.float32)

            # prompt 구성 (코드 1의 구성)
            role_str = (
                "You are an advanced AI that sees the environment from drone-view videos "
                "and analyzes surrounding vehicles. You assist in trajectory generation."
            )
            instr = (
                f"Analyze the current vehicle (ID={track_id}) motion and predict the next {out_len} frames of its trajectory."
            )
            new_ctx = f"{role_str}\n\nContext:\n{original_ctx}\n\n{instr}\n\nAnswer:\n"

            sample_input = {
                "trajectory_embeddings": in_traj_t,  # (seq_len,2)
                "vision_embeddings": in_vision,        # (seq_len,D)
                "context_str": new_ctx,
                "norm_stat": (min_x_, max_x_, min_y_, max_y_),
                "track_id": track_id,
                # polygon 그대로 저장
                "lane_polygon": lane_polygon,  # (N,2) ndarray
            }
            inputs_list.append(sample_input)
            outputs_list.append(out_traj_t)

    return inputs_list, outputs_list

########################################
# 1. Dataset / Collate
########################################
class MultiModalTrajectoryDataset(Dataset):
    def __init__(self, inputs_list, outputs_list, max_polygon_points=64):
        super().__init__()
        assert len(inputs_list) == len(outputs_list)
        self.inputs_list = inputs_list
        self.outputs_list = outputs_list
        self.max_polygon_points = max_polygon_points

    def __len__(self):
        return len(self.inputs_list)

    def __getitem__(self, idx):
        sample = {
            "traj_emb": self.inputs_list[idx]["trajectory_embeddings"],  # (T_in,2)
            "vision_emb": self.inputs_list[idx]["vision_embeddings"],    # (T_in,D)
            "context_str": self.inputs_list[idx]["context_str"],
            "norm_stat": self.inputs_list[idx]["norm_stat"],
            "target_traj": self.outputs_list[idx],
            "track_id": self.inputs_list[idx].get("track_id", None),
        }
        # lane_polygon: (N,2) -> 패딩 또는 자르기 (max_polygon_points,2)
        polygon = self.inputs_list[idx]["lane_polygon"]
        n_p = polygon.shape[0]
        if n_p > self.max_polygon_points:
            polygon = polygon[:self.max_polygon_points, :]
            poly_len = self.max_polygon_points
        else:
            poly_len = n_p

        # 패딩
        padded = np.zeros((self.max_polygon_points, 2), dtype=np.float32)
        padded[:poly_len, :] = polygon
        sample["lane_polygon"] = torch.tensor(padded, dtype=torch.float32)
        sample["lane_polygon_len"] = poly_len

        return sample

def custom_collate_fn(batch):
    # 궤적, 비전, lane_polygon을 batch-dim으로 묶음
    traj_list = [b["traj_emb"] for b in batch]
    targ_list = [b["target_traj"] for b in batch]
    vision_list = [b["vision_emb"] for b in batch]

    x_3d = []
    y_3d = []
    for t_in, t_out in zip(traj_list, targ_list):
        x_3d.append(t_in.transpose(0, 1))  # (2,T_in)
        y_3d.append(t_out.transpose(0, 1))   # (2,T_out)

    x_3d = torch.stack(x_3d, dim=0)  # (B,2,T_in)
    y_3d = torch.stack(y_3d, dim=0)  # (B,2,T_out)

    vision_3d = torch.stack(vision_list, dim=0)  # (B,T_in,D)

    poly_list = [b["lane_polygon"] for b in batch]
    lane_polygon_tensor = torch.stack(poly_list, dim=0)  # (B, max_polygon_points,2)
    poly_len_list = [b["lane_polygon_len"] for b in batch]

    norm_stats = [b["norm_stat"] for b in batch]
    context = [b["context_str"] for b in batch]
    track_ids = [b["track_id"] for b in batch]

    return {
        "traj_emb": x_3d,
        "target_traj": y_3d,
        "vision_emb": vision_3d,
        "lane_polygon": lane_polygon_tensor,
        "lane_polygon_len": poly_len_list,
        "norm_stat": norm_stats,
        "context_str": context,
        "track_id": track_ids
    }

########################################
# 2. LanePolygonEncoder (Transformer Encoder)
########################################
class LanePolygonEncoder(nn.Module):
    """
    lane 폴리곤 좌표 (N,2)를 입력받아, TransformerEncoder로 처리 후
    최종 mean pooling으로 하나의 벡터(embedding)를 뽑는 형태.
    """
    def __init__(self, d_model=64, nhead=4, num_layers=2, max_points=64):
        super().__init__()
        self.d_model = d_model
        self.max_points = max_points

        # 임베딩 레이어 (2차원 좌표 -> d_model)
        self.input_proj = nn.Linear(2, d_model)
        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

        # 간단한 위치 인코딩 (패딩 마스크를 이용해 유효 부분만 활용)
        self.pos_embedding = nn.Parameter(torch.zeros(1, max_points, d_model))

    def forward(self, polygon_batch, poly_len_list):
        """
        polygon_batch: (B, max_points, 2)
        poly_len_list: list of actual polygon lengths (<=max_points)
        """
        B, P, _ = polygon_batch.shape
        x = self.input_proj(polygon_batch)  # (B,P,d_model)
        # 위치 인코딩은 전체 max_points에 대해 더함 (패딩 마스크 처리로 후반부 무시)
        x = x + self.pos_embedding[:, :P, :]

        pad_mask = torch.zeros((B, P), dtype=torch.bool, device=x.device)
        for i in range(B):
            valid_len = poly_len_list[i]
            if valid_len < P:
                pad_mask[i, valid_len:] = True

        enc_out = self.encoder(x, src_key_padding_mask=pad_mask)  # (B,P,d_model)
        emb_list = []
        for i in range(B):
            valid_len = poly_len_list[i]
            emb_i = enc_out[i, :valid_len, :]
            emb_mean = emb_i.mean(dim=0)
            emb_list.append(emb_mean)
        emb_batch = torch.stack(emb_list, dim=0)  # (B,d_model)
        return emb_batch

########################################
# 3. BLIP-like Q-Former (Encoder+Decoder)
########################################
class BlipQFormer(nn.Module):
    """
    - Vision embeddings (T_v, D_in)를 TransformerEncoder로 처리하고,
    - 고정 개수의 query token을 사용해 TransformerDecoder 진행
    """
    def __init__(self,
                 vision_dim=512,
                 hidden_size=768,
                 nhead=8,
                 num_encoder_layers=4,
                 num_decoder_layers=4,
                 num_query_tokens=16):
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
        """
        vision_embs: (B,T_v,vision_dim)
        return: (B,num_query_tokens,hidden_size)
        """
        B, Tv, _ = vision_embs.shape
        x = self.vision_proj(vision_embs)  # (B,T_v,hidden_size)
        enc_out = self.encoder(x)  # (B,T_v,hidden_size)

        query = self.query_tokens.unsqueeze(0).expand(B, -1, -1)  # (B,num_query_tokens,hidden_size)
        dec_out = self.decoder(query, enc_out)  # (B,num_query_tokens,hidden_size)
        return dec_out

########################################
# 4. LLaMA MultiModal (BLIP-like)
########################################
class LlamaMultiModal(nn.Module):
    def __init__(self,
                 base_model_name="meta-llama/Llama-7B",
                 use_lora=True,
                 lora_r=8,
                 lora_alpha=32,
                 lora_dropout=0.1,
                 vision_dim=512,
                 q_hidden_size=768,
                 q_nhead=8,
                 q_enc_layers=2,
                 q_dec_layers=2,
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

        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.llama_model = AutoModelForCausalLM.from_pretrained(
            base_model_name, load_in_8bit=False, device_map=None
        )
        if use_lora:
            peft_config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                bias="none",
                task_type=TaskType.CAUSAL_LM
            )
            self.llama_model = get_peft_model(self.llama_model, peft_config)

        self.llama_hidden_size = self.llama_model.config.hidden_size
        self.q_hidden_size = q_hidden_size

        if self.llama_hidden_size != q_hidden_size:
            self.q_proj = nn.Linear(q_hidden_size, self.llama_hidden_size)
        else:
            self.q_proj = nn.Identity()
        
        # 모달리티 type embedding 추가
        self.vision_modality_embedding = nn.Parameter(torch.randn(1, 1, self.llama_hidden_size))
        self.text_modality_embedding = nn.Parameter(torch.randn(1, 1, self.llama_hidden_size))

    def forward(self, vision_embs, context_str):
        """
        vision_embs: (B,T_v,vision_dim)
        context_str: list[str]
        return: final_hidden (B,L,hidden), n_img_tokens
        """
        device = vision_embs.device
        B = vision_embs.size(0)

        # Q-Former로부터 이미지 토큰 생성 및 차원 맞춤, 모달리티 임베딩 추가
        image_tokens = self.qformer(vision_embs)  # (B,num_query_tokens,q_hidden_size)
        image_tokens = self.q_proj(image_tokens)  # (B,num_query_tokens,llama_hidden_size)
        image_tokens = image_tokens + self.vision_modality_embedding  # 모달리티 정보 추가

        # 텍스트 임베딩 (모달리티 임베딩 추가)
        text_inputs = self.tokenizer(context_str, return_tensors="pt", padding=True, truncation=True)
        for k, v in text_inputs.items():
            text_inputs[k] = v.to(device)
        input_ids = text_inputs["input_ids"]      # (B, seq)
        attn_mask = text_inputs["attention_mask"]   # (B, seq)
        text_embeds = self.llama_model.get_input_embeddings()(input_ids)  # (B, seq, hidden)
        text_embeds = text_embeds + self.text_modality_embedding

        # 두 모달리티를 concatenate
        cat_embeds = torch.cat([image_tokens, text_embeds], dim=1)  # (B, num_query_tokens+seq, hidden)
        img_mask = torch.ones((B, image_tokens.size(1)), dtype=attn_mask.dtype, device=device)
        cat_mask = torch.cat([img_mask, attn_mask], dim=1)

        outputs = self.llama_model(
            inputs_embeds=cat_embeds,
            attention_mask=cat_mask,
            output_hidden_states=True
        )
        final_hidden = outputs.hidden_states[-1]
        return final_hidden, image_tokens.size(1)

########################################
# 5. LTSF (Encoder/Decoder) + Self-Attn + LanePolygon
#    -> 디코더 내부에서 Cross-Attn(LLaMA 정보) 삽입 (fusion_layer 추가)
########################################
class SelfAttentionBlock(nn.Module):
    def __init__(self, embed_dim, nhead=1, dropout_rate=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.mha = nn.MultiheadAttention(embed_dim, num_heads=nhead, dropout=dropout_rate)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim*4),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(embed_dim*4, embed_dim)
        )
        self.dropout2 = nn.Dropout(dropout_rate)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # x: (B, E, T)
        B, E, T = x.size()
        x_perm = x.permute(2, 0, 1)  # (T, B, E)
        x_norm = self.norm1(x_perm)
        attn_out, _ = self.mha(x_norm, x_norm, x_norm)
        attn_out = self.dropout1(attn_out)
        res1 = x_norm + attn_out

        res1_norm = self.norm2(res1)
        ffn_out = self.ffn(res1_norm)
        ffn_out = self.dropout2(ffn_out)
        out = res1_norm + ffn_out
        out = out.permute(1, 2, 0)  # (B, E, T)
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
        # x: (B, d_model, T)
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
            out_ = self.encoder_linear(x_sub.view(BC, T)).view(B, C, T)
            encoded = out_
        encoded = encoded + seq_last
        return encoded

class LTSF_NLinearDecoder(nn.Module):
    """
    LTSF 디코더 내부에:
    1) 기존 N-Linear 기반 디코딩
    2) Cross-Attn(LLaMA final_hidden) 융합 후 추가적인 Fusion 레이어로 정보를 정제
    3) 최종 출력은 output_feature_dim (예: 2)로 투영
    """
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

        self.cross_dim = cross_dim
        self.cross_attn = nn.MultiheadAttention(embed_dim=cross_dim, num_heads=cross_nhead, dropout=dropout_rate, batch_first=False)

        self.dec_proj = nn.Linear(d_model, cross_dim)
        self.dec_unproj = nn.Linear(cross_dim, d_model)

        # 추가 Fusion 레이어 (Residual + LayerNorm + FFN)
        self.fusion_layer = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )

        # 최종 출력 투영: (d_model -> output_feature_dim)
        self.out_proj = nn.Linear(d_model, output_feature_dim)

    def forward(self, encoded, lane_polygon_emb, final_hidden):
        """
        encoded: (B, d_model, T_out)
        lane_polygon_emb: (B, polygon_embed_dim)
        final_hidden: (B, L, cross_dim)
        """
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
            dec_out = self.decoder_linear(x_sub.view(BC, T)).view(B, C, self.forecast_size)
            decoded = dec_out
        decoded = decoded + seq_last.repeat(1, 1, self.forecast_size)

        lane_adj = self.lane_fc(lane_polygon_emb)
        lane_adj = lane_adj.view(B, C, self.forecast_size)
        decoded = decoded + lane_adj

        if self.use_post_mlp:
            flat_dec = decoded.reshape(B, -1)
            post_out = self.post_mlp(flat_dec)
            post_out = post_out.view(B, C, self.forecast_size)
            decoded = post_out

        # Cross-Attn Fusion
        dec_t = decoded.permute(0, 2, 1)  # (B, T_out, d_model)
        proj_dec = self.dec_proj(dec_t)    # (B, T_out, cross_dim)
        query = proj_dec.transpose(0, 1)    # (T_out, B, cross_dim)
        key   = final_hidden.transpose(0, 1)  # (L, B, cross_dim)
        val   = final_hidden.transpose(0, 1)  # (L, B, cross_dim)
        cross_out = self.cross_attn(query, key, val)[0].transpose(0, 1)  # (B, T_out, cross_dim)
        cross_to_d = self.dec_unproj(cross_out)  # (B, T_out, d_model)
        # Residual fusion + fusion_layer
        fused = dec_t + cross_to_d
        fused = self.fusion_layer(fused)  # (B, T_out, d_model)
        decoded = fused.permute(0, 2, 1)   # (B, d_model, T_out)

        # 최종 출력 투영: (B, d_model, T_out) -> (B, T_out, d_model) -> (B, T_out, output_feature_dim)
        decoded = decoded.permute(0, 2, 1)
        decoded = self.out_proj(decoded)
        decoded = decoded.permute(0, 2, 1)  # (B, output_feature_dim, T_out)
        return decoded

class TransformerLTSF(nn.Module):
    def __init__(self,
                 seq_len, out_len, individual, feature_size,
                 d_model,  # transformer embedding dim (예: 64)
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
        # 수정: LTSF_NLinearDecoder 호출 시 불필요한 nhead 인자는 제거
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
        # x: (B, feature_size, T)
        x_proj = self.token_proj(x)  # (B, d_model, T)
        enc = self.nlinear_encoder(x_proj)  # (B, d_model, T)
        enc = enc + self.pos_encoding[:, :, :enc.size(2)]
        enc = self.attn_block(enc)
        dec = self.decoder(enc, lane_polygon_emb, final_hidden)
        return dec

########################################
# 6. 최종 모델
#    -> LTSF 내부에서 Cross-Attn 수행 후, 최종 출력은 2차원 (예측 궤적)
########################################
class MultiModalTrajectoryModel(nn.Module):
    def __init__(self,
                 seq_len,
                 out_len,
                 individual,
                 feature_size=2,  # 원래 궤적 차원
                 d_model=64,      # transformer 내부 차원
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
                 q_enc_layers=2,
                 q_dec_layers=2,
                 q_num_query_tokens=16,
                 ltsf_nhead=2,
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
            q_num_query_tokens=q_num_query_tokens,
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

    def forward(self, x, vision_embs, context_str,
                lane_polygon_batch, lane_polygon_len,
                y=None, norm_stat=None):
        # x: (B, feature_size, T_in)
        device = x.device
        B = x.size(0)
        # lane polygon encoder
        poly_emb = self.lane_polygon_encoder(lane_polygon_batch, lane_polygon_len)
        # LLaMA multi-modal (vision + text)
        final_hidden, n_img_tokens = self.mllm(vision_embs, context_str)
        # LTSF prediction
        decoded = self.ltsf(x, lane_polygon_emb=poly_emb, final_hidden=final_hidden)

        # Standard residual skip connection (마지막 입력 반복하여 단순 더함)
        last_in = x[:, :, -1:].clone()        # (B, feature_size, 1)
        last_exp = last_in.repeat(1, 1, self.out_len)  # (B, feature_size, T_out)
        decoded = decoded + last_exp

        if (y is not None) and (norm_stat is not None):
            min_x = torch.tensor([ns[0] for ns in norm_stat], device=device).view(B, 1, 1)
            max_x = torch.tensor([ns[1] for ns in norm_stat], device=device).view(B, 1, 1)
            min_y = torch.tensor([ns[2] for ns in norm_stat], device=device).view(B, 1, 1)
            max_y = torch.tensor([ns[3] for ns in norm_stat], device=device).view(B, 1, 1)
            rx = max_x - min_x
            ry = max_y - min_y

            dec_den = decoded.clone()
            gt_den = y.clone()

            dec_den[:, 0, :] = dec_den[:, 0, :] * rx.squeeze(2) + min_x.squeeze(2)
            dec_den[:, 1, :] = dec_den[:, 1, :] * ry.squeeze(2) + min_y.squeeze(2)
            gt_den[:, 0, :]  = gt_den[:, 0, :] * rx.squeeze(2) + min_x.squeeze(2)
            gt_den[:, 1, :]  = gt_den[:, 1, :] * ry.squeeze(2) + min_y.squeeze(2)

            loss_x = nn.MSELoss()(dec_den[:, 0, :], gt_den[:, 0, :])
            loss_y = nn.MSELoss()(dec_den[:, 1, :], gt_den[:, 1, :])
            loss = loss_x + loss_y
            return loss, decoded
        else:
            return decoded

########################################
# 7. TrackGroupSampler (예시)
########################################
class TrackGroupSampler(Sampler):
    def __init__(self, dataset, shuffle=True):
        self.dataset = dataset
        self.shuffle = shuffle
        self.groups = {}
        for idx in range(len(dataset)):
            t_id = dataset[idx]["track_id"]
            if t_id not in self.groups:
                self.groups[t_id] = []
            self.groups[t_id].append(idx)
        self.group_list = list(self.groups.values())

    def __iter__(self):
        if self.shuffle:
            np.random.shuffle(self.group_list)
        for group in self.group_list:
            for idx in group:
                yield idx

    def __len__(self):
        return len(self.dataset)

########################################
# 8. 시각화
########################################
def visualize_one_sample(model, sample, device, idx=0,
                         prefix="test_sample", save_dir="visualization_test_ddp"):
    # Validation/일반 시각화용 (단일 예측)
    os.makedirs(save_dir, exist_ok=True)
    model.eval()

    in_traj = sample["traj_emb"]    # (T_in,2)
    out_traj = sample["target_traj"] # (T_out,2)
    min_x, max_x, min_y, max_y = sample["norm_stat"]

    x_in = in_traj.unsqueeze(0).permute(0, 2, 1).to(device)  # (1,2,T_in)
    vision = sample["vision_emb"].unsqueeze(0).to(device)
    polygon = sample["lane_polygon"].unsqueeze(0).to(device)
    poly_len = [sample["lane_polygon_len"]]
    ctx = [sample["context_str"]]

    with torch.no_grad():
        pred = model(x_in, vision_embs=vision, context_str=ctx,
                     lane_polygon_batch=polygon, lane_polygon_len=poly_len)

    pred_np = pred.squeeze(0).cpu().numpy().transpose(1, 0)
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
    plt.plot(gt_den[:, 0], gt_den[:, 1], 'go-', label='GT Future')
    plt.plot(pd_den[:, 0], pd_den[:, 1], 'ro-', label='Pred')
    plt.title(f"{prefix} idx={idx}")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid(True)
    plt.legend()

    filename = f"{prefix}_idx_{idx}.png"
    save_path = os.path.join(save_dir, filename)
    plt.savefig(save_path)
    plt.close()
    print(f"[{prefix}] Sample {idx} => Saved to {save_path}")

def visualize_candidates_sample(model, sample, device, idx=0,
                                prefix="test_sample", save_dir="test_visualization_candidates", num_candidates=20):
    # Test 시 후보 예측과 최종 후보(최소 RMSE)를 함께 시각화 (색상 다르게)
    os.makedirs(save_dir, exist_ok=True)
    # 데이터 추출
    in_traj = sample["traj_emb"]    # (T_in,2)
    out_traj = sample["target_traj"] # (T_out,2)
    min_x, max_x, min_y, max_y = sample["norm_stat"]

    x_in = in_traj.unsqueeze(0).permute(0, 2, 1).to(device)  # (1,2,T_in)
    vision = sample["vision_emb"].unsqueeze(0).to(device)
    polygon = sample["lane_polygon"].unsqueeze(0).to(device)
    poly_len = [sample["lane_polygon_len"]]
    ctx = [sample["context_str"]]

    # Denormalization 함수
    def denorm(traj):
        traj_den = traj.copy()
        traj_den[:, 0] = traj_den[:, 0] * (max_x - min_x) + min_x
        traj_den[:, 1] = traj_den[:, 1] * (max_y - min_y) + min_y
        return traj_den

    in_np = in_traj.cpu().numpy()
    in_den = denorm(in_np)
    gt_np = out_traj.cpu().numpy()
    gt_den = denorm(gt_np)

    candidate_preds = []
    candidate_rmse = []
    with torch.no_grad():
        for k in range(num_candidates):
            pred = model(x_in, vision_embs=vision, context_str=ctx,
                         lane_polygon_batch=polygon, lane_polygon_len=poly_len)
            pred_np = pred.squeeze(0).cpu().numpy().transpose(1, 0)  # (T_out,2)
            pred_den = denorm(pred_np)
            candidate_preds.append(pred_den)
            rmse = np.sqrt(np.mean((pred_den - gt_den)**2))
            candidate_rmse.append(rmse)
    candidate_preds = np.array(candidate_preds)  # (num_candidates, T_out,2)
    candidate_rmse = np.array(candidate_rmse)
    best_idx = int(np.argmin(candidate_rmse))
    best_pred = candidate_preds[best_idx]

    plt.figure(figsize=(8, 6))
    plt.plot(in_den[:, 0], in_den[:, 1], 'bo-', label='Past')
    plt.plot(gt_den[:, 0], gt_den[:, 1], 'go-', label='GT Future')
    # 각 후보 예측을 점선으로 그리기 (최적 후보 제외)
    for k in range(num_candidates):
        if k == best_idx:
            continue
        pred = candidate_preds[k]
        plt.plot(pred[:, 0], pred[:, 1], color='orange', linestyle='--', alpha=0.5, label='Candidate' if k==0 else None)
    # 최적 후보 예측을 굵은 빨간 선으로 표시
    plt.plot(best_pred[:, 0], best_pred[:, 1], 'r.-', linewidth=2, markersize=8, label='Best Candidate')
    plt.title(f"{prefix} idx={idx} | Best RMSE: {candidate_rmse[best_idx]:.2f}")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid(True)
    plt.legend()
    filename = f"{prefix}_idx_{idx}.png"
    save_path = os.path.join(save_dir, filename)
    plt.savefig(save_path)
    plt.close()
    print(f"[{prefix}] Sample {idx} => Saved to {save_path}")

def visualize_test_candidates_samples(model, dataset, device, num_samples=5,
                                      prefix="test_sample", save_dir="test_visualization_candidates", num_candidates=20):
    os.makedirs(save_dir, exist_ok=True)
    indices = random.sample(range(len(dataset)), min(num_samples, len(dataset)))
    for i, idx in enumerate(indices):
        sample = dataset[idx]
        visualize_candidates_sample(model, sample, device, idx=idx, prefix=prefix, save_dir=save_dir, num_candidates=num_candidates)

########################################
# 9. DDP 학습 함수 (Best Val Loss 체크포인트 포함)
########################################
def train_ddp(args):
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    dist.init_process_group("nccl", rank=local_rank, world_size=world_size)
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    # 전체 데이터셋 로드 및 7:2:1 분할
    with open(args["all_data_pkl"], 'rb') as f:
        all_data = pickle.load(f)
    train_data, val_data, test_data = split_all_data(all_data, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1)

    train_inputs, train_outputs = build_dataset_from_tracks_sliding(
        track_list=train_data,
        seq_len=args["seq_len"],
        out_len=args["out_len"],
        stride=args["stride"],
        max_step=args["max_step"],
        max_speed_diff=args["max_speed_diff"],
        image_width=args["image_width"],
        image_height=args["image_height"],
        downsample=args["downsample"]
    )
    train_dataset = MultiModalTrajectoryDataset(train_inputs, train_outputs,
                                                  max_polygon_points=args["max_polygon_points"])

    val_inputs, val_outputs = build_dataset_from_tracks_sliding(
        track_list=val_data,
        seq_len=args["seq_len"],
        out_len=args["out_len"],
        stride=args["stride"],
        max_step=args["max_step"],
        max_speed_diff=args["max_speed_diff"],
        image_width=args["image_width"],
        image_height=args["image_height"],
        downsample=args["downsample"]
    )
    val_dataset = MultiModalTrajectoryDataset(val_inputs, val_outputs,
                                                max_polygon_points=args["max_polygon_points"])

    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=local_rank, shuffle=True)
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=local_rank, shuffle=False)

    train_loader = DataLoader(train_dataset, batch_size=args["batch_size"],
                              sampler=train_sampler, collate_fn=custom_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=args["batch_size"],
                            sampler=val_sampler, collate_fn=custom_collate_fn)

    model = MultiModalTrajectoryModel(
        seq_len=args["seq_len"],
        out_len=args["out_len"],
        individual=True,
        feature_size=2,
        d_model=args["d_model"],
        lane_polygon_d_model=64,
        lane_polygon_nhead=4,
        lane_polygon_layers=2,
        max_polygon_points=args["max_polygon_points"],
        use_post_mlp=True,
        post_mlp_hidden_dim=64,
        ltsf_nhead=args["ltsf_nhead"],
        ltsf_dropout=args["ltsf_dropout"],
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
    ).to(device)

    ddp_model = pnl.DistributedDataParallel(
        model, device_ids=[local_rank],
        output_device=local_rank, find_unused_parameters=True
    )

    optimizer = torch.optim.AdamW(ddp_model.parameters(), lr=args["lr"], weight_decay=1e-4)

    epochs = args["epochs"]
    best_val_loss = float('inf')
    best_ckpt_path = "fix_seed_best_val_checkpoint.pt"

    for epoch in range(epochs):
        ddp_model.train()
        train_sampler.set_epoch(epoch)
        total_loss = 0.0

        for batch_data in train_loader:
            x = batch_data["traj_emb"].to(device)  # (B,2,T_in)
            y = batch_data["target_traj"].to(device)
            v = batch_data["vision_emb"].to(device)
            c = batch_data["context_str"]
            ns = batch_data["norm_stat"]
            p = batch_data["lane_polygon"].to(device)
            pl = batch_data["lane_polygon_len"]

            optimizer.zero_grad()
            ret = ddp_model(
                x, vision_embs=v, context_str=c,
                lane_polygon_batch=p, lane_polygon_len=pl,
                y=y, norm_stat=ns
            )
            if isinstance(ret, tuple):
                loss, _ = ret
            else:
                loss = torch.tensor(0.0, device=device)

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        ddp_model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_data in val_loader:
                x_v = batch_data["traj_emb"].to(device)
                y_v = batch_data["target_traj"].to(device)
                vv = batch_data["vision_emb"].to(device)
                cc = batch_data["context_str"]
                ns_ = batch_data["norm_stat"]
                pp = batch_data["lane_polygon"].to(device)
                pll = batch_data["lane_polygon_len"]

                r_v = ddp_model(
                    x_v, vision_embs=vv, context_str=cc,
                    lane_polygon_batch=pp, lane_polygon_len=pll,
                    y=y_v, norm_stat=ns_
                )
                if isinstance(r_v, tuple):
                    l_v, _ = r_v
                else:
                    l_v = torch.tensor(0.0, device=device)
                val_loss += l_v.item()

        avg_val_loss = val_loss / len(val_loader)

        if local_rank == 0:
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(ddp_model.module.state_dict(), best_ckpt_path)
                print(f"Best Val Loss updated: {best_val_loss:.4f} | Checkpoint saved to {best_ckpt_path}")

            print(f"[Epoch {epoch+1}/{epochs}] Train Loss: {avg_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

            rand_idx = random.randint(0, len(val_dataset)-1)
            sample_val = val_dataset[rand_idx]
            visualize_one_sample(ddp_model.module, sample_val, device, idx=epoch,
                                 prefix="val_sample", save_dir="seed_fix_train_visualization_val")

    if local_rank == 0 and os.path.exists(best_ckpt_path):
        best_state = torch.load(best_ckpt_path)
        ddp_model.module.load_state_dict(best_state)
        print(f"Loaded best model state from {best_ckpt_path}")

    if local_rank == 0:
        test_inputs, test_outputs = build_dataset_from_tracks_sliding(
            track_list=test_data,
            seq_len=args["seq_len"],
            out_len=args["out_len"],
            stride=args["stride"],
            max_step=args["max_step"],
            max_speed_diff=args["max_speed_diff"],
            image_width=args["image_width"],
            image_height=args["image_height"],
            downsample=args["downsample"]
        )
        test_dataset = MultiModalTrajectoryDataset(test_inputs, test_outputs,
                                                    max_polygon_points=args["max_polygon_points"])
        test_sampler = DistributedSampler(test_dataset, num_replicas=world_size, rank=local_rank, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=args["batch_size"],
                                 sampler=test_sampler, collate_fn=custom_collate_fn)
        # 테스트 시 확률적 예측을 위해 dropout 활성화 (train 모드 유지)
        ddp_model.train()
        total_min_ade = 0.0
        total_min_fde = 0.0
        total_min_rmse = 0.0
        total_samples = 0
        num_candidates = 6  # 후보 예측 수

        with torch.no_grad():
            for batch_data in test_loader:
                x_t = batch_data["traj_emb"].to(device)
                y_t = batch_data["target_traj"].to(device)
                v_t = batch_data["vision_emb"].to(device)
                c_t = batch_data["context_str"]
                n_t = batch_data["norm_stat"]
                p_t = batch_data["lane_polygon"].to(device)
                pl_t = batch_data["lane_polygon_len"]

                B_ = x_t.size(0)
                # norm statistics로 denormalization 위한 값들
                min_x = torch.tensor([ns[0] for ns in n_t], device=device).view(B_, 1, 1)
                max_x = torch.tensor([ns[1] for ns in n_t], device=device).view(B_, 1, 1)
                min_y = torch.tensor([ns[2] for ns in n_t], device=device).view(B_, 1, 1)
                max_y = torch.tensor([ns[3] for ns in n_t], device=device).view(B_, 1, 1)
                rx = max_x - min_x
                ry = max_y - min_y

                # GT denormalization
                y_den = y_t.clone()
                y_den[:, 0, :] = y_den[:, 0, :] * rx.squeeze(2) + min_x.squeeze(2)
                y_den[:, 1, :] = y_den[:, 1, :] * ry.squeeze(2) + min_y.squeeze(2)

                all_ade = []
                all_fde = []
                all_rmse = []
                for k in range(num_candidates):
                    pred_t = ddp_model(
                        x_t, vision_embs=v_t, context_str=c_t,
                        lane_polygon_batch=p_t, lane_polygon_len=pl_t,
                        y=None, norm_stat=None
                    )
                    pred_den = pred_t.clone()
                    pred_den[:, 0, :] = pred_den[:, 0, :] * rx.squeeze(2) + min_x.squeeze(2)
                    pred_den[:, 1, :] = pred_den[:, 1, :] * ry.squeeze(2) + min_y.squeeze(2)
                    # Euclidean distance per time step
                    dists = torch.sqrt(((pred_den - y_den) ** 2).sum(dim=1))  # (B, T_out)
                    ade_k = dists.mean(dim=1)  # (B,)
                    fde_k = dists[:, -1]       # (B,)
                    rmse_k = torch.sqrt(torch.mean((pred_den - y_den)**2, dim=[1,2]))  # (B,)
                    all_ade.append(ade_k.unsqueeze(1))
                    all_fde.append(fde_k.unsqueeze(1))
                    all_rmse.append(rmse_k.unsqueeze(1))
                all_ade = torch.cat(all_ade, dim=1)  # (B, num_candidates)
                all_fde = torch.cat(all_fde, dim=1)
                all_rmse = torch.cat(all_rmse, dim=1)
                min_ade, _ = torch.min(all_ade, dim=1)
                min_fde, _ = torch.min(all_fde, dim=1)
                min_rmse, _ = torch.min(all_rmse, dim=1)
                total_min_ade += min_ade.sum().item()
                total_min_fde += min_fde.sum().item()
                total_min_rmse += min_rmse.sum().item()
                total_samples += B_

        avg_min_ade = total_min_ade / total_samples if total_samples > 0 else 0.0
        avg_min_fde = total_min_fde / total_samples if total_samples > 0 else 0.0
        avg_min_rmse = total_min_rmse / total_samples if total_samples > 0 else 0.0
        print(f"[Test] minADE: {avg_min_ade:.4f}, minFDE: {avg_min_fde:.4f}, minRMSE: {avg_min_rmse:.4f}")

        # 후보 예측 및 최종 후보 시각화
        visualize_test_candidates_samples(ddp_model.module, test_dataset, device,
                                          num_samples=20, prefix="test_sample", save_dir="seed_fix_test_visualization_candidates", num_candidates=num_candidates)

def main():
    args = {
        "all_data_pkl": "/home/user/MLLM/data/all_data.pkl",  # 전체 데이터셋 파일 경로 (pickle 형식)
        "seq_len": 24,
        "out_len": 24,
        "batch_size": 16,
        "epochs": 3,
        "lr": 5e-4,
        "stride": 6,
        "downsample": 5,
        "max_step": 50.0,
        "max_speed_diff": 30.0,
        "image_width": 3840,
        "image_height": 2160,
        "max_polygon_points": 64,
        # LLaMA
        "base_model_name": "meta-llama/Llama-3.2-1B",
        "use_lora": True,
        "lora_r": 8,
        "lora_alpha": 32,
        "lora_dropout": 0.1,
        "vision_dim": 512,
        # Q-Former
        "qformer_hidden_size": 768,
        "qformer_nhead": 8,
        "qformer_enc_layers": 2,
        "qformer_dec_layers": 2,
        "qformer_num_query_tokens": 16,
        # LTSF
        "ltsf_nhead": 2,
        "ltsf_dropout": 0.1,
        # transformer 내부 embedding 차원
        "d_model": 64,
    }
    train_ddp(args)

if __name__ == "__main__":
    main()
