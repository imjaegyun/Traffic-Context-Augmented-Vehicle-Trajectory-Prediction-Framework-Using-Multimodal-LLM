import os
import re
import math
import pickle
import random
import numpy as np

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.nn.parallel as pnl
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, TaskType, get_peft_model
from contextlib import nullcontext


#################################################
# 0. 유틸 함수 및 데이터셋
#################################################
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


def split_all_data(all_data, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1):
    random.shuffle(all_data)
    N = len(all_data)
    train_end = int(N * train_ratio)
    val_end = train_end + int(N * val_ratio)
    return all_data[:train_end], all_data[train_end:val_end], all_data[val_end:]


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
        if lane_label == "R2L":
            if np.any(x_vals[1:] > x_vals[:-1]):
                return True
        elif lane_label == "L2R":
            if np.any(x_vals[1:] < x_vals[:-1]):
                return True
    return False


#################################################
# extract() => device mismatch 해결
#################################################
def extract(a, t, x_shape):
    b = t.shape[0]
    a_t = torch.tensor(a, dtype=torch.float32, device=t.device)
    out = a_t.gather(0, t)  # shape=[B]
    return out.reshape(b, 1, 1).expand(b, x_shape[1], x_shape[2])


#################################################
# build_dataset_from_tracks_sliding
#################################################
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
        if vision_emb is not None:
            if isinstance(vision_emb, torch.Tensor):
                vision_emb = vision_emb.cpu()
            vision_emb = vision_emb[::downsample]
            emb_np = vision_emb if isinstance(vision_emb, np.ndarray) else vision_emb.numpy()
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

        if is_trajectory_abnormal(raw_traj, lane_label=lane_direction,
                                  max_step=max_step, max_speed_diff=max_speed_diff):
            continue
        N = raw_traj.shape[0]
        if N < (seq_len + out_len):
            continue

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
            if range_x_ < 1e-6 or range_y_ < 1e-6:
                continue

            in_norm = (in_traj - [min_x_, min_y_]) / [range_x_, range_y_]
            out_norm = (out_traj - [min_x_, min_y_]) / [range_x_, range_y_]
            in_traj_t = torch.tensor(in_norm, dtype=torch.float32)
            out_traj_t = torch.tensor(out_norm, dtype=torch.float32)

            if vision_emb is not None:
                if isinstance(vision_emb, np.ndarray):
                    vision_tensor = torch.from_numpy(vision_emb)
                else:
                    vision_tensor = vision_emb
                in_vision = vision_tensor[start:start+seq_len]
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
                "1. **Road Geometry and Infrastructure**\n"
                "2. **Surrounding Environment**\n"
                "3. **Vehicle Dynamics**\n"
                "4. **Neighboring Entities**\n"
                "5. **Safety and Hazard Indicators**\n\n"
                "Provide your comprehensive answer as a natural language paragraph."
            )
            answer_text = original_ctx

            if tokenizer is not None:
                prompt_enc = tokenizer(prompt_text, truncation=True, max_length=max_length,
                                       return_tensors="pt", add_special_tokens=False)
                answer_enc = tokenizer(answer_text, truncation=True, max_length=max_length,
                                       return_tensors="pt", add_special_tokens=False)
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
                input_ids = torch.zeros(1, 1, dtype=torch.long)
                attention_mask = torch.ones(1, 1, dtype=torch.long)
                labels = torch.zeros(1, 1, dtype=torch.long)

            sample_input = {
                "trajectory_embeddings": in_traj_t,
                "vision_embeddings": in_vision,
                "context_str": prompt_text,
                "answer_str": answer_text,
                "norm_stat": (min_x_, max_x_, min_y_, max_y_),
                "lane_roi": item.get("lane_roi", None),
                "input_ids": input_ids.squeeze(0),
                "attention_mask": attention_mask.squeeze(0),
                "labels": labels.squeeze(0)
            }
            inputs_list.append(sample_input)
            outputs_list.append(out_traj_t)

    return inputs_list, outputs_list


class MultiModalTrajectoryDataset(Dataset):
    def __init__(self, inputs_list, outputs_list, max_polygon_points=64):
        self.inputs_list = inputs_list
        self.outputs_list = outputs_list
        self.max_polygon_points = max_polygon_points
        assert len(self.inputs_list) == len(self.outputs_list)

    def __len__(self):
        return len(self.inputs_list)

    def __getitem__(self, idx):
        sample_in = self.inputs_list[idx]
        sample_out = self.outputs_list[idx]
        # polygon dummy
        polygon = np.zeros((self.max_polygon_points, 2), dtype=np.float32)
        sample = {
            "traj_emb": sample_in["trajectory_embeddings"],
            "vision_emb": sample_in["vision_embeddings"],
            "context_str": sample_in["context_str"],
            "answer_str": sample_in["answer_str"],
            "norm_stat": sample_in["norm_stat"],
            "target_traj": sample_out,
            "lane_polygon": torch.tensor(polygon, dtype=torch.float32),
            "lane_polygon_len": 0,
            "input_ids": sample_in["input_ids"],
            "attention_mask": sample_in["attention_mask"],
            "labels": sample_in["labels"]
        }
        return sample


def custom_collate_fn(batch):
    from torch.nn.utils.rnn import pad_sequence
    x_in_list = [b["traj_emb"] for b in batch]
    y_out_list = [b["target_traj"] for b in batch]
    v_in_list = [b["vision_emb"] for b in batch]

    x_3d = torch.stack([t.transpose(0, 1) for t in x_in_list], dim=0)  # [B,2,T_in]
    y_3d = torch.stack([t.transpose(0, 1) for t in y_out_list], dim=0)  # [B,2,T_out]
    v_3d = torch.stack(v_in_list, dim=0)  # [B, seq_len, feat_dim]

    poly_list = [b["lane_polygon"] for b in batch]
    poly_len_list = [b["lane_polygon_len"] for b in batch]
    poly_3d = torch.stack(poly_list, dim=0)

    norm_stats = [b["norm_stat"] for b in batch]
    context_strs = [b["context_str"] for b in batch]
    ans_strs = [b["answer_str"] for b in batch]

    in_ids_list = [b["input_ids"] for b in batch]
    attn_mask_list = [b["attention_mask"] for b in batch]
    labs_list = [b["labels"] for b in batch]

    from torch.nn.utils.rnn import pad_sequence
    input_ids_pad = pad_sequence(in_ids_list, batch_first=True, padding_value=0)
    attn_mask_pad = pad_sequence(attn_mask_list, batch_first=True, padding_value=0)
    labs_pad = pad_sequence(labs_list, batch_first=True, padding_value=-100)

    return {
        "traj_emb": x_3d,
        "target_traj": y_3d,
        "vision_emb": v_3d,
        "lane_polygon": poly_3d,
        "lane_polygon_len": poly_len_list,
        "norm_stat": norm_stats,
        "context_str": context_strs,
        "answer_str": ans_strs,
        "input_ids": input_ids_pad,
        "attention_mask": attn_mask_pad,
        "labels": labs_pad
    }


#################################################
# LLM 기반 '궤적 예측' 모델들
#################################################

class LanePolygonEncoder(nn.Module):
    def __init__(self, d_model=64, nhead=4, num_layers=2, max_points=64):
        super().__init__()
        self.d_model = d_model
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
        out = self.encoder(x, src_key_padding_mask=pad_mask)
        emb_list = []
        for i in range(B):
            valid_len = poly_len_list[i]
            if valid_len > 0:
                emb_i = out[i, :valid_len]
                emb_mean = emb_i.mean(dim=0)
            else:
                emb_mean = torch.zeros(self.d_model, device=x.device)
            emb_list.append(emb_mean)
        return torch.stack(emb_list, dim=0)


class BlipQFormer(nn.Module):
    def __init__(self,
                 vision_dim=512,
                 hidden_size=768,
                 nhead=8,
                 num_encoder_layers=4,
                 num_decoder_layers=4,
                 num_query_tokens=16):
        super().__init__()
        self.vision_proj = nn.Linear(vision_dim, hidden_size)
        enc_layer = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=nhead, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_encoder_layers)
        self.query_tokens = nn.Parameter(torch.randn(num_query_tokens, hidden_size))
        dec_layer = nn.TransformerDecoderLayer(d_model=hidden_size, nhead=nhead, batch_first=True)
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=num_decoder_layers)

    def forward(self, vision_embs):
        B, T, _ = vision_embs.shape
        x = self.vision_proj(vision_embs)
        enc_out = self.encoder(x)
        query = self.query_tokens.unsqueeze(0).expand(B, -1, -1)
        dec_out = self.decoder(query, enc_out)
        return dec_out


class LlamaWithCrossAttnPEFT(nn.Module):
    def __init__(self, base_model_name, use_lora=True,
                 lora_r=8, lora_alpha=32, lora_dropout=0.1):
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
                target_modules=["q_proj", "v_proj"]
            )
            self.llama_model = get_peft_model(self.llama_model, peft_config)
            for n, p in self.llama_model.named_parameters():
                if "lora_" in n:
                    p.requires_grad = True
                else:
                    p.requires_grad = False

        self.config = self.llama_model.config
        self.hidden_size = self.config.hidden_size

    def forward(self, inputs_embeds, attention_mask, labels=None, output_hidden_states=False):
        out = self.llama_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
            output_hidden_states=output_hidden_states,
            return_dict=True
        )
        return out


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
        self.q_hidden_size = q_hidden_size
        if self.llama_hidden_size != self.q_hidden_size:
            self.q_proj = nn.Linear(q_hidden_size, self.llama_hidden_size)
        else:
            self.q_proj = nn.Identity()

        self.vision_modality_embed = nn.Parameter(torch.randn(1, 1, self.llama_hidden_size))
        self.text_modality_embed = nn.Parameter(torch.randn(1, 1, self.llama_hidden_size))

        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def forward(self, vision_embs, context_str,
                input_ids=None, attention_mask=None, labels=None):
        device = vision_embs.device
        B = vision_embs.size(0)
        image_tokens = self.qformer(vision_embs)
        image_tokens = self.q_proj(image_tokens) + self.vision_modality_embed

        if (input_ids is not None) and (attention_mask is not None):
            text_embeds = self.llama_wrapper.llama_model.get_input_embeddings()(input_ids)
            text_embeds = text_embeds + self.text_modality_embed
            fused_embeds = torch.cat([image_tokens, text_embeds], dim=1)

            img_mask = torch.ones((B, image_tokens.size(1)), dtype=attention_mask.dtype, device=device)
            fused_mask = torch.cat([img_mask, attention_mask], dim=1)

            if labels is not None:
                fused_labels = torch.full(
                    (B, fused_mask.size(1)), -100,
                    dtype=labels.dtype, device=device
                )
                fused_labels[:, image_tokens.size(1):] = labels
            else:
                fused_labels = None

            out = self.llama_wrapper(
                inputs_embeds=fused_embeds,
                attention_mask=fused_mask,
                labels=fused_labels,
                output_hidden_states=True
            )
            final_hidden = out.hidden_states[-1]
            return final_hidden, image_tokens.size(1)

        else:
            # inference path
            text_inputs = self.tokenizer(context_str, return_tensors="pt",
                                         padding=True, truncation=True).to(device)
            text_embeds = self.llama_wrapper.llama_model.get_input_embeddings()(text_inputs["input_ids"])
            text_embeds = text_embeds + self.text_modality_embed
            fused_embeds = torch.cat([image_tokens, text_embeds], dim=1)

            img_mask = torch.ones((B, image_tokens.size(1)), dtype=text_inputs["attention_mask"].dtype, device=device)
            fused_mask = torch.cat([img_mask, text_inputs["attention_mask"]], dim=1)

            out = self.llama_wrapper(
                inputs_embeds=fused_embeds,
                attention_mask=fused_mask,
                labels=None,
                output_hidden_states=True
            )
            final_hidden = out.hidden_states[-1]
            return final_hidden, image_tokens.size(1)


class SelfAttentionBlock(nn.Module):
    def __init__(self, embed_dim, nhead=1, dropout_rate=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.mha = nn.MultiheadAttention(embed_dim, nhead, dropout=dropout_rate)
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
        B, E, T = x.shape
        x_perm = x.permute(2, 0, 1)
        x_norm = self.norm1(x_perm)
        attn_out, _ = self.mha(x_norm, x_norm, x_norm)
        res1 = x_norm + self.dropout1(attn_out)
        res1 = self.norm2(res1)
        ffn_out = self.ffn(res1)
        out = res1 + self.dropout2(ffn_out)
        return out.permute(1, 2, 0)


class LTSF_NLinearEncoder(nn.Module):
    def __init__(self, window_size, individual, d_model):
        super().__init__()
        self.window_size = window_size
        self.individual = individual
        self.channels = d_model
        if individual:
            self.linears = nn.ModuleList([nn.Linear(window_size, window_size) for _ in range(d_model)])
        else:
            self.linear = nn.Linear(window_size, window_size)

    def forward(self, x):
        B, C, T = x.shape
        last = x[:, :, -1:].clone()
        x_sub = x - last
        if self.individual:
            out_list = []
            for i in range(C):
                out_i = self.linears[i](x_sub[:, i, :])
                out_list.append(out_i.unsqueeze(1))
            enc = torch.cat(out_list, dim=1)
        else:
            out_ = self.linear(x_sub.reshape(B * C, T)).reshape(B, C, T)
            enc = out_
        enc = enc + last
        return enc


class LTSF_NLinearDecoder(nn.Module):
    def __init__(self, window_size, forecast_size, individual, d_model,
                 polygon_embed_dim=64, use_post_mlp=True, post_mlp_hidden_dim=64,
                 post_mlp_output_dim=None, dropout_rate=0.1,
                 cross_dim=768, cross_nhead=2, output_feature_dim=2):
        super().__init__()
        self.window_size = window_size
        self.forecast_size = forecast_size
        self.individual = individual
        self.channels = d_model
        if individual:
            self.linears = nn.ModuleList([nn.Linear(window_size, forecast_size) for _ in range(d_model)])
        else:
            self.linear = nn.Linear(window_size, forecast_size)

        self.lane_fc = nn.Linear(polygon_embed_dim, self.channels * self.forecast_size)
        self.use_post_mlp = use_post_mlp
        if post_mlp_output_dim is None:
            post_mlp_output_dim = self.channels * self.forecast_size
        if use_post_mlp:
            self.post_mlp = nn.Sequential(
                nn.Linear(self.channels * self.forecast_size, post_mlp_hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(post_mlp_hidden_dim, post_mlp_output_dim)
            )

        self.cross_attn = nn.MultiheadAttention(cross_dim, cross_nhead, batch_first=False)
        self.dec_proj = nn.Linear(d_model, cross_dim)
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
        last = encoded[:, :, -1:].clone()
        x_sub = encoded - last
        if self.individual:
            out_list = []
            for i in range(C):
                dec_i = self.linears[i](x_sub[:, i, :])
                out_list.append(dec_i.unsqueeze(1))
            dec = torch.cat(out_list, dim=1)
        else:
            out_ = self.linear(x_sub.reshape(B * C, T)).reshape(B, C, self.forecast_size)
            dec = out_
        dec = dec + last.repeat(1, 1, self.forecast_size)

        lane_adj = self.lane_fc(lane_polygon_emb).reshape(B, C, self.forecast_size)
        dec = dec + lane_adj

        if self.use_post_mlp:
            flat_dec = dec.reshape(B, -1)
            post_out = self.post_mlp(flat_dec).reshape(B, C, self.forecast_size)
            dec = post_out

        dec_t = dec.permute(0, 2, 1)
        proj_dec = self.dec_proj(dec_t)
        query = proj_dec.transpose(0, 1)
        key = final_hidden.transpose(0, 1)
        val = final_hidden.transpose(0, 1)
        cross_out = self.cross_attn(query, key, val)[0].transpose(0, 1)
        cross_to_d = self.dec_unproj(cross_out)
        fused = dec_t + cross_to_d
        fused = self.fusion_layer(fused)
        out = fused.permute(0, 2, 1)
        out = out.permute(0, 2, 1)
        out = self.out_proj(out)
        out = out.permute(0, 2, 1)
        return out


class TransformerLTSF(nn.Module):
    def __init__(self, seq_len, out_len, individual, feature_size,
                 d_model, polygon_embed_dim=64, use_post_mlp=True,
                 post_mlp_hidden_dim=64, post_mlp_output_dim=None,
                 nhead=1, dropout_rate=0.1,
                 cross_dim=768, cross_nhead=2,
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
        poly_emb = self.lane_polygon_encoder(lane_polygon_batch, lane_polygon_len)
        final_hidden, _ = self.mllm(
            vision_embs,
            context_str,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        dec = self.ltsf(x, poly_emb, final_hidden)

        # last frame concat
        last_in = x[:, :, -1:].clone()
        dec = dec + last_in.repeat(1, 1, self.out_len)

        if (y is not None) and (norm_stat is not None):
            B = x.size(0)
            min_x = torch.tensor([ns[0] for ns in norm_stat], device=device).view(B, 1, 1)
            max_x = torch.tensor([ns[1] for ns in norm_stat], device=device).view(B, 1, 1)
            min_y = torch.tensor([ns[2] for ns in norm_stat], device=device).view(B, 1, 1)
            max_y = torch.tensor([ns[3] for ns in norm_stat], device=device).view(B, 1, 1)
            rx = max_x - min_x
            ry = max_y - min_y
            dec_den = dec.clone()
            gt_den = y.clone()
            dec_den[:, 0, :] = dec_den[:, 0, :] * rx.squeeze(-1) + min_x.squeeze(-1)
            dec_den[:, 1, :] = dec_den[:, 1, :] * ry.squeeze(-1) + min_y.squeeze(-1)
            gt_den[:, 0, :] = gt_den[:, 0, :] * rx.squeeze(-1) + min_x.squeeze(-1)
            gt_den[:, 1, :] = gt_den[:, 1, :] * ry.squeeze(-1) + min_y.squeeze(-1)

            loss_x = nn.MSELoss()(dec_den[:, 0, :], gt_den[:, 0, :])
            loss_y = nn.MSELoss()(dec_den[:, 1, :], gt_den[:, 1, :])
            loss = loss_x + loss_y
            return loss, dec
        else:
            return dec


#################################################
# ★ 시계열 플롯 (for predict mode)
#################################################
def visualize_one_sample(model, sample, device,
                         idx=0, prefix="test_sample",
                         save_dir="time_series_plots"):
    os.makedirs(save_dir, exist_ok=True)
    model.eval()

    in_traj = sample["traj_emb"]
    out_traj = sample["target_traj"]
    min_x, max_x, min_y, max_y = sample["norm_stat"]

    x_in = in_traj.unsqueeze(0).permute(0, 2, 1).to(device)
    v_in = sample["vision_emb"].unsqueeze(0).to(device)
    poly = sample["lane_polygon"].unsqueeze(0).to(device)
    poly_len = [sample["lane_polygon_len"]]
    ctx = [sample["context_str"]]

    i_ids = sample["input_ids"].unsqueeze(0).to(device)
    a_m = sample["attention_mask"].unsqueeze(0).to(device)
    labs = sample["labels"].unsqueeze(0).to(device)

    with torch.no_grad():
        ret = model(
            x_in, v_in, ctx, poly, poly_len,
            y=out_traj.unsqueeze(0).permute(0, 2, 1).to(device),
            norm_stat=[(min_x, max_x, min_y, max_y)],
            input_ids=i_ids,
            attention_mask=a_m,
            labels=labs
        )
    if isinstance(ret, tuple):
        pred = ret[1]  # shape [1,2,T_out]
    else:
        pred = ret

    pred_np = pred.squeeze(0).cpu().numpy()
    in_np = in_traj.cpu().numpy()
    gt_np = out_traj.cpu().numpy()

    rx = (max_x - min_x)
    ry = (max_y - min_y)

    # Denormalize
    in_den = in_np.copy()
    gt_den = gt_np.copy()
    pd_den = pred_np.copy().T

    in_den[:, 0] = in_den[:, 0] * rx + min_x
    in_den[:, 1] = in_den[:, 1] * ry + min_y
    gt_den[:, 0] = gt_den[:, 0] * rx + min_x
    gt_den[:, 1] = gt_den[:, 1] * ry + min_y
    pd_den[:, 0] = pd_den[:, 0] * rx + min_x
    pd_den[:, 1] = pd_den[:, 1] * ry + min_y

    T_in = in_den.shape[0]
    T_out = gt_den.shape[0]
    t_past = np.arange(T_in)
    t_future = np.arange(T_in, T_in + T_out)

    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

    # X
    ax1 = axes[0]
    ax1.plot(t_past, in_den[:, 0], 'bo-', label='Past(3s) X')
    ax1.plot(t_future, gt_den[:, 0], 'go-', label='GT-Future(3s) X')
    ax1.plot(t_future, pd_den[:, 0], 'ro-', label='Pred-Future(3s) X')
    ax1.grid(True)
    ax1.set_ylabel("X coord")
    ax1.legend()

    # Y
    ax2 = axes[1]
    ax2.plot(t_past, in_den[:, 1], 'bo-', label='Past(3s) Y')
    ax2.plot(t_future, gt_den[:, 1], 'go-', label='GT-Future(3s) Y')
    ax2.plot(t_future, pd_den[:, 1], 'ro-', label='Pred-Future(3s) Y')
    ax2.grid(True)
    ax2.set_xlabel("Time index (frame)")
    ax2.set_ylabel("Y coord")
    ax2.legend()

    fig.suptitle(f"{prefix} idx={idx}")
    save_path = os.path.join(save_dir, f"{prefix}_idx_{idx}_timeseries.png")
    plt.savefig(save_path)
    plt.close()
    print(f"[{prefix}] sample {idx} => timeseries saved: {save_path}")


def visualize_test_samples(model, dataset, device, num_samples=5,
                           prefix="test_sample", save_dir="time_series_plots"):
    os.makedirs(save_dir, exist_ok=True)
    idxs = random.sample(range(len(dataset)), min(num_samples, len(dataset)))
    for i, idx in enumerate(idxs):
        sample = dataset[idx]
        visualize_one_sample(model, sample, device,
                             idx=idx, prefix=prefix,
                             save_dir=save_dir)


#################################################
# (A) predict_trajectory
#################################################
def predict_trajectory(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with open(args["all_data_pkl"], 'rb') as f:
        all_data = pickle.load(f)
    all_data = check_data_sanity(all_data)
    _, _, test_data = split_all_data(all_data, 0.7, 0.2, 0.1)

    base_tokenizer = AutoTokenizer.from_pretrained(args["base_model_name"])
    if base_tokenizer.pad_token is None:
        base_tokenizer.pad_token = base_tokenizer.eos_token
    test_in, test_out = build_dataset_from_tracks_sliding(
        test_data,
        seq_len=args["seq_len"],
        out_len=args["out_len"],
        stride=args["stride"],
        max_step=args["max_step"],
        max_speed_diff=args["max_speed_diff"],
        downsample=args["downsample"],
        tokenizer=base_tokenizer,
        max_length=512
    )
    test_dataset = MultiModalTrajectoryDataset(test_in, test_out)
    test_loader = DataLoader(test_dataset, batch_size=args["batch_size"],
                             shuffle=False, collate_fn=custom_collate_fn)

    model = MultiModalTrajectoryModel(
        seq_len=args["seq_len"],
        out_len=args["out_len"],
        individual=True,
        feature_size=2,
        d_model=args["d_model"],
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
        ltsf_dropout=args["ltsf_dropout"]
    ).to(device)

    if os.path.exists(args["inference_ckpt"]):
        model.load_state_dict(torch.load(args["inference_ckpt"], map_location=device))
        print(f"[Predict] load ckpt: {args['inference_ckpt']}")
    else:
        print(f"[Predict] no ckpt found at {args['inference_ckpt']}")

    model.eval()
    total_ade = 0.0
    total_fde = 0.0
    total_samp = 0
    with torch.no_grad():
        for batch_data in test_loader:
            x = batch_data["traj_emb"].to(device)
            y = batch_data["target_traj"].to(device)
            v = batch_data["vision_emb"].to(device)
            c = batch_data["context_str"]
            ns = batch_data["norm_stat"]
            p = batch_data["lane_polygon"].to(device)
            pl = batch_data["lane_polygon_len"]
            i_ids = batch_data["input_ids"].to(device)
            a_m = batch_data["attention_mask"].to(device)
            labs = batch_data["labels"].to(device)

            out = model(x, v, c, p, pl,
                        y=None, norm_stat=None,
                        input_ids=i_ids,
                        attention_mask=a_m,
                        labels=None)
            if isinstance(out, tuple):
                out = out[1]

            B_ = x.size(0)
            pred_den = out.clone()
            y_den = y.clone()

            min_x = torch.tensor([t[0] for t in ns], device=device).view(B_, 1, 1)
            max_x = torch.tensor([t[1] for t in ns], device=device).view(B_, 1, 1)
            min_y = torch.tensor([t[2] for t in ns], device=device).view(B_, 1, 1)
            max_y = torch.tensor([t[3] for t in ns], device=device).view(B_, 1, 1)
            rx = max_x - min_x
            ry = max_y - min_y

            pred_den[:, 0, :] = pred_den[:, 0, :] * rx.squeeze(2) + min_x.squeeze(2)
            pred_den[:, 1, :] = pred_den[:, 1, :] * ry.squeeze(2) + min_y.squeeze(2)
            y_den[:, 0, :] = y_den[:, 0, :] * rx.squeeze(2) + min_x.squeeze(2)
            y_den[:, 1, :] = y_den[:, 1, :] * ry.squeeze(2) + min_y.squeeze(2)

            ade_batch = torch.sqrt(((pred_den - y_den)**2).sum(dim=1)).mean(dim=1)
            fde_batch = torch.sqrt(((pred_den - y_den)**2).sum(dim=1))[:, -1]
            total_ade += ade_batch.sum().item()
            total_fde += fde_batch.sum().item()
            total_samp += B_

    if total_samp > 0:
        avg_ade = total_ade / total_samp
        avg_fde = total_fde / total_samp
    else:
        avg_ade = 0.0
        avg_fde = 0.0

    print(f"[Predict] TestSamp={total_samp}, ADE={avg_ade:.4f}, FDE={avg_fde:.4f}")

    visualize_test_samples(model, test_dataset, device,
                           num_samples=5, prefix="predict_sample",
                           save_dir="time_series_plots")


#################################################
# (B) Diffusion (DDPM + CFG + EMA + AMP)
#################################################
class EMA:
    def __init__(self, model, decay=0.9999):
        self.decay = decay
        self.model_params = {}
        for n, p in model.named_parameters():
            if p.requires_grad:
                self.model_params[n] = p.data.clone()

    def update(self, model):
        with torch.no_grad():
            for n, p in model.named_parameters():
                if p.requires_grad and n in self.model_params:
                    self.model_params[n] = self.decay * self.model_params[n] + (1.0 - self.decay) * p.data

    def apply_shadow(self, model):
        for n, p in model.named_parameters():
            if n in self.model_params:
                p.data = self.model_params[n].clone()


class ConditionalUNet(nn.Module):
    """
    1D UNet with cross-attn & CFG, designed for DDPM.
    in_channels=2 => 2D trajectory
    """
    def __init__(self, in_channels=2, cond_dim=128,
                 base_channels=64, n_layers=3):
        super().__init__()
        self.encoders = nn.ModuleList()
        ch = in_channels
        for i in range(n_layers):
            self.encoders.append(
                nn.Sequential(
                    nn.Conv1d(ch, base_channels, 3, 1, 1),
                    nn.ReLU(),
                    nn.Conv1d(base_channels, base_channels, 3, 1, 1),
                    nn.ReLU()
                )
            )
            ch = base_channels
        self.mid = nn.Sequential(
            nn.Conv1d(base_channels, base_channels, 3, 1, 1),
            nn.ReLU()
        )
        self.decoders = nn.ModuleList()
        for i in range(n_layers):
            self.decoders.append(
                nn.Sequential(
                    nn.Conv1d(base_channels * 2, base_channels, 3, 1, 1),
                    nn.ReLU(),
                    nn.Conv1d(base_channels, base_channels, 3, 1, 1),
                    nn.ReLU()
                )
            )
        self.out_conv = nn.Conv1d(base_channels, in_channels, 3, 1, 1)

        self.cond_proj = nn.Linear(cond_dim, base_channels)
        self.cross_attn = nn.MultiheadAttention(base_channels, 1, batch_first=False)

    def forward(self, x, cond_emb, cfg_scale=3.0):
        """
        x: [B, in_channels=2, T]
        cond_emb: [B, cond_dim], half are uncond, half are cond
        We do classifier-free guidance.
        """
        B, C, T = x.shape
        half = B // 2
        x_uncond = x[:half]
        x_cond = x[half:]
        c_uncond = cond_emb[:half]
        c_cond = cond_emb[half:]

        def forward_single(x_, c_):
            h = x_
            cond_feat = self.cond_proj(c_).unsqueeze(-1)  # [b, base_channels, 1]
            feats = []
            for enc in self.encoders:
                h = enc(h)
                feats.append(h)
            h = self.mid(h)

            # cross-attn on entire sequence
            h_perm = h.permute(2, 0, 1)
            c_perm = cond_feat.permute(2, 0, 1)
            attn_out, _ = self.cross_attn(h_perm, c_perm, c_perm)
            h = h_perm + attn_out
            h = h.permute(1, 2, 0)

            for i, dec in enumerate(self.decoders):
                enc_feat = feats[-(i + 1)]
                h = torch.cat([h, enc_feat], dim=1)
                h = dec(h)
            out = self.out_conv(h)
            return out

        out_uncond = forward_single(x_uncond, c_uncond)
        out_cond = forward_single(x_cond, c_cond)
        # classifier-free guidance
        out = out_uncond + cfg_scale * (out_cond - out_uncond)
        return torch.cat([out_uncond, out], dim=0)


def make_beta_schedule(schedule_type="linear", n_timestep=1000, start=1e-4, end=0.02):
    if schedule_type == "linear":
        betas = np.linspace(start, end, n_timestep, dtype=np.float64)
    elif schedule_type == "cosine":
        steps = np.arange(n_timestep + 1, dtype=np.float64)
        alphas_cum = np.cos(((steps / n_timestep) + 0.008) / 1.008 * math.pi * 0.5)**2
        alphas_cum = alphas_cum / alphas_cum[0]
        betas = 1 - (alphas_cum[1:] / alphas_cum[:-1])
    else:
        raise ValueError("unknown schedule_type")
    return betas


def generate_trajectory(args):
    """
    Training + sampling with a more standard DDPM approach (with CFG, EMA, AMP).
    """
    import torch.nn.utils as nn_utils
    from torch.cuda.amp import autocast, GradScaler

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    frames_3sec = 18

    # ---------------------------
    # 0) Load data
    # ---------------------------
    with open(args["all_data_pkl"], 'rb') as f:
        all_data = pickle.load(f)
    all_data = check_data_sanity(all_data)
    train_data, val_data, test_data = split_all_data(all_data, 0.7, 0.2, 0.1)

    base_tokenizer = AutoTokenizer.from_pretrained(args["base_model_name"])
    if base_tokenizer.pad_token is None:
        base_tokenizer.pad_token = base_tokenizer.eos_token

    # Build train
    train_in, train_out = build_dataset_from_tracks_sliding(
        train_data,
        seq_len=frames_3sec,
        out_len=frames_3sec,
        stride=args["stride"],
        max_step=args["max_step"],
        max_speed_diff=args["max_speed_diff"],
        downsample=args["downsample"],
        tokenizer=base_tokenizer,
        max_length=512
    )
    train_ds = MultiModalTrajectoryDataset(train_in, train_out)
    train_dl = DataLoader(train_ds, batch_size=args["batch_size"],
                          shuffle=True, collate_fn=custom_collate_fn)

    # Build val
    val_in, val_out = build_dataset_from_tracks_sliding(
        val_data,
        seq_len=frames_3sec,
        out_len=frames_3sec,
        stride=args["stride"],
        max_step=args["max_step"],
        max_speed_diff=args["max_speed_diff"],
        downsample=args["downsample"],
        tokenizer=base_tokenizer,
        max_length=512
    )
    val_ds = MultiModalTrajectoryDataset(val_in, val_out)
    val_dl = DataLoader(val_ds, batch_size=args["batch_size"],
                        shuffle=False, collate_fn=custom_collate_fn)

    diffusion_model = ConditionalUNet(in_channels=2, cond_dim=128,
                                      base_channels=64, n_layers=3).to(device)
    optimizer = torch.optim.Adam(diffusion_model.parameters(),
                                 lr=args["diffusion_lr"],
                                 weight_decay=1e-4)
    ema = EMA(diffusion_model, decay=0.9999)

    # Setup DDPM schedule
    n_timestep = 1000
    betas = make_beta_schedule("cosine", n_timestep, start=1e-4, end=0.02)
    alphas = 1.0 - betas
    alphas_cum = np.cumprod(alphas)
    sqrt_alphas_cum = np.sqrt(alphas_cum)
    sqrt_one_minus_alphas_cum = np.sqrt(1 - alphas_cum)

    mse = nn.MSELoss()

    # q_sample: forward diffusion
    def q_sample(x0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x0)
        sqrt_alpha = extract(sqrt_alphas_cum, t, x0.shape)
        sqrt_1m_alpha = extract(sqrt_one_minus_alphas_cum, t, x0.shape)
        return sqrt_alpha * x0 + sqrt_1m_alpha * noise

    # Sample a random embedding for condition
    def get_cond_embedding(batch_data):
        B = batch_data["traj_emb"].size(0)
        cond = torch.zeros(B, 128, device=device)  # placeholder
        return cond

    scaler = GradScaler(enabled=True)
    epochs = args["diffusion_epochs"]

    # ---------------------------
    # 1) Train loop
    # ---------------------------
    for epoch in range(epochs):
        diffusion_model.train()
        total_loss = 0.0
        for step, batch_data in enumerate(train_dl):
            x0 = batch_data["target_traj"].to(device)  # [B,2,T_out]
            B_ = x0.size(0)

            cond_emb = get_cond_embedding(batch_data)
            uncond_emb = torch.randn_like(cond_emb) * 0.05  # "unconditional"
            full_emb = torch.cat([uncond_emb, cond_emb], dim=0)
            x0_full = torch.cat([x0, x0], dim=0)

            t_ = torch.randint(0, n_timestep, (2 * B_,), device=device).long()
            noise = torch.randn_like(x0_full)
            x_t = q_sample(x0_full, t_, noise)

            with autocast(enabled=True, dtype=torch.float16):
                noise_pred = diffusion_model(x_t, full_emb, cfg_scale=args.get("cfg_scale", 3.0))
                loss = mse(noise_pred, noise)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            total_loss += loss.item()
            ema.update(diffusion_model)

        avg_loss = total_loss / len(train_dl) if len(train_dl) > 0 else 0.0

        # Validate
        diffusion_model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for step, batch_data in enumerate(val_dl):
                x0 = batch_data["target_traj"].to(device)
                B_ = x0.size(0)
                cond_emb = get_cond_embedding(batch_data)
                uncond_emb = torch.randn_like(cond_emb) * 0.05
                full_emb = torch.cat([uncond_emb, cond_emb], dim=0)
                x0_full = torch.cat([x0, x0], dim=0)
                t_ = torch.randint(0, n_timestep, (2 * B_,), device=device).long()
                noise = torch.randn_like(x0_full)
                x_t = q_sample(x0_full, t_, noise)
                with autocast(enabled=True, dtype=torch.float16):
                    noise_pred = diffusion_model(x_t, full_emb, cfg_scale=args.get("cfg_scale", 3.0))
                    lv = mse(noise_pred, noise)
                val_loss += lv.item()

        avg_val = val_loss / len(val_dl) if len(val_dl) > 0 else 0.0
        print(f"[DDPM][Epoch {epoch+1}/{epochs}] Train={avg_loss:.4f}, Val={avg_val:.4f}")

    # apply EMA at the end
    ema.apply_shadow(diffusion_model)

    # ---------------------------
    # 2) Sampling on test
    # ---------------------------
    test_in, test_out = build_dataset_from_tracks_sliding(
        test_data,
        seq_len=frames_3sec,
        out_len=frames_3sec,
        stride=args["stride"],
        max_step=args["max_step"],
        max_speed_diff=args["max_speed_diff"],
        downsample=args["downsample"],
        tokenizer=base_tokenizer,
        max_length=512
    )
    if not test_in:
        print("[DDPM] No test data found.")
        return

    # pick 1 random sample, do 5 generation attempts
    sample_idx = random.randint(0, len(test_in) - 1)
    sample_in = test_in[sample_idx]
    sample_out = test_out[sample_idx]
    min_x, max_x, min_y, max_y = sample_in["norm_stat"]
    rx = max_x - min_x
    ry = max_y - min_y

    # "p_mean_variance": compute posterior distribution from x_t
    def p_mean_variance(model, x_t, t, cond_emb):
        """
        Standard DDPM formula to get predicted mean & variance
        x_t: shape [B, C, T]
        """
        # predict noise
        noise_pred = model(x_t, cond_emb, cfg_scale=args.get("cfg_scale", 3.0))
        # expansions
        sqrt_oma_t = extract(sqrt_one_minus_alphas_cum, t, x_t.shape)
        sqrt_alpha_cum_t = extract(sqrt_alphas_cum, t, x_t.shape)

        # eq. x0_pred
        x0_pred = (x_t - noise_pred * sqrt_oma_t) / (sqrt_alpha_cum_t + 1e-7)
        x0_pred = torch.clamp(x0_pred, -1.0, 1.0)  # keep safe bounds

        # posterior variance
        alphas_cum_ = alphas_cum
        alphas_cum_prev_ = np.append(1.0, alphas_cum_[:-1])
        posterior_var_ = betas * (1 - alphas_cum_prev_) / (1 - alphas_cum_)

        var_t = extract(posterior_var_, t, x_t.shape)
        # eq. posterior mean: see eq. (12) in DDPM paper
        alpha_cum_prev_t = extract(alphas_cum_prev_, t, x_t.shape)
        mean = ( torch.sqrt(alpha_cum_prev_t) * x0_pred +
                 torch.sqrt(1 - alpha_cum_prev_t) * noise_pred )
        return mean, var_t

    # standard DDPM sampling
    def p_sample(model, x_t, t, cond_emb):
        mean_t, var_t = p_mean_variance(model, x_t, t, cond_emb)
        if (t == 0).all():
            return mean_t
        else:
            z = torch.randn_like(x_t)
            return mean_t + torch.sqrt(var_t) * z

    def p_sample_loop(model, shape, cond_emb):
        x = torch.randn(shape, device=device)
        B_ = x.size(0)
        for i in reversed(range(n_timestep)):
            t_ = torch.full((B_,), i, dtype=torch.long, device=device)
            x = p_sample(model, x, t_, cond_emb)
        return x

    # Prepare uncond/cond embedding
    c_cond = torch.zeros(1, 128, device=device)
    c_uncond = torch.randn_like(c_cond) * 0.05
    full_c = torch.cat([c_uncond, c_cond], dim=0)

    # 5 generated samples
    diffusion_model.eval()
    gen_list = []
    with torch.no_grad():
        shape_ = (2, 2, frames_3sec)  # B=2, channels=2, T=18
        for i in range(5):
            gen_ = p_sample_loop(diffusion_model, shape_, full_c)
            gen_cond = gen_[1].cpu().numpy()  # shape [2, T]
            gen_list.append(gen_cond)

    # Past + GT
    in_np = sample_in["trajectory_embeddings"].numpy()  # [T_in, 2]
    gt_np = sample_out.numpy()                           # [T_out, 2]

    in_den = in_np.copy()
    in_den[:, 0] = in_den[:, 0] * rx + min_x
    in_den[:, 1] = in_den[:, 1] * ry + min_y

    gt_den = gt_np.copy()
    gt_den[:, 0] = gt_den[:, 0] * rx + min_x
    gt_den[:, 1] = gt_den[:, 1] * ry + min_y

    gen_den_list = []
    for arr in gen_list:
        arr_ = arr.T  # => shape [T_out, 2]
        arr_[:, 0] = arr_[:, 0] * rx + min_x
        arr_[:, 1] = arr_[:, 1] * ry + min_y
        gen_den_list.append(arr_)

    T_in = in_den.shape[0]
    T_out = gt_den.shape[0]
    t_past = np.arange(T_in)
    t_future = np.arange(T_in, T_in + T_out)

    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

    # X
    ax1 = axes[0]
    ax1.plot(t_past, in_den[:, 0], 'b-o', label='Past(3s)')
    ax1.plot(t_future, gt_den[:, 0], 'g-o', label='GT-Future(3s)')
    colors = ['r', 'm', 'c', 'orange', 'purple']
    for i, gg in enumerate(gen_den_list):
        ax1.plot(t_future, gg[:, 0], color=colors[i], marker='o', label=f'Gen#{i+1}')
    ax1.grid(True)
    ax1.set_ylabel("X coord")
    ax1.legend()

    # Y
    ax2 = axes[1]
    ax2.plot(t_past, in_den[:, 1], 'b-o', label='Past(3s)')
    ax2.plot(t_future, gt_den[:, 1], 'g-o', label='GT-Future(3s)')
    for i, gg in enumerate(gen_den_list):
        ax2.plot(t_future, gg[:, 1], color=colors[i], marker='o', label=f'Gen#{i+1}')
    ax2.grid(True)
    ax2.set_xlabel("Frame index (time)")
    ax2.set_ylabel("Y coord")
    ax2.legend()

    fig.suptitle("DDPM 3s -> Next 3s (5 samples, time-series)")
    os.makedirs("diffusion_generation", exist_ok=True)
    fn = os.path.join("diffusion_generation", "sample_generation_timeseries.png")
    plt.savefig(fn)
    plt.close()
    print(f"[DDPM] saved sample generation => {fn}")


#################################################
# 2. (C) 기존 궤적 예측용 DDP 학습
#################################################
def train_ddp(args):
    import torch.nn.utils as nn_utils

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    if world_size > 1:
        dist.init_process_group("nccl", rank=local_rank, world_size=world_size)
        torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    with open(args["all_data_pkl"], 'rb') as f:
        all_data = pickle.load(f)
    all_data = check_data_sanity(all_data)
    train_data, val_data, test_data = split_all_data(all_data, 0.7, 0.2, 0.1)

    base_tokenizer = AutoTokenizer.from_pretrained(args["base_model_name"])
    if base_tokenizer.pad_token is None:
        base_tokenizer.pad_token = base_tokenizer.eos_token

    train_in, train_out = build_dataset_from_tracks_sliding(
        train_data,
        seq_len=args["seq_len"],
        out_len=args["out_len"],
        stride=args["stride"],
        max_step=args["max_step"],
        max_speed_diff=args["max_speed_diff"],
        downsample=args["downsample"],
        tokenizer=base_tokenizer,
        max_length=512
    )
    train_ds = MultiModalTrajectoryDataset(train_in, train_out)

    val_in, val_out = build_dataset_from_tracks_sliding(
        val_data,
        seq_len=args["seq_len"],
        out_len=args["out_len"],
        stride=args["stride"],
        max_step=args["max_step"],
        max_speed_diff=args["max_speed_diff"],
        downsample=args["downsample"],
        tokenizer=base_tokenizer,
        max_length=512
    )
    val_ds = MultiModalTrajectoryDataset(val_in, val_out)

    if world_size > 1:
        train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=local_rank, shuffle=True)
        val_sampler = DistributedSampler(val_ds, num_replicas=world_size, rank=local_rank, shuffle=False)
    else:
        train_sampler = None
        val_sampler = None

    train_dl = DataLoader(train_ds, batch_size=args["batch_size"],
                          sampler=train_sampler, shuffle=(train_sampler is None),
                          collate_fn=custom_collate_fn)
    val_dl = DataLoader(val_ds, batch_size=args["batch_size"],
                        sampler=val_sampler, shuffle=False,
                        collate_fn=custom_collate_fn)

    model = MultiModalTrajectoryModel(
        seq_len=args["seq_len"],
        out_len=args["out_len"],
        individual=True,
        feature_size=2,
        d_model=args["d_model"],
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
        ltsf_dropout=args["ltsf_dropout"]
    ).to(device)

    if world_size > 1:
        ddp_model = pnl.DistributedDataParallel(model, device_ids=[local_rank],
                                                output_device=local_rank,
                                                find_unused_parameters=False)
    else:
        ddp_model = model

    trainable = [p for p in ddp_model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args["lr"], weight_decay=1e-4)
    epochs = args["epochs"]
    best_val = float('inf')
    best_ckpt = args.get("ckpt_path", "3_3_best_val_checkpoint.pt")

    for epoch in range(epochs):
        ddp_model.train()
        if world_size > 1:
            train_sampler.set_epoch(epoch)
        total_loss = 0.0

        for step, batch_data in enumerate(train_dl):
            x = batch_data["traj_emb"].to(device)
            y = batch_data["target_traj"].to(device)
            v = batch_data["vision_emb"].to(device)
            c = batch_data["context_str"]
            ns = batch_data["norm_stat"]
            p = batch_data["lane_polygon"].to(device)
            pl = batch_data["lane_polygon_len"]
            i_ids = batch_data["input_ids"].to(device)
            a_m = batch_data["attention_mask"].to(device)
            labs = batch_data["labels"].to(device)

            optimizer.zero_grad()
            ret = ddp_model(
                x, v, c, p, pl,
                y=y, norm_stat=ns,
                input_ids=i_ids,
                attention_mask=a_m,
                labels=labs
            )
            if isinstance(ret, tuple):
                loss = ret[0]
            else:
                loss = torch.tensor(0.0, device=device)

            loss.backward()
            if torch.isfinite(loss):
                nn_utils.clip_grad_norm_(ddp_model.parameters(), 1.0)
                optimizer.step()
            else:
                print(f"[WARNING][Rank={local_rank}] NaN @ step {step}")
                optimizer.zero_grad()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_dl)

        ddp_model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for step, batch_data in enumerate(val_dl):
                x_v = batch_data["traj_emb"].to(device)
                y_v = batch_data["target_traj"].to(device)
                vv = batch_data["vision_emb"].to(device)
                cc = batch_data["context_str"]
                ns_ = batch_data["norm_stat"]
                pp = batch_data["lane_polygon"].to(device)
                pll = batch_data["lane_polygon_len"]
                i_ids_v = batch_data["input_ids"].to(device)
                am_v = batch_data["attention_mask"].to(device)
                labs_v = batch_data["labels"].to(device)

                out = ddp_model(x_v, vv, cc, pp, pll,
                                y=y_v, norm_stat=ns_,
                                input_ids=i_ids_v,
                                attention_mask=am_v,
                                labels=labs_v)
                if isinstance(out, tuple):
                    lv = out[0]
                else:
                    lv = torch.tensor(0.0, device=device)
                val_loss += lv.item()

        avg_val = val_loss / len(val_dl) if len(val_dl) > 0 else 0.0

        if local_rank == 0:
            print(f"[Epoch {epoch+1}/{epochs}] train={avg_loss:.4f}, val={avg_val:.4f}")
            if avg_val < best_val:
                best_val = avg_val
                torch.save(ddp_model.module.state_dict() if world_size > 1 else ddp_model.state_dict(),
                           best_ckpt)
                print(f"  best updated => {best_val:.4f}, ckpt={best_ckpt}")


#################################################
# 3. 메인
#################################################
def main():
    args = {
        "mode": "generate",  # "predict" / "generate" / "train_ddp"
        "all_data_pkl": "/home/user/MLLM/data/all_data.pkl",
        "seq_len": 18,
        "out_len": 18,
        "batch_size": 8,
        "epochs": 15,
        "lr": 1e-5,
        "stride": 6,
        "downsample": 5,
        "max_step": 50.0,
        "max_speed_diff": 30.0,
        "base_model_name": "meta-llama/Llama-3.2-1B",
        "use_lora": True,
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
        "d_model": 64,
        "inference_ckpt": "./3_3_best_val_checkpoint.pt",
        "max_polygon_points": 64,
        # Diffusion
        "diffusion_lr": 1e-4,
        "diffusion_epochs": 20,
        "cfg_scale": 3.0,
        # ddp
        "ckpt_path": "3_3_best_val_checkpoint.pt"
    }

    mode = args["mode"]
    if mode == "predict":
        predict_trajectory(args)
    elif mode == "generate":
        generate_trajectory(args)
    elif mode == "train_ddp":
        train_ddp(args)
    else:
        print(f"Unknown mode={mode}")


if __name__ == "__main__":
    main()
