### ablation_study_without_lora.py ###

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

from transformers import AutoTokenizer, AutoModelForCausalLM

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

########################################
# build_dataset_from_tracks_sliding
# Prompt+Answer -> input_ids/labels
########################################
def build_dataset_from_tracks_sliding(track_list,
                                      seq_len=30,
                                      out_len=60,
                                      stride=1,
                                      max_step=50.0,
                                      max_speed_diff=30.0,
                                      image_width=3840,
                                      image_height=1280,
                                      downsample=5,
                                      tokenizer=None,
                                      max_length=512):
    inputs_list = []
    outputs_list = []

    for track_idx, item in enumerate(track_list):
        raw_traj = item["raw_trajectory"]
        raw_traj = raw_traj[::downsample]

        vision_emb = item.get("vision_embeddings", None)
        if vision_emb is not None:
            vision_emb = vision_emb[::downsample]

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

            if vision_emb is not None:
                in_vision = vision_emb[start:start+seq_len]
                if in_vision.shape[0] < seq_len:
                    pad_sz = seq_len - in_vision.shape[0]
                    pad_emb = torch.zeros(pad_sz, in_vision.shape[1], dtype=in_vision.dtype)
                    in_vision = torch.cat([in_vision, pad_emb], dim=0)
                in_vision = in_vision.float()
            else:
                in_vision = torch.zeros(seq_len, 1, dtype=torch.float32)

            prompt_text = (
                f"You are analyzing the ego vehicle with track_id={track_id}.\n"
                "Below is partial information about this ego vehicle and its surroundings.\n"
                "Use the provided data (<vision>) to create a comprehensive text describing:\n"
                "1) the ego vehicle's lane, site, and bounding box dimensions,\n"
                "2) velocity, acceleration, and heading info,\n"
                "3) neighbor vehicles,\n"
                "4) average speed in the area.\n\n"
                "Please provide your answer as a natural language paragraph.\n\n"
                "Answer:\n"
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
# Dataset
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
            poly_len = n_p if n_p>0 else 0
        padded = np.zeros((self.max_polygon_points, 2), dtype=np.float32)
        if n_p>0:
            padded[:poly_len, :] = polygon
        sample["lane_polygon"] = torch.tensor(padded, dtype=torch.float32)
        sample["lane_polygon_len"] = poly_len
        return sample

def custom_collate_fn(batch):
    from torch.nn.utils.rnn import pad_sequence
    traj_list = [b["traj_emb"] for b in batch]
    targ_list = [b["target_traj"] for b in batch]
    vision_list = [b["vision_emb"] for b in batch]

    x_3d, y_3d = [], []
    for t_in, t_out in zip(traj_list, targ_list):
        x_3d.append(t_in.transpose(0, 1))
        y_3d.append(t_out.transpose(0, 1))

    x_3d = torch.stack(x_3d, dim=0)
    y_3d = torch.stack(y_3d, dim=0)
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
            if valid_len>0:
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
# LlamaWithCrossAttnPEFT (LoRA 제거)
########################################
class LlamaWithCrossAttnPEFT(nn.Module):
    def __init__(self, base_model_name):
        super().__init__()
        self.llama_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            load_in_8bit=False,
            device_map=None
        )
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
# LlamaMultiModal (LoRA 제거 반영)
########################################
class LlamaMultiModal(nn.Module):
    def __init__(self,
                 base_model_name="meta-llama/Llama-2-7b-hf",
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
        self.q_hidden_size = q_hidden_size

        self.llama_wrapper = LlamaWithCrossAttnPEFT(base_model_name=base_model_name)
        self.llama_hidden_size = self.llama_wrapper.hidden_size

        if self.llama_hidden_size != self.q_hidden_size:
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

        image_tokens = self.qformer(vision_embs)
        image_tokens = self.q_proj(image_tokens)
        image_tokens = image_tokens + self.vision_modality_embedding

        if (input_ids is not None) and (attention_mask is not None):
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

        else:
            text_inputs = self.tokenizer(context_str, return_tensors="pt", padding=True, truncation=True)
            text_inputs = {k: v.to(device) for k,v in text_inputs.items()}
            input_ids_ = text_inputs["input_ids"]
            mask_ = text_inputs["attention_mask"]
            text_embeds = self.llama_wrapper.llama_model.get_input_embeddings()(input_ids_)
            text_embeds = text_embeds + self.text_modality_embedding
            fused_embeds = torch.cat([image_tokens, text_embeds], dim=1)
            img_mask = torch.ones((B, image_tokens.size(1)), dtype=mask_.dtype, device=device)
            fused_mask = torch.cat([img_mask, mask_], dim=1)

            outputs = self.llama_wrapper(
                inputs_embeds=fused_embeds,
                attention_mask=fused_mask,
                labels=None,
                output_hidden_states=True
            )
            final_hidden = outputs.hidden_states[-1]
            return final_hidden, image_tokens.size(1)

    def generate_batch(self,
                       vision_embs,
                       prompt_ids,
                       tokenizer,
                       max_new_tokens=128,
                       temperature=0.9,
                       top_k=40,
                       top_p=0.9,
                       device="cuda"):
        self.eval()
        B = vision_embs.size(0)
        with torch.no_grad():
            image_tokens = self.qformer(vision_embs)
            image_tokens = self.q_proj(image_tokens)
            image_tokens = image_tokens + self.vision_modality_embedding

            text_embeds = self.llama_wrapper.llama_model.get_input_embeddings()(prompt_ids)
            text_embeds = text_embeds + self.text_modality_embedding

            fused_embeds = torch.cat([image_tokens, text_embeds], dim=1)
            img_len = image_tokens.size(1)
            fused_mask = torch.cat([
                torch.ones(B, img_len, dtype=prompt_ids.dtype, device=device),
                torch.ones_like(prompt_ids)
            ], dim=1)

        orig_embedding = self.llama_wrapper.llama_model.get_input_embeddings()
        prefix_len = fused_embeds.size(1)

        def patched_embedding(ids):
            seq_len = ids.shape[1]
            if seq_len <= prefix_len:
                return fused_embeds[:, :seq_len, :]
            else:
                prefix_part = fused_embeds[:, :prefix_len, :]
                new_token_ids = ids[:, prefix_len:]
                new_embeds = orig_embedding(new_token_ids)
                return torch.cat([prefix_part, new_embeds], dim=1)

        self.llama_wrapper.llama_model.set_input_embeddings(
            nn.Embedding(orig_embedding.num_embeddings,
                         orig_embedding.embedding_dim).to(device)
        )
        self.llama_wrapper.llama_model.get_input_embeddings().forward = patched_embedding

        fake_attention_mask = fused_mask
        fake_input_ids = prompt_ids.clone()

        gen_params = dict(
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            do_sample=True,
            no_repeat_ngram_size=3,
            repetition_penalty=1.2
        )
        with torch.no_grad():
            outputs = self.llama_wrapper.llama_model.generate(
                input_ids=fake_input_ids,
                attention_mask=fake_attention_mask,
                **gen_params
            )

        self.llama_wrapper.llama_model.set_input_embeddings(orig_embedding)
        generated_texts = []
        for out_ids in outputs:
            text = tokenizer.decode(out_ids, skip_special_tokens=True)
            cutoff_marker = "No right-following vehicle."
            if cutoff_marker in text:
                cutoff_idx = text.index(cutoff_marker) + len(cutoff_marker)
                text = text[:cutoff_idx]
            generated_texts.append(text)
        return generated_texts

########################################
# SelfAttentionBlock, LTSF
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
            BC = B*C
            out_ = self.encoder_linear(x_sub.view(BC, T)).view(B, C, T)
            encoded = out_
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

        self.cross_dim = cross_dim
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
            BC = B*C
            dec_out = self.decoder_linear(x_sub.view(BC, T)).view(B, C, self.forecast_size)
            decoded = dec_out
        decoded = decoded + seq_last.repeat(1, 1, self.forecast_size)

        lane_adj = self.lane_fc(lane_polygon_emb).view(B, C, self.forecast_size)
        decoded = decoded + lane_adj

        if self.use_post_mlp:
            flat_dec = decoded.reshape(B, -1)
            post_out = self.post_mlp(flat_dec)
            post_out = post_out.view(B, C, self.forecast_size)
            decoded = post_out

        dec_t = decoded.permute(0, 2, 1)
        proj_dec = self.dec_proj(dec_t)
        query = proj_dec.transpose(0,1)
        key   = final_hidden.transpose(0,1)
        val   = final_hidden.transpose(0,1)
        cross_out = self.cross_attn(query, key, val)[0].transpose(0,1)
        cross_to_d = self.dec_unproj(cross_out)
        fused = dec_t + cross_to_d
        fused = self.fusion_layer(fused)
        out = fused.permute(0,2,1)
        out = out.permute(0,2,1)
        out = self.out_proj(out)
        out = out.permute(0,2,1)
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
                                           post_mlp_output_dim=d_model*out_len,
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
# 최종 모델 (LoRA 제거 반영)
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
            post_mlp_output_dim=d_model*out_len,
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
        last_exp = last_in.repeat(1,1,self.out_len)
        decoded = decoded + last_exp

        if (y is not None) and (norm_stat is not None):
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
            loss = loss_x + loss_y
            return loss, decoded
        else:
            return decoded

########################################
# 시각화
########################################
def visualize_one_sample(model, sample, device, idx=0,
                         prefix="test_sample", save_dir="visualization_test_ddp"):
    os.makedirs(save_dir, exist_ok=True)
    model.eval()

    in_traj = sample["traj_emb"]
    out_traj= sample["target_traj"]
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
    if isinstance(out, tuple):
        pred = out[1]
    else:
        pred = out

    pred_np = pred.squeeze(0).cpu().numpy().transpose(1,0)
    in_np = in_traj.cpu().numpy()
    gt_np = out_traj.cpu().numpy()

    range_x = max_x - min_x
    range_y = max_y - min_y
    in_den = in_np.copy()
    gt_den = gt_np.copy()
    pd_den = pred_np.copy()
    in_den[:,0] = in_den[:,0]*range_x + min_x
    in_den[:,1] = in_den[:,1]*range_y + min_y
    gt_den[:,0] = gt_den[:,0]*range_x + min_x
    gt_den[:,1] = gt_den[:,1]*range_y + min_y
    pd_den[:,0] = pd_den[:,0]*range_x + min_x
    pd_den[:,1] = pd_den[:,1]*range_y + min_y

    plt.figure(figsize=(8,6))
    plt.plot(in_den[:,0], in_den[:,1],'bo-',label='Past')
    plt.plot(gt_den[:,0], gt_den[:,1],'go-',label='GT')
    plt.plot(pd_den[:,0], pd_den[:,1],'ro-',label='Pred')
    plt.title(f"{prefix} idx={idx}")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid(True)
    plt.legend()
    filename= f"{prefix}_idx_{idx}.png"
    save_path= os.path.join(save_dir, filename)
    plt.savefig(save_path)
    plt.close()
    print(f"[{prefix}] Sample {idx} => Saved to {save_path}")

def visualize_test_samples(model, dataset, device, num_samples=5,
                           prefix="test_sample", save_dir="visualization_test_ddp"):
    os.makedirs(save_dir, exist_ok=True)
    indices= random.sample(range(len(dataset)), min(num_samples, len(dataset)))
    for i, idx in enumerate(indices):
        sample= dataset[idx]
        visualize_one_sample(model, sample, device, idx=idx,
                             prefix=prefix, save_dir=save_dir)

########################################
# Train Loop (체크포인트 조정 추가)
########################################
def train_ddp(args):
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    dist.init_process_group("nccl", rank=local_rank, world_size=world_size)
    torch.cuda.set_device(local_rank)
    device= torch.device(f"cuda:{local_rank}")

    with open(args["all_data_pkl"], 'rb') as f:
        all_data= pickle.load(f)
    train_data, val_data, test_data= split_all_data(all_data, 0.7,0.2,0.1)

    base_tokenizer = AutoTokenizer.from_pretrained(args["base_model_name"])
    if base_tokenizer.pad_token is None:
        base_tokenizer.pad_token= base_tokenizer.eos_token

    train_inputs, train_outputs = build_dataset_from_tracks_sliding(
        track_list=train_data,
        seq_len=args["seq_len"],
        out_len=args["out_len"],
        stride=args["stride"],
        max_step=args["max_step"],
        max_speed_diff=args["max_speed_diff"],
        image_width=args["image_width"],
        image_height=args["image_height"],
        downsample=args["downsample"],
        tokenizer=base_tokenizer,
        max_length=512
    )
    train_dataset= MultiModalTrajectoryDataset(train_inputs, train_outputs,
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
        downsample=args["downsample"],
        tokenizer=base_tokenizer,
        max_length=512
    )
    val_dataset= MultiModalTrajectoryDataset(val_inputs, val_outputs,
                                             max_polygon_points=args["max_polygon_points"])

    train_sampler= DistributedSampler(train_dataset, num_replicas=world_size, rank=local_rank, shuffle=True)
    val_sampler= DistributedSampler(val_dataset, num_replicas=world_size, rank=local_rank, shuffle=False)

    train_loader= DataLoader(train_dataset, batch_size=args["batch_size"],
                             sampler=train_sampler, collate_fn=custom_collate_fn)
    val_loader= DataLoader(val_dataset, batch_size=args["batch_size"],
                           sampler=val_sampler, collate_fn=custom_collate_fn)

    model= MultiModalTrajectoryModel(
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
        base_model_name=args["base_model_name"],
        vision_dim=args["vision_dim"],
        q_hidden_size=args["qformer_hidden_size"],
        q_nhead=args["qformer_nhead"],
        q_enc_layers=args["qformer_enc_layers"],
        q_dec_layers=args["qformer_dec_layers"],
        q_num_query_tokens=args["qformer_num_query_tokens"],
        ltsf_nhead=args["ltsf_nhead"],
        ltsf_dropout=args["ltsf_dropout"],
    ).to(device)

    ddp_model= pnl.DistributedDataParallel(
        model,
        device_ids=[local_rank],
        output_device=local_rank,
        find_unused_parameters=True
    )

    # 체크포인트 키 조정 함수
    def adjust_state_dict(state_dict, prefix_to_remove="base_model.model."):
        new_state_dict = {}
        for key, value in state_dict.items():
            if "lora_A" in key or "lora_B" in key:
                continue
            new_key = key.replace(prefix_to_remove, "")
            new_state_dict[new_key] = value
        return new_state_dict

    if local_rank==0:
        print("[Info] Loading MLLM ckpt:", args["mllm_ckpt"])
    state_dict= torch.load(args["mllm_ckpt"], map_location=device)
    adjusted_state_dict = adjust_state_dict(state_dict)
    ddp_model.module.mllm.load_state_dict(adjusted_state_dict, strict=True)

    for param in ddp_model.module.mllm.parameters():
        param.requires_grad= False

    trainable_params= [p for p in ddp_model.parameters() if p.requires_grad]
    optimizer= torch.optim.AdamW(trainable_params, lr=args["lr"], weight_decay=1e-4)

    epochs= args["epochs"]
    best_val_loss= float('inf')
    best_ckpt_path= "/home/user/MLLM/scripts/6_30_best_val_checkpoint.pt"

    # 주석 처리된 학습 루프 (필요 시 활성화)
    # for epoch in range(epochs):
    #     ddp_model.train()
    #     train_sampler.set_epoch(epoch)
    #     total_loss= 0.0
    #     for step,batch_data in enumerate(train_loader):
    #         x= batch_data["traj_emb"].to(device)
    #         y= batch_data["target_traj"].to(device)
    #         v= batch_data["vision_emb"].to(device)
    #         c= batch_data["context_str"]
    #         ns= batch_data["norm_stat"]
    #         p= batch_data["lane_polygon"].to(device)
    #         pl= batch_data["lane_polygon_len"]
    #         in_ids= batch_data["input_ids"].to(device)
    #         attn_m= batch_data["attention_mask"].to(device)
    #         labs  = batch_data["labels"].to(device)
    #         optimizer.zero_grad()
    #         ret= ddp_model(x, v, c, p, pl, y=y, norm_stat=ns,
    #                        input_ids=in_ids, attention_mask=attn_m, labels=labs)
    #         if isinstance(ret,tuple):
    #             loss,_= ret
    #         else:
    #             loss= torch.tensor(0.0, device=device)
    #         loss.backward()
    #         optimizer.step()
    #         total_loss+= loss.item()
    #     avg_loss= total_loss/len(train_loader)
    #     ddp_model.eval()
    #     val_loss=0.0
    #     with torch.no_grad():
    #         for batch_data in val_loader:
    #             x_v= batch_data["traj_emb"].to(device)
    #             y_v= batch_data["target_traj"].to(device)
    #             vv= batch_data["vision_emb"].to(device)
    #             cc= batch_data["context_str"]
    #             ns_= batch_data["norm_stat"]
    #             pp= batch_data["lane_polygon"].to(device)
    #             pll= batch_data["lane_polygon_len"]
    #             in_ids_v= batch_data["input_ids"].to(device)
    #             attn_v= batch_data["attention_mask"].to(device)
    #             labs_v= batch_data["labels"].to(device)
    #             rr= ddp_model(x_v, vv, cc, pp, pll, y=y_v, norm_stat=ns_,
    #                           input_ids=in_ids_v, attention_mask=attn_v, labels=labs_v)
    #             if isinstance(rr, tuple):
    #                 lv,_= rr
    #             else:
    #                 lv= torch.tensor(0.0, device=device)
    #             val_loss+= lv.item()
    #     avg_val_loss= val_loss/len(val_loader)
    #     if local_rank==0:
    #         print(f"[Epoch {epoch+1}/{epochs}] Train Loss={avg_loss:.4f}, Val Loss={avg_val_loss:.4f}")
    #         if avg_val_loss< best_val_loss:
    #             best_val_loss= avg_val_loss
    #             torch.save(ddp_model.module.state_dict(), best_ckpt_path)
    #             print(f"  Best Val Loss updated => {best_val_loss:.4f}, ckpt={best_ckpt_path}")
    #         sample_data= next(iter(val_loader))
    #         sample_vision= sample_data["vision_emb"][0:1].to(device)
    #         prompt_ids= sample_data["input_ids"][0:1].to(device)
    #         gen_texts= ddp_model.module.mllm.generate_batch(
    #             vision_embs=sample_vision,
    #             prompt_ids=prompt_ids,
    #             tokenizer=ddp_model.module.mllm.tokenizer,
    #             max_new_tokens=128, 
    #             temperature=0.9,
    #             top_k=40, 
    #             top_p=0.9,
    #             device=device
    #         )
    #         print(f"[Epoch {epoch+1}] Sample Generation:\n{gen_texts[0]}\n")
    #         rand_idx= random.randint(0, len(val_dataset)-1)
    #         sample_val= val_dataset[rand_idx]
    #         visualize_one_sample(ddp_model.module, sample_val, device, idx=epoch,
    #                              prefix="val_sample", save_dir="30_18_train_visualization_val")

    if local_rank==0 and os.path.exists(best_ckpt_path):
        best_state= torch.load(best_ckpt_path, map_location=device)
        ddp_model.module.load_state_dict(best_state)
        print(f"Loaded best model from {best_ckpt_path}")

    if local_rank==0:
        test_inputs, test_outputs= build_dataset_from_tracks_sliding(
            track_list=test_data,
            seq_len=args["seq_len"],
            out_len=args["out_len"],
            stride=args["stride"],
            max_step=args["max_step"],
            max_speed_diff=args["max_speed_diff"],
            image_width=args["image_width"],
            image_height=args["image_height"],
            downsample=args["downsample"],
            tokenizer=base_tokenizer,
            max_length=512
        )
        test_dataset= MultiModalTrajectoryDataset(test_inputs, test_outputs,
                                                  max_polygon_points=args["max_polygon_points"])
        test_sampler= DistributedSampler(test_dataset, num_replicas=world_size, rank=local_rank, shuffle=False)
        test_loader= DataLoader(test_dataset, batch_size=args["batch_size"],
                                sampler=test_sampler, collate_fn=custom_collate_fn)
        ddp_model.eval()
        total_ade=0.0
        total_fde=0.0
        total_rmse=0.0
        total_samp=0
        with torch.no_grad():
            for batch_data in test_loader:
                x_t= batch_data["traj_emb"].to(device)
                y_t= batch_data["target_traj"].to(device)
                v_t= batch_data["vision_emb"].to(device)
                c_t= batch_data["context_str"]
                n_t= batch_data["norm_stat"]
                p_t= batch_data["lane_polygon"].to(device)
                pl_t= batch_data["lane_polygon_len"]
                in_ids_t= batch_data["input_ids"].to(device)
                attn_t= batch_data["attention_mask"].to(device)
                labs_t= batch_data["labels"].to(device)
                out_t= ddp_model(
                    x_t, v_t, c_t, p_t, pl_t,
                    y=None, norm_stat=None,
                    input_ids=in_ids_t,
                    attention_mask=attn_t,
                    labels=None
                )
                if isinstance(out_t, tuple):
                    out_t= out_t[1]
                B_= x_t.size(0)
                pred_den= out_t.clone()
                y_den= y_t.clone()
                min_x= torch.tensor([ns[0] for ns in n_t], device=device).view(B_,1,1)
                max_x= torch.tensor([ns[1] for ns in n_t], device=device).view(B_,1,1)
                min_y= torch.tensor([ns[2] for ns in n_t], device=device).view(B_,1,1)
                max_y= torch.tensor([ns[3] for ns in n_t], device=device).view(B_,1,1)
                rx= max_x- min_x
                ry= max_y- min_y
                pred_den[:,0,:]= pred_den[:,0,:]*rx.squeeze(2)+ min_x.squeeze(2)
                pred_den[:,1,:]= pred_den[:,1,:]*ry.squeeze(2)+ min_y.squeeze(2)
                y_den[:,0,:]= y_den[:,0,:]*rx.squeeze(2)+ min_x.squeeze(2)
                y_den[:,1,:]= y_den[:,1,:]*ry.squeeze(2)+ min_y.squeeze(2)
                ade_batch= torch.sqrt(((pred_den- y_den)**2).sum(dim=1)).mean(dim=1)
                total_ade+= ade_batch.sum().item()
                fde_batch= torch.sqrt(((pred_den- y_den)**2).sum(dim=1))[:, -1]
                total_fde+= fde_batch.sum().item()
                rmse_batch = torch.sqrt(torch.mean((pred_den - y_den) ** 2, dim=[1, 2]))
                total_rmse += rmse_batch.sum().item()
                total_samp+= B_
        avg_ade= total_ade/total_samp if total_samp>0 else 0.0
        avg_fde= total_fde/total_samp if total_samp>0 else 0.0
        avg_rmse = total_rmse / total_samp if total_samp > 0 else 0.0
        print(f"[Test] ADE={avg_ade:.4f}, FDE={avg_fde:.4f}, RMSE={avg_rmse:.4f}")
        visualize_test_samples(ddp_model.module, test_dataset, device,
                               num_samples=5, prefix="test_sample", save_dir="6_30_test_visualization")

def main():
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    args= {
        "all_data_pkl": "/home/user/MLLM/data/all_data.pkl",
        "seq_len": 6,
        "out_len": 30,
        "batch_size": 16,
        "epochs": 210,
        "lr": 5e-4,
        "stride": 6,
        "downsample": 5,
        "max_step": 50.0,
        "max_speed_diff": 30.0,
        "image_width": 3840,
        "image_height": 2160,
        "max_polygon_points":64,
        "base_model_name": "meta-llama/Llama-3.2-1B",
        "vision_dim":512,
        "qformer_hidden_size":768,
        "qformer_nhead":8,
        "qformer_enc_layers":4,
        "qformer_dec_layers":4,
        "qformer_num_query_tokens":16,
        "ltsf_nhead":2,
        "ltsf_dropout":0.1,
        "d_model":64,
        "mllm_ckpt":"/home/user/MLLM/models/mllm_lora_ddp_finetuned.pt"
    }
    train_ddp(args)

if __name__=="__main__":
    main()