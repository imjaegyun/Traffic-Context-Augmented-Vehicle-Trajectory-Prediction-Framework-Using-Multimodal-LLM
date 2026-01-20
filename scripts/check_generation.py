import os
import pickle
import random

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.nn.parallel as pnl

from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from peft import LoraConfig, TaskType, get_peft_model

###############################################################################
# 1. BLIP-2 스타일의 Q-Former
###############################################################################
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
        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=nhead,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_encoder_layers)
        self.query_tokens = nn.Parameter(torch.randn(num_query_tokens, hidden_size))
        dec_layer = nn.TransformerDecoderLayer(
            d_model=hidden_size,
            nhead=nhead,
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=num_decoder_layers)

    def forward(self, vision_embs):
        B, Tv, _ = vision_embs.shape
        x = self.vision_proj(vision_embs)
        enc_out = self.encoder(x)
        query = self.query_tokens.unsqueeze(0).expand(B, -1, -1)
        dec_out = self.decoder(query, enc_out)
        return dec_out

###############################################################################
# 2. LlamaWithCrossAttnPEFT (LoRA 적용)
###############################################################################
class LlamaWithCrossAttnPEFT(nn.Module):
    def __init__(self,
                 base_model_name,
                 use_lora=True,
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
                task_type=TaskType.CAUSAL_LM
            )
            self.llama_model = get_peft_model(self.llama_model, peft_config)
        self.config = self.llama_model.config
        self.hidden_size = self.config.hidden_size

    def forward(self, inputs_embeds, attention_mask, labels=None):
        outputs = self.llama_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels
        )
        return outputs

###############################################################################
# 3. LlamaMultiModal (Q-Former + LLaMA 결합)
###############################################################################
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
        self.q_hidden_size = q_hidden_size
        self.llama_wrapper = LlamaWithCrossAttnPEFT(
            base_model_name=base_model_name,
            use_lora=use_lora,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout
        )
        self.llama_hidden_size = self.llama_wrapper.hidden_size
        if self.llama_hidden_size != self.q_hidden_size:
            self.q_proj = nn.Linear(q_hidden_size, self.llama_hidden_size)
        else:
            self.q_proj = nn.Identity()
        self.vision_modality_embedding = nn.Parameter(torch.randn(1, 1, self.llama_hidden_size))
        self.text_modality_embedding = nn.Parameter(torch.randn(1, 1, self.llama_hidden_size))

    def forward(self, vision_embs, input_ids, attention_mask, labels=None):
        device = vision_embs.device
        B = vision_embs.size(0)
        image_tokens = self.qformer(vision_embs)
        image_tokens = self.q_proj(image_tokens)
        image_tokens = image_tokens + self.vision_modality_embedding
        text_embeds = self.llama_wrapper.llama_model.get_input_embeddings()(input_ids)
        text_embeds = text_embeds + self.text_modality_embedding
        fused_embeds = torch.cat([image_tokens, text_embeds], dim=1)
        img_len = image_tokens.size(1)
        txt_len = text_embeds.size(1)
        img_mask = torch.ones((B, img_len), dtype=attention_mask.dtype, device=device)
        fused_mask = torch.cat([img_mask, attention_mask], dim=1)
        if labels is not None:
            new_labels = torch.full((B, img_len + txt_len), -100, dtype=labels.dtype, device=device)
            new_labels[:, img_len:] = labels
        else:
            new_labels = None
        outputs = self.llama_wrapper(inputs_embeds=fused_embeds, attention_mask=fused_mask, labels=new_labels)
        return outputs

    def generate_batch(self, vision_embs, prompt_ids, tokenizer, max_new_tokens=128,
                       temperature=0.9, top_k=40, top_p=0.9, device="cuda"):
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
            fused_mask = torch.cat([torch.ones(B, img_len, dtype=prompt_ids.dtype, device=device),
                                    torch.ones_like(prompt_ids)], dim=1)
        orig_embedding_func = self.llama_wrapper.llama_model.get_input_embeddings()
        prefix_embeds = fused_embeds
        prefix_mask = fused_mask
        prefix_length = prefix_embeds.size(1)

        def patched_embedding(ids):
            seq_len = ids.shape[1]
            if seq_len <= prefix_length:
                return prefix_embeds[:, :seq_len, :]
            else:
                prefix_part = prefix_embeds[:, :prefix_length, :]
                new_token_ids = ids[:, prefix_length:]
                new_embeds = orig_embedding_func(new_token_ids)
                return torch.cat([prefix_part, new_embeds], dim=1)

        self.llama_wrapper.llama_model.set_input_embeddings(
            nn.Embedding(orig_embedding_func.num_embeddings, orig_embedding_func.embedding_dim).to(device))
        self.llama_wrapper.llama_model.get_input_embeddings().forward = patched_embedding
        fake_input_ids = prompt_ids.clone()
        fake_attention_mask = prefix_mask
        generation_config = GenerationConfig(
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            do_sample=True,
            no_repeat_ngram_size=3,
            repetition_penalty=1.2,
            eos_token_id=tokenizer.eos_token_id
        )
        with torch.no_grad():
            outputs = self.llama_wrapper.llama_model.generate(
                input_ids=fake_input_ids,
                attention_mask=fake_attention_mask,
                generation_config=generation_config
            )
        self.llama_wrapper.llama_model.set_input_embeddings(orig_embedding_func)
        # 생성된 텍스트 디코딩 및 후처리
        generated_texts = []
        for out_ids in outputs:
            text = tokenizer.decode(out_ids, skip_special_tokens=True)
            # "Answer:" 이후의 텍스트만 추출하고 불필요한 부분 제거
            if "Answer:" in text:
                answer_start = text.index("Answer:") + len("Answer:")
                answer_text = text[answer_start:].strip()
                # 문장이 끝나는 지점(마침표, 개행 등) 이후 제거
                end_chars = [".", "\n"]
                end_idx = len(answer_text)
                for char in end_chars:
                    idx = answer_text.find(char, answer_text.find("No right-following vehicle") + len("No right-following vehicle"))
                    if idx != -1 and idx < end_idx:
                        end_idx = idx + 1
                clean_text = answer_text[:end_idx].strip()
                generated_texts.append(clean_text)
            else:
                generated_texts.append(text)
        return generated_texts

###############################################################################
# 4. Dataset / Collate
###############################################################################
class VisionTextDataset(Dataset):
    def __init__(self, data_list, tokenizer, max_length=512):
        self.data_list = data_list
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        item = self.data_list[idx]
        vision_emb = item["vision_embeddings"]
        if not isinstance(vision_emb, torch.Tensor):
            vision_emb = torch.tensor(vision_emb, dtype=torch.float32)
        else:
            vision_emb = vision_emb.to(torch.float32)
        vision_emb = vision_emb.cpu()
        track_id = item.get("track_id", "unknown")
        answer_text = item["context_str"]
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
        prompt_enc = self.tokenizer(prompt_text, truncation=True, max_length=self.max_length,
                                    return_tensors="pt", add_special_tokens=False)
        answer_enc = self.tokenizer(answer_text, truncation=True, max_length=self.max_length,
                                    return_tensors="pt", add_special_tokens=False)
        input_ids = torch.cat([prompt_enc["input_ids"], answer_enc["input_ids"]], dim=1)
        attention_mask = torch.cat([prompt_enc["attention_mask"], answer_enc["attention_mask"]], dim=1)
        labels = torch.full_like(input_ids, -100)
        prompt_len = prompt_enc["input_ids"].size(1)
        labels[:, prompt_len:] = input_ids[:, prompt_len:]
        if input_ids.size(1) > self.max_length:
            input_ids = input_ids[:, :self.max_length]
            attention_mask = attention_mask[:, :self.max_length]
            labels = labels[:, :self.max_length]
        return {
            "track_id": track_id,
            "vision_emb": vision_emb,
            "input_ids": input_ids.squeeze(0),
            "attention_mask": attention_mask.squeeze(0),
            "labels": labels.squeeze(0),
            "reference_text": answer_text
        }

def vision_text_collate_fn(batch):
    emb_dim = batch[0]["vision_emb"].size(-1)
    Tv_list = [b["vision_emb"].size(0) for b in batch]
    Tv_max = max(Tv_list)
    vision_list = []
    track_id_list = []
    ref_text_list = []
    for b in batch:
        v = b["vision_emb"]
        track_id_list.append(b["track_id"])
        ref_text_list.append(b["reference_text"])
        pad_len = Tv_max - v.size(0)
        if pad_len > 0:
            pad_zeros = torch.zeros(pad_len, emb_dim, dtype=v.dtype)
            v_pad = torch.cat([v, pad_zeros], dim=0)
        else:
            v_pad = v
        vision_list.append(v_pad.unsqueeze(0))
    vision_batch = torch.cat(vision_list, dim=0)
    input_ids_list = [b["input_ids"] for b in batch]
    attn_mask_list = [b["attention_mask"] for b in batch]
    labels_list = [b["labels"] for b in batch]
    input_ids_pad = pad_sequence(input_ids_list, batch_first=True, padding_value=0)
    attn_mask_pad = pad_sequence(attn_mask_list, batch_first=True, padding_value=0)
    labels_pad = pad_sequence(labels_list, batch_first=True, padding_value=-100)
    return {
        "track_id": track_id_list,
        "vision_emb": vision_batch,
        "input_ids": input_ids_pad,
        "attention_mask": attn_mask_pad,
        "labels": labels_pad,
        "reference_text": ref_text_list,
    }

###############################################################################
# 5. Train/Val/Test Split
###############################################################################
def split_dataset(data_list, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1, seed=42):
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6
    random.seed(seed)
    random.shuffle(data_list)
    N = len(data_list)
    train_end = int(N * train_ratio)
    val_end = train_end + int(N * val_ratio)
    train_data = data_list[:train_end]
    val_data = data_list[train_end:val_end]
    test_data = data_list[val_end:]
    return train_data, val_data, test_data

###############################################################################
# 6. 분산 환경에서 Test 평가
###############################################################################
def distributed_generate_and_save_test(ddp_model, dataset, tokenizer, output_file,
                                       device="cuda", batch_size=8, max_new_tokens=128,
                                       top_k=40, top_p=0.9, temperature=0.9):
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    sampler = DistributedSampler(dataset, shuffle=False)
    sampler.set_epoch(0)
    loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, collate_fn=vision_text_collate_fn)
    ddp_model.eval()
    local_results = []
    with torch.no_grad():
        for batch_data in loader:
            track_ids = batch_data["track_id"]
            ref_texts = batch_data["reference_text"]
            vision_emb = batch_data["vision_emb"].to(device)
            input_ids = batch_data["input_ids"].to(device)
            gen_texts = ddp_model.module.generate_batch(
                vision_embs=vision_emb,
                prompt_ids=input_ids,
                tokenizer=tokenizer,
                max_new_tokens=max_new_tokens,
                device=device,
                top_k=top_k,
                top_p=top_p,
                temperature=temperature
            )
            for t_id, pred, ref in zip(track_ids, gen_texts, ref_texts):
                local_results.append((t_id, pred, ref))
    gathered_data = [None for _ in range(world_size)]
    dist.all_gather_object(gathered_data, local_results)
    if rank == 0:
        all_data = []
        for each_rank_list in gathered_data:
            all_data.extend(each_rank_list)
        with open(output_file, "w", encoding="utf-8") as f:
            for t_id, pred, ref in all_data:
                f.write(f"[track_id={t_id}] Generated:\n{pred}\n---\nReference:\n{ref}\n\n")
        print(f"[Rank0] Test generation results saved to {output_file}")
    dist.barrier()

###############################################################################
# 7. 메인 테스트 함수 (학습 부분 주석 처리)
###############################################################################
def test_mllm_lora_ddp():
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    dist.init_process_group(backend='nccl', rank=local_rank, world_size=world_size)
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    # 하이퍼파라미터
    base_model_name = "meta-llama/Llama-3.2-1B"
    use_lora = True
    lora_r = 8
    lora_alpha = 32
    lora_dropout = 0.1
    vision_dim = 512
    batch_size = 8
    max_length = 1024
    data_path = "/home/user/MLLM/data/all_data.pkl"
    checkpoint_path = "/home/user/MLLM/models/mllm_lora_ddp_finetuned.pt"
    output_file = "test_generation_results.txt"

    if local_rank == 0:
        print(f"[Rank 0] Loading data from {data_path}...")
    with open(data_path, "rb") as f:
        all_data = pickle.load(f)

    _, _, test_data = split_dataset(all_data, 0.7, 0.2, 0.1, seed=42)

    if local_rank == 0:
        print(f"Total: {len(all_data)}, Test={len(test_data)}")

    if local_rank == 0:
        print(f"[Rank 0] Loading tokenizer for {base_model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    test_dataset = VisionTextDataset(test_data, tokenizer, max_length)

    if local_rank == 0:
        print("[Rank0] Initializing model and loading checkpoint...")
    model = LlamaMultiModal(
        base_model_name=base_model_name,
        use_lora=use_lora,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        vision_dim=vision_dim,
        q_hidden_size=768,
        q_nhead=8,
        q_enc_layers=4,
        q_dec_layers=4,
        q_num_query_tokens=16
    ).to(device)
    
    # 체크포인트 로드
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint, strict=False)
    
    ddp_model = pnl.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank,
                                            find_unused_parameters=True)

    # 테스트만 진행
    if local_rank == 0:
        print(f"[Main] Generating text on TEST and saving to {output_file} ...")
    distributed_generate_and_save_test(
        ddp_model,
        test_dataset,
        tokenizer,
        output_file=output_file,
        device=device,
        batch_size=batch_size,
        max_new_tokens=128,
        top_k=40,
        top_p=0.9,
        temperature=0.9
    )

    dist.destroy_process_group()

if __name__ == "__main__":
    """
    Usage:
      torchrun --nproc_per_node=2 test_mllm_lora_ddp.py
    """
    test_mllm_lora_ddp()