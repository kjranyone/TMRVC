import torch
import torch.nn as nn
import random
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Optional

from .models import DisentangledUCLM, uclm_loss

class UCLMTrainer:
    """Trainer for Disentangled UCLM (TTS & VC).
    
    Supports:
        - Multi-task training (randomly switch between TTS and VC)
        - Classifier-Free Guidance (CFG) dropout
        - Adversarial disentanglement loss
        - Duration prediction loss (TTS mode)
    """
    def __init__(
        self, 
        model: DisentangledUCLM, 
        optimizer: torch.optim.Optimizer, 
        device: str = "cuda",
        tts_prob: float = 0.5
    ):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.device = device
        self.tts_prob = tts_prob
        
    def train_step(self, batch: dict) -> dict:
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        
        # Move common tensors to device
        target_a = batch["target_a"].to(self.device)
        target_b = batch["target_b"].to(self.device)
        explicit_state = batch["explicit_state"].to(self.device)
        ssl_state = batch["ssl_state"].to(self.device)
        speaker_embed = batch["speaker_embed"].to(self.device)
        speaker_labels = batch["speaker_id"].to(self.device)
        f0_condition = batch.get("f0_condition")
        if f0_condition is not None:
            f0_condition = f0_condition.to(self.device)
            
        # Classifier-Free Guidance Dropout (15% chance to drop conditions)
        cfg_scale = 1.0
        if random.random() < 0.15:
            explicit_state = torch.zeros_like(explicit_state)
            ssl_state = torch.zeros_like(ssl_state)
            speaker_embed = torch.zeros_like(speaker_embed)
            
        # Multi-task sampling
        mode = "vc"
        if batch.get("phoneme_ids") is not None and random.random() < self.tts_prob:
            mode = "tts"
        
        if mode == "vc":
            source_a_t = batch["source_a_t"].to(self.device)
            source_mask = (target_a[:, 0, :] != -1)
            out = self.model.forward_vc(
                source_a_t=source_a_t,
                target_b=target_b,
                explicit_state=explicit_state,
                ssl_state=ssl_state,
                speaker_embed=speaker_embed,
                source_mask=source_mask,
                f0_condition=f0_condition,
                cfg_scale=cfg_scale
            )
            
            losses = uclm_loss(
                logits_a=out["logits_a"],
                logits_b=out["logits_b"],
                target_a=target_a,
                target_b=target_b,
                vq_loss=out.get("vq_loss"),
                adv_logits=out.get("adv_logits"),
                speaker_labels=speaker_labels
            )
            
        else: # tts
            phonemes = batch["phoneme_ids"].to(self.device)
            phoneme_lens = batch["phoneme_lens"].to(self.device)
            language_ids = batch["language_id"].to(self.device)
            durations = batch["durations"].to(self.device)
            
            out = self.model.forward_tts(
                phonemes=phonemes,
                phoneme_lens=phoneme_lens,
                language_ids=language_ids,
                target_b=target_b,
                explicit_state=explicit_state,
                ssl_state=ssl_state,
                speaker_embed=speaker_embed,
                durations=durations,
                f0_condition=f0_condition,
                cfg_scale=cfg_scale
            )
            
            # Mask for durations (padded phonemes)
            L = phonemes.shape[1]
            phoneme_mask = (
                torch.arange(L, device=self.device).unsqueeze(0)
                >= phoneme_lens.unsqueeze(1)
            )
            
            losses = uclm_loss(
                logits_a=out["logits_a"],
                logits_b=out["logits_b"],
                target_a=target_a,
                target_b=target_b,
                log_durations=out.get("log_durations"),
                dur_target=durations,
                phoneme_mask=phoneme_mask,
                adv_logits=out.get("adv_logits"),
                speaker_labels=speaker_labels
            )
            
        # Optimization
        losses["loss"].backward()
        self.optimizer.step()
        
        res = {k: v.item() for k, v in losses.items()}
        res["mode"] = 1 if mode == "tts" else 0
        return res

    def train_epoch(self, dataloader: DataLoader):
        total_loss = 0
        pbar = tqdm(dataloader, desc="Training")
        for batch in pbar:
            metrics = self.train_step(batch)
            total_loss += metrics["loss"]
            pbar.set_postfix({"loss": metrics["loss"], "mode": "TTS" if metrics["mode"] else "VC"})
        return total_loss / len(dataloader)
