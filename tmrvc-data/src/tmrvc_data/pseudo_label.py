"""Pseudo-labeling pipeline for emotion data augmentation.

Uses a pre-trained EmotionClassifier to assign emotion labels to unlabeled
audio data. This addresses the Japanese emotion data scarcity (JVNV ~4h only)
by bootstrapping labels from unlabeled VCTK/JVS/LibriTTS-R data.

Pipeline:
1. Train EmotionClassifier on labeled data (Expresso+JVNV+EmoV-DB+RAVDESS)
2. Apply to unlabeled cache entries (VCTK, JVS, etc.)
3. Keep only high-confidence predictions (threshold=0.8)
4. Write pseudo_emotion.json alongside existing cache entries

Usage::

    from tmrvc_data.pseudo_label import PseudoLabeler
    labeler = PseudoLabeler(classifier_ckpt="checkpoints/emotion_cls.pt")
    stats = labeler.label_dataset(cache_dir, dataset="vctk", split="train")
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from tmrvc_core.constants import N_MELS
from tmrvc_core.dialogue_types import EMOTION_CATEGORIES

logger = logging.getLogger(__name__)


@dataclass
class PseudoLabelStats:
    """Statistics from a pseudo-labeling run."""

    total: int = 0
    labeled: int = 0
    skipped_low_confidence: int = 0
    skipped_existing: int = 0
    errors: int = 0

    @property
    def label_rate(self) -> float:
        return self.labeled / max(self.total, 1)


class PseudoLabeler:
    """Apply a trained emotion classifier to unlabeled cache entries.

    Args:
        classifier_ckpt: Path to trained EmotionClassifier checkpoint.
        confidence_threshold: Minimum softmax confidence to accept a label.
        max_frames: Fixed frame count for batching (crop/pad).
        device: Inference device.
    """

    def __init__(
        self,
        classifier_ckpt: str | Path,
        confidence_threshold: float = 0.8,
        max_frames: int = 200,
        device: str = "cpu",
    ) -> None:
        self.confidence_threshold = confidence_threshold
        self.max_frames = max_frames
        self.device = torch.device(device)

        from tmrvc_train.models.emotion_classifier import EmotionClassifier

        self.classifier = EmotionClassifier()
        ckpt = torch.load(classifier_ckpt, map_location="cpu", weights_only=True)
        self.classifier.load_state_dict(ckpt["classifier"])
        self.classifier.to(self.device)
        self.classifier.eval()
        logger.info("Loaded emotion classifier from %s", classifier_ckpt)

    def label_dataset(
        self,
        cache_dir: str | Path,
        dataset: str,
        split: str = "train",
        overwrite: bool = False,
        batch_size: int = 32,
    ) -> PseudoLabelStats:
        """Apply pseudo-labels to all entries in a dataset.

        Args:
            cache_dir: Feature cache root.
            dataset: Dataset name (e.g. "vctk", "jvs").
            split: Split name.
            overwrite: If True, overwrite existing pseudo_emotion.json.
            batch_size: Inference batch size.

        Returns:
            PseudoLabelStats with counts.
        """
        cache_dir = Path(cache_dir)
        ds_dir = cache_dir / dataset / split
        if not ds_dir.exists():
            logger.warning("Dataset dir not found: %s", ds_dir)
            return PseudoLabelStats()

        # Collect all utterance directories
        utt_dirs = []
        for spk_dir in sorted(ds_dir.iterdir()):
            if not spk_dir.is_dir():
                continue
            for utt_dir in sorted(spk_dir.iterdir()):
                if not utt_dir.is_dir():
                    continue
                mel_path = utt_dir / "mel.npy"
                if mel_path.exists():
                    utt_dirs.append(utt_dir)

        stats = PseudoLabelStats(total=len(utt_dirs))
        logger.info(
            "Found %d utterances in %s/%s", len(utt_dirs), dataset, split
        )

        # Process in batches
        for i in range(0, len(utt_dirs), batch_size):
            batch_dirs = utt_dirs[i : i + batch_size]
            self._process_batch(batch_dirs, stats, overwrite)

            if (i + batch_size) % (batch_size * 10) == 0:
                logger.info(
                    "Progress: %d/%d (labeled=%d, skipped=%d)",
                    min(i + batch_size, len(utt_dirs)),
                    len(utt_dirs),
                    stats.labeled,
                    stats.skipped_low_confidence,
                )

        logger.info(
            "Done: total=%d, labeled=%d (%.1f%%), low_conf=%d, existing=%d, errors=%d",
            stats.total,
            stats.labeled,
            stats.label_rate * 100,
            stats.skipped_low_confidence,
            stats.skipped_existing,
            stats.errors,
        )
        return stats

    def _process_batch(
        self,
        utt_dirs: list[Path],
        stats: PseudoLabelStats,
        overwrite: bool,
    ) -> None:
        """Process a batch of utterance directories."""
        mels = []
        valid_dirs = []

        for utt_dir in utt_dirs:
            pseudo_path = utt_dir / "pseudo_emotion.json"
            if pseudo_path.exists() and not overwrite:
                stats.skipped_existing += 1
                continue

            try:
                mel = np.load(utt_dir / "mel.npy")  # [80, T]
                mel = self._normalize_frames(mel)
                mels.append(mel)
                valid_dirs.append(utt_dir)
            except Exception as e:
                logger.debug("Failed to load %s: %s", utt_dir, e)
                stats.errors += 1

        if not mels:
            return

        # Stack and predict
        mel_batch = torch.from_numpy(np.stack(mels)).float().to(self.device)
        predictions = self.classifier.predict(mel_batch)

        # Write results
        for j, utt_dir in enumerate(valid_dirs):
            conf = predictions["confidence"][j].item()
            if conf < self.confidence_threshold:
                stats.skipped_low_confidence += 1
                continue

            emotion_id = predictions["emotion_ids"][j].item()
            emotion = EMOTION_CATEGORIES[emotion_id]
            vad = predictions["vad"][j].cpu().tolist()
            probs = predictions["emotion_probs"][j].cpu().tolist()

            pseudo = {
                "emotion_id": emotion_id,
                "emotion": emotion,
                "confidence": round(conf, 4),
                "vad": [round(v, 4) for v in vad],
                "probs": {
                    EMOTION_CATEGORIES[k]: round(p, 4)
                    for k, p in enumerate(probs)
                    if p > 0.01
                },
                "source": "pseudo_label",
            }

            pseudo_path = utt_dir / "pseudo_emotion.json"
            with open(pseudo_path, "w", encoding="utf-8") as f:
                json.dump(pseudo, f, ensure_ascii=False, indent=2)
            stats.labeled += 1

    def _normalize_frames(self, mel: np.ndarray) -> np.ndarray:
        """Crop or pad mel to fixed frame count."""
        T = mel.shape[1]
        if T > self.max_frames:
            # Random crop
            start = np.random.randint(0, T - self.max_frames)
            return mel[:, start : start + self.max_frames]
        elif T < self.max_frames:
            pad = np.zeros((mel.shape[0], self.max_frames - T), dtype=mel.dtype)
            return np.concatenate([mel, pad], axis=1)
        return mel


def train_emotion_classifier(
    cache_dir: str | Path,
    datasets: list[str],
    output_path: str | Path,
    max_frames: int = 200,
    batch_size: int = 64,
    max_steps: int = 10_000,
    lr: float = 1e-3,
    device: str = "cpu",
) -> Path:
    """Train an EmotionClassifier on labeled emotion datasets.

    Args:
        cache_dir: Feature cache root containing emotion datasets.
        datasets: List of dataset names with emotion.json labels.
        output_path: Where to save the trained classifier checkpoint.
        max_frames: Fixed mel frame count.
        batch_size: Training batch size.
        max_steps: Maximum training steps.
        lr: Learning rate.
        device: Training device.

    Returns:
        Path to saved checkpoint.
    """
    from tmrvc_data.emotion_dataset import EmotionDataset, create_emotion_dataloader
    from tmrvc_data.cache import FeatureCache
    from tmrvc_train.models.emotion_classifier import EmotionClassifier

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    device_obj = torch.device(device)

    cache = FeatureCache(Path(cache_dir))
    dataset = EmotionDataset(cache, datasets=datasets, split="train", max_frames=max_frames)
    if len(dataset) == 0:
        raise ValueError(f"No emotion data found for datasets={datasets}")

    dataloader = create_emotion_dataloader(
        cache, datasets=datasets, split="train",
        batch_size=batch_size, max_frames=max_frames,
    )

    classifier = EmotionClassifier().to(device_obj)
    optimizer = torch.optim.AdamW(classifier.parameters(), lr=lr, weight_decay=0.01)

    logger.info(
        "Training EmotionClassifier: %d samples, max_steps=%d",
        len(dataset), max_steps,
    )

    classifier.train()
    step = 0
    best_loss = float("inf")

    while step < max_steps:
        for batch in dataloader:
            if step >= max_steps:
                break

            mel = batch["mel"].to(device_obj)
            emotion_id = batch["emotion_id"].to(device_obj)
            vad = batch["vad"].to(device_obj)

            out = classifier(mel)
            cls_loss = torch.nn.functional.cross_entropy(
                out["emotion_logits"], emotion_id
            )
            vad_loss = torch.nn.functional.mse_loss(out["vad"], vad)
            total = cls_loss + 0.5 * vad_loss

            optimizer.zero_grad()
            total.backward()
            torch.nn.utils.clip_grad_norm_(classifier.parameters(), 1.0)
            optimizer.step()

            step += 1
            if step % 100 == 0:
                logger.info(
                    "Step %d: total=%.4f cls=%.4f vad=%.4f",
                    step, total.item(), cls_loss.item(), vad_loss.item(),
                )
                if total.item() < best_loss:
                    best_loss = total.item()

    # Save
    torch.save({"classifier": classifier.state_dict(), "step": step}, output_path)
    logger.info("Saved EmotionClassifier to %s (step=%d)", output_path, step)
    return output_path
