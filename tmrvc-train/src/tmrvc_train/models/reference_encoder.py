"""Reference Encoder for prosody target extraction from ground-truth audio.

Follows the Global Style Token (GST) reference encoder pattern:
Mel spectrogram -> 6-layer CNN (with batch norm) -> GRU -> Linear -> d_prosody.

Used during training to extract prosody targets from ground-truth audio
when pre-computed prosody_targets.npy files are not available.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from tmrvc_core.constants import D_MODEL


class ReferenceEncoder(nn.Module):
    """Extracts prosody latent from target waveform for flow-matching training.

    Architecture (GST-style):
        1. Mel spectrogram input [B, n_mels, T_mel]
        2. 6 CNN layers with batch norm, ReLU, and stride-2 downsampling
        3. Bidirectional GRU over the downsampled feature sequence
        4. Linear projection to d_prosody

    The output is a fixed-size [B, d_prosody] prosody latent that summarises
    the global prosodic characteristics (pitch contour, energy envelope,
    speaking rate) of the input utterance.
    """

    def __init__(
        self,
        d_model: int = D_MODEL,
        d_prosody: int = 128,
        n_mels: int = 80,
        n_cnn_layers: int = 6,
        cnn_channels: tuple[int, ...] = (32, 32, 64, 64, 128, 128),
        cnn_kernel_size: int = 3,
        gru_hidden: int = 128,
    ):
        super().__init__()
        self.d_prosody = d_prosody
        self.n_mels = n_mels

        if len(cnn_channels) != n_cnn_layers:
            raise ValueError(
                f"cnn_channels length ({len(cnn_channels)}) must match "
                f"n_cnn_layers ({n_cnn_layers})"
            )

        # Build CNN stack: each layer halves the time dimension via stride=2
        cnn_layers: list[nn.Module] = []
        in_ch = 1  # mel spectrogram treated as single-channel 2D input
        for i, out_ch in enumerate(cnn_channels):
            cnn_layers.extend([
                nn.Conv2d(
                    in_ch, out_ch,
                    kernel_size=cnn_kernel_size,
                    stride=2,
                    padding=cnn_kernel_size // 2,
                ),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            ])
            in_ch = out_ch

        self.cnn = nn.Sequential(*cnn_layers)

        # After 6 layers of stride-2, mel bins are reduced by factor 2^6 = 64
        # n_mels=80 -> ceil(80/64) = 2 remaining frequency bins (approximately)
        # We compute this dynamically in forward to handle variable n_mels
        self._cnn_out_channels = cnn_channels[-1]

        # GRU operates over the time axis of the CNN output
        # Input size = cnn_out_channels * remaining_freq_bins (computed dynamically)
        self.gru = nn.GRU(
            input_size=1,  # placeholder, set properly via lazy init
            hidden_size=gru_hidden,
            batch_first=True,
            bidirectional=True,
        )
        self._gru_input_initialized = False
        self._gru_hidden = gru_hidden

        # Final projection: bidirectional GRU output -> d_prosody
        self.output_proj = nn.Linear(gru_hidden * 2, d_prosody)

    def _init_gru_input(self, freq_bins: int) -> None:
        """Lazily initialize GRU input size based on actual CNN output shape."""
        if self._gru_input_initialized:
            return
        gru_input_size = self._cnn_out_channels * freq_bins
        device = self.output_proj.weight.device
        self.gru = nn.GRU(
            input_size=gru_input_size,
            hidden_size=self._gru_hidden,
            batch_first=True,
            bidirectional=True,
        ).to(device)
        self._gru_input_initialized = True

    def forward(self, mel_spectrogram: torch.Tensor) -> torch.Tensor:
        """Extract prosody latent from mel spectrogram.

        Args:
            mel_spectrogram: [B, n_mels, T_mel] or [B, 1, n_mels, T_mel].
                If 3D, it is reshaped to [B, 1, n_mels, T_mel] internally.

        Returns:
            prosody_latent: [B, d_prosody] fixed-size prosody embedding.
        """
        x = mel_spectrogram
        if x.ndim == 3:
            # [B, n_mels, T] -> [B, 1, n_mels, T]
            x = x.unsqueeze(1)

        # CNN: [B, 1, n_mels, T] -> [B, C, F', T']
        x = self.cnn(x)
        B, C, F_out, T_out = x.shape

        # Lazy GRU input init
        self._init_gru_input(F_out)

        # Reshape for GRU: [B, T', C*F']
        x = x.permute(0, 3, 1, 2).reshape(B, T_out, C * F_out)

        # GRU: [B, T', C*F'] -> [B, T', 2*gru_hidden]
        self.gru.flatten_parameters()
        gru_out, _ = self.gru(x)

        # Take the last time step output (summary of entire sequence)
        last_output = gru_out[:, -1, :]  # [B, 2*gru_hidden]

        # Project to prosody space
        prosody_latent = self.output_proj(last_output)  # [B, d_prosody]

        return prosody_latent


class ReferenceEncoderFromWaveform(nn.Module):
    """Convenience wrapper: waveform -> mel spectrogram -> ReferenceEncoder.

    Computes mel spectrogram on the fly from raw audio waveforms, then
    delegates to ReferenceEncoder for prosody extraction.
    """

    def __init__(
        self,
        d_model: int = D_MODEL,
        d_prosody: int = 128,
        n_mels: int = 80,
        sample_rate: int = 24000,
        n_fft: int = 1024,
        hop_length: int = 256,
        win_length: int = 1024,
        f_min: float = 0.0,
        f_max: float = 8000.0,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.n_mels = n_mels
        self.f_min = f_min
        self.f_max = f_max

        self.ref_encoder = ReferenceEncoder(
            d_model=d_model, d_prosody=d_prosody, n_mels=n_mels
        )

        # Register mel filterbank as a buffer (not a parameter)
        mel_fb = self._create_mel_filterbank()
        self.register_buffer("mel_fb", mel_fb)

    def _create_mel_filterbank(self) -> torch.Tensor:
        """Create a mel filterbank matrix [n_mels, n_fft//2 + 1]."""
        n_freqs = self.n_fft // 2 + 1

        # Mel scale conversion helpers
        def hz_to_mel(hz: float) -> float:
            return 2595.0 * (1.0 + hz / 700.0).__class__(1.0 + hz / 700.0)

        def mel_to_hz(mel: float) -> float:
            return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

        # Use simple triangular filterbank
        import math

        low_mel = 2595.0 * math.log10(1.0 + self.f_min / 700.0)
        high_mel = 2595.0 * math.log10(1.0 + self.f_max / 700.0)
        mel_points = torch.linspace(low_mel, high_mel, self.n_mels + 2)
        hz_points = 700.0 * (10.0 ** (mel_points / 2595.0) - 1.0)
        bin_points = (hz_points * self.n_fft / self.sample_rate).long()

        fb = torch.zeros(self.n_mels, n_freqs)
        for i in range(self.n_mels):
            left = bin_points[i]
            center = bin_points[i + 1]
            right = bin_points[i + 2]

            # Rising slope
            for j in range(left, center):
                if j < n_freqs and center > left:
                    fb[i, j] = (j - left) / (center - left)
            # Falling slope
            for j in range(center, right):
                if j < n_freqs and right > center:
                    fb[i, j] = (right - j) / (right - center)

        return fb

    def compute_mel(self, waveform: torch.Tensor) -> torch.Tensor:
        """Compute mel spectrogram from waveform.

        Args:
            waveform: [B, T_audio] or [B, 1, T_audio] raw audio.

        Returns:
            mel: [B, n_mels, T_mel] log-mel spectrogram.
        """
        if waveform.ndim == 3:
            waveform = waveform.squeeze(1)

        # STFT
        window = torch.hann_window(self.win_length, device=waveform.device)
        stft = torch.stft(
            waveform,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=window,
            return_complex=True,
        )
        mag = stft.abs()  # [B, n_freqs, T_mel]

        # Apply mel filterbank
        mel = torch.matmul(self.mel_fb.to(mag.device), mag)  # [B, n_mels, T_mel]

        # Log scale with floor
        mel = torch.log(mel.clamp(min=1e-5))

        return mel

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """Extract prosody latent from raw waveform.

        Args:
            waveform: [B, T_audio] or [B, 1, T_audio].

        Returns:
            prosody_latent: [B, d_prosody].
        """
        mel = self.compute_mel(waveform)
        return self.ref_encoder(mel)
