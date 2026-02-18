"""Flow matching scheduler: rectified flow forward process and Euler ODE sampling."""

from __future__ import annotations

import torch
from scipy.optimize import linear_sum_assignment


class FlowMatchingScheduler:
    """Rectified Flow scheduler for v-prediction.

    Forward process: x_t = (1 - t) * x_0 + t * noise
    Velocity target: v = noise - x_0
    """

    def forward_process(
        self,
        x_0: torch.Tensor,
        t: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Add noise to clean data via rectified flow.

        Args:
            x_0: ``[B, C, T]`` clean mel spectrogram.
            t: ``[B, 1, 1]`` timestep in [0, 1].

        Returns:
            Tuple of (x_t [B, C, T], v_target [B, C, T]).
        """
        noise = torch.randn_like(x_0)
        x_t = (1.0 - t) * x_0 + t * noise
        v_target = noise - x_0
        return x_t, v_target

    def ot_forward_process(
        self,
        x_0: torch.Tensor,
        t: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Add noise via Optimal Transport Conditional Flow Matching.

        Uses minibatch OT to pair data and noise samples, producing
        straighter flow trajectories.

        Args:
            x_0: ``[B, C, T]`` clean mel spectrogram.
            t: ``[B, 1, 1]`` timestep in [0, 1].

        Returns:
            Tuple of (x_t [B, C, T], v_target [B, C, T]).
        """
        noise = torch.randn_like(x_0)
        B = x_0.shape[0]
        if B > 1:
            # Minibatch OT: find optimal pairing between data and noise
            cost = torch.cdist(
                x_0.reshape(B, -1).float(),
                noise.reshape(B, -1).float(),
                p=2,
            ).pow(2)
            _, col_ind = linear_sum_assignment(cost.detach().cpu().numpy())
            noise = noise[col_ind]
        x_t = (1.0 - t) * x_0 + t * noise
        v_target = noise - x_0
        return x_t, v_target

    @staticmethod
    def _sway_timesteps(steps: int, sway_coeff: float) -> torch.Tensor:
        """Compute non-uniform timesteps using sway sampling (F5-TTS).

        Concentrates steps in the mid-range where denoising is most critical.

        Args:
            steps: Number of sampling steps.
            sway_coeff: Sway coefficient. 0 = uniform, >0 = non-uniform (recommended: 1.0).

        Returns:
            ``[steps + 1]`` tensor of timesteps from 1.0 → 0.0.
        """
        indices = torch.linspace(0.0, 1.0, steps + 1)
        timesteps = 1.0 - indices.pow(1.0 + sway_coeff)
        timesteps[-1] = 0.0
        return timesteps

    @torch.no_grad()
    def sample(
        self,
        model: torch.nn.Module,
        shape: tuple[int, ...],
        steps: int = 20,
        device: str = "cpu",
        sway_coefficient: float = 0.0,
        **cond_kwargs,
    ) -> torch.Tensor:
        """Sample from the model using Euler ODE solver.

        Integrates from t=1 (noise) to t=0 (clean) using the predicted velocity.

        Args:
            model: Model that predicts velocity given (x_t, t, **cond).
            shape: Output shape ``(B, C, T)``.
            steps: Number of Euler steps.
            device: Device for computation.
            sway_coefficient: Sway sampling coefficient. 0 = uniform spacing,
                >0 = non-uniform (recommended: 1.0 for improved quality).
            **cond_kwargs: Conditioning arguments passed to model.

        Returns:
            ``[B, C, T]`` generated sample.
        """
        x = torch.randn(shape, device=device)

        if sway_coefficient > 0.0:
            timesteps = self._sway_timesteps(steps, sway_coefficient).to(device)
        else:
            timesteps = torch.linspace(1.0, 0.0, steps + 1, device=device)

        for i in range(steps):
            t_val = timesteps[i]
            dt = timesteps[i] - timesteps[i + 1]
            t = torch.full((shape[0],), t_val.item(), device=device)
            v = model(x, t, **cond_kwargs)
            x = x - v * dt

        return x

    @torch.no_grad()
    def sample_cfg(
        self,
        model: torch.nn.Module,
        shape: tuple[int, ...],
        steps: int = 20,
        device: str = "cpu",
        cfg_scale: float = 1.0,
        sway_coefficient: float = 0.0,
        **cond_kwargs,
    ) -> torch.Tensor:
        """Sample with Classifier-Free Guidance.

        When ``cfg_scale == 1.0``, equivalent to standard ``sample()`` (single pass).
        When ``cfg_scale > 1.0``, uses 2-pass explicit CFG.

        Args:
            model: Model that predicts velocity given (x_t, t, **cond).
            shape: Output shape ``(B, C, T)``.
            steps: Number of Euler steps.
            device: Device for computation.
            cfg_scale: Guidance scale. 1.0 = no guidance, >1.0 = explicit CFG.
            sway_coefficient: Sway sampling coefficient.
            **cond_kwargs: Conditioning arguments passed to model.

        Returns:
            ``[B, C, T]`` generated sample.
        """
        if cfg_scale == 1.0:
            return self.sample(
                model, shape, steps=steps, device=device,
                sway_coefficient=sway_coefficient, **cond_kwargs,
            )

        x = torch.randn(shape, device=device)

        if sway_coefficient > 0.0:
            timesteps = self._sway_timesteps(steps, sway_coefficient).to(device)
        else:
            timesteps = torch.linspace(1.0, 0.0, steps + 1, device=device)

        # Build unconditional kwargs (zero out conditioning tensors)
        uncond_kwargs = {}
        for k, v in cond_kwargs.items():
            if isinstance(v, torch.Tensor):
                uncond_kwargs[k] = torch.zeros_like(v)
            else:
                uncond_kwargs[k] = v

        for i in range(steps):
            t_val = timesteps[i]
            dt = timesteps[i] - timesteps[i + 1]
            t = torch.full((shape[0],), t_val.item(), device=device)
            v_cond = model(x, t, **cond_kwargs)
            v_uncond = model(x, t, **uncond_kwargs)
            v = v_uncond + cfg_scale * (v_cond - v_uncond)
            x = x - v * dt

        return x

    @staticmethod
    def generate_reflow_pairs(
        model: torch.nn.Module,
        x_0: torch.Tensor,
        steps: int = 20,
        **cond_kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate (noise, clean) pairs for Reflow training.

        Runs the ODE forward from x_0 to noise (t=0→1) to create paired data
        for trajectory straightening.

        Args:
            model: Trained teacher model.
            x_0: ``[B, C, T]`` clean mel spectrogram.
            steps: Number of ODE steps for pair generation.
            **cond_kwargs: Conditioning arguments passed to model.

        Returns:
            Tuple of (x_1_noise [B, C, T], x_0_teacher [B, C, T]).
        """
        device = x_0.device
        B = x_0.shape[0]
        x = x_0.clone()
        dt = 1.0 / steps

        with torch.no_grad():
            for i in range(steps):
                t_val = i * dt
                t = torch.full((B,), t_val, device=device)
                v = model(x, t, **cond_kwargs)
                x = x + v * dt

        return x, x_0

    @staticmethod
    def reflow_forward_process(
        x_0_teacher: torch.Tensor,
        x_1_noise: torch.Tensor,
        t: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward process for Reflow training using pre-generated pairs.

        Args:
            x_0_teacher: ``[B, C, T]`` teacher-generated clean data.
            x_1_noise: ``[B, C, T]`` ODE-transported noise.
            t: ``[B, 1, 1]`` timestep in [0, 1].

        Returns:
            Tuple of (x_t [B, C, T], v_target [B, C, T]).
        """
        x_t = (1.0 - t) * x_0_teacher + t * x_1_noise
        v_target = x_1_noise - x_0_teacher
        return x_t, v_target
