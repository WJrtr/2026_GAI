"""EDM training loss + Karras/Heun sampler with classifier-free guidance.

Reference: Karras et al. 2022, "Elucidating the Design Space of Diffusion-Based
Generative Models" (EDM). Loss and sampler implemented from scratch (no
diffusers).
"""
import torch


class EDM:
    def __init__(self, sigma_data=0.5, P_mean=-1.2, P_std=1.2,
                 sigma_min=0.002, sigma_max=80.0, rho=7.0):
        self.sigma_data = sigma_data
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.rho = rho

    # ---------------- training ----------------
    def loss(self, model, x, cond):
        """x in [-1,1], cond = dict of conditioning tensors (CFG already applied)."""
        B = x.shape[0]
        sigma = (torch.randn(B, device=x.device) * self.P_std + self.P_mean).exp()
        s = sigma.view(-1, 1, 1, 1)
        n = torch.randn_like(x) * s
        D = model(x + n, sigma, **cond)
        weight = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2
        return (weight.view(-1, 1, 1, 1) * (D - x) ** 2).mean()

    # ---------------- sampling ----------------
    def sigmas(self, n, device):
        ramp = torch.linspace(0, 1, n, device=device)
        min_inv = self.sigma_min ** (1 / self.rho)
        max_inv = self.sigma_max ** (1 / self.rho)
        sig = (max_inv + ramp * (min_inv - max_inv)) ** self.rho
        return torch.cat([sig, sig.new_zeros(1)])

    @staticmethod
    def _denoise_cfg(model, x, sigma_scalar, cond, uncond, guidance):
        B = x.shape[0]
        sig = torch.full((B,), float(sigma_scalar), device=x.device)
        if guidance == 1.0 or uncond is None:
            return model(x, sig, **cond)
        # batch cond + uncond in one forward
        x2 = torch.cat([x, x], 0)
        sig2 = torch.cat([sig, sig], 0)
        merged = {k: torch.cat([cond[k], uncond[k]], 0) for k in cond}
        D = model(x2, sig2, **merged)
        Dc, Du = D.chunk(2, 0)
        return Du + guidance * (Dc - Du)

    @torch.no_grad()
    def sample(self, model, shape, cond, uncond=None, steps=30, guidance=1.0,
               device="cuda", churn=0.0):
        sigmas = self.sigmas(steps, device)
        x = torch.randn(shape, device=device) * sigmas[0]
        for i in range(steps):
            s, s_next = sigmas[i], sigmas[i + 1]
            D = self._denoise_cfg(model, x, s, cond, uncond, guidance)
            d = (x - D) / s
            x_next = x + (s_next - s) * d
            if s_next > 0:  # Heun 2nd-order correction
                D2 = self._denoise_cfg(model, x_next, s_next, cond, uncond, guidance)
                d2 = (x_next - D2) / s_next
                x_next = x + (s_next - s) * 0.5 * (d + d2)
            x = x_next
        return x.clamp(-1, 1)
