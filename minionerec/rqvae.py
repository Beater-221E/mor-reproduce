"""Residual Quantized VAE for Semantic IDs (paper §3.2)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from minionerec.util import CODEBOOK_SIZE, NUM_CODEBOOK_LAYERS, format_sid


class RQVAE(nn.Module):
    def __init__(
        self,
        in_dim: int,
        latent_dim: int = 32,
        hidden_dim: int = 256,
        num_layers: int = NUM_CODEBOOK_LAYERS,
        codebook_size: int = CODEBOOK_SIZE,
        beta: float = 0.25,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.codebook_size = codebook_size
        self.beta = beta
        self.latent_dim = latent_dim

        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, in_dim),
        )
        self.codebooks = nn.ParameterList(
            [nn.Parameter(torch.randn(codebook_size, latent_dim) * 0.01) for _ in range(num_layers)]
        )
        self._codebooks_initialized = False

    @torch.no_grad()
    def warm_start(self, batch: torch.Tensor) -> None:
        z = self.encoder(batch)
        residual = z
        n = residual.size(0)
        for layer in range(self.num_layers):
            x_np = residual.detach().cpu().numpy()
            k = min(self.codebook_size, n)
            km = KMeans(n_clusters=k, n_init=10, random_state=42)
            km.fit(x_np)
            centers = torch.tensor(km.cluster_centers_, device=batch.device, dtype=batch.dtype)
            if k < self.codebook_size:
                # pad remaining codebook entries with noise around existing centers
                pad = centers[torch.randint(0, k, (self.codebook_size - k,), device=batch.device)]
                pad = pad + 0.01 * torch.randn_like(pad)
                centers = torch.cat([centers, pad], dim=0)
            self.codebooks[layer].copy_(centers)
            dists = torch.cdist(residual, self.codebooks[layer])
            idx = dists.argmin(dim=-1)
            residual = residual - self.codebooks[layer][idx]
        self._codebooks_initialized = True

    def quantize(self, z: torch.Tensor):
        residual = z
        indices = []
        quantized_sum = torch.zeros_like(z)
        vq_loss = z.new_zeros(())
        for layer in range(self.num_layers):
            cb = self.codebooks[layer]
            dists = torch.cdist(residual, cb)
            idx = dists.argmin(dim=-1)
            quantized = cb[idx]
            # commitment + codebook loss
            vq_loss = vq_loss + F.mse_loss(residual.detach(), quantized) + self.beta * F.mse_loss(
                residual, quantized.detach()
            )
            # straight-through
            quantized_st = residual + (quantized - residual).detach()
            quantized_sum = quantized_sum + quantized_st
            residual = residual - quantized
            indices.append(idx)
        return quantized_sum, torch.stack(indices, dim=1), vq_loss

    def forward(self, x: torch.Tensor):
        z = self.encoder(x)
        z_q, indices, vq_loss = self.quantize(z)
        recon = self.decoder(z_q)
        recon_loss = F.mse_loss(recon, x)
        loss = recon_loss + vq_loss
        return loss, recon_loss.detach(), vq_loss.detach(), indices

    @torch.no_grad()
    def encode_indices(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        _, indices, _ = self.quantize(z)
        return indices


def train_rqvae(
    emb_path: Path,
    ids_path: Path,
    out_dir: Path,
    epochs: int = 10000,
    batch_size: int = 2048,
    lr: float = 1e-3,
    latent_dim: int = 32,
    hidden_dim: int = 256,
    beta: float = 0.25,
    device: str = "cuda:0",
    log_every: int = 50,
    early_collision_patience: int = 200,
) -> None:
    emb = np.load(emb_path)
    with open(ids_path, encoding="utf-8") as f:
        item_ids = json.load(f)
    assert len(item_ids) == emb.shape[0]

    # L2-normalize semantic embeddings for stable RQ training
    norms = np.linalg.norm(emb, axis=1, keepdims=True).clip(min=1e-8)
    emb = emb / norms
    x = torch.tensor(emb, dtype=torch.float32)
    ds = TensorDataset(x)
    # large batch as in paper when possible
    bs = min(batch_size, len(ds))
    loader = DataLoader(ds, batch_size=bs, shuffle=True, drop_last=False)

    model = RQVAE(in_dim=emb.shape[1], latent_dim=latent_dim, hidden_dim=hidden_dim, beta=beta).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    # warm-start on first batch
    first = next(iter(loader))[0].to(device)
    model.warm_start(first)

    out_dir.mkdir(parents=True, exist_ok=True)
    best_collision = 1.0
    stale = 0
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.0
        for (batch,) in loader:
            batch = batch.to(device)
            loss, recon_loss, vq_loss, _ = model(batch)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()

        if epoch % log_every == 0 or epoch == 1:
            model.eval()
            with torch.no_grad():
                all_idx = []
                for (batch,) in DataLoader(ds, batch_size=bs, shuffle=False):
                    all_idx.append(model.encode_indices(batch.to(device)).cpu())
                indices = torch.cat(all_idx, dim=0).numpy()
            # collision rate: fraction of items sharing SID with another
            keys = [tuple(row.tolist()) for row in indices]
            unique = len(set(keys))
            collision = 1.0 - unique / len(keys)
            avg_loss = total / max(1, len(loader))
            history.append({"epoch": epoch, "loss": avg_loss, "collision": collision, "unique": unique})
            print(f"epoch={epoch} loss={avg_loss:.6f} collision={collision:.4f} unique={unique}/{len(keys)}")
            if collision < best_collision - 1e-6:
                best_collision = collision
                stale = 0
                torch.save(
                    {"model": model.state_dict(), "in_dim": emb.shape[1], "latent_dim": latent_dim},
                    out_dir / "rqvae.pt",
                )
                # export indices
                sid_map = {}
                used = {}
                for iid, codes in zip(item_ids, indices):
                    code_t = tuple(int(c) for c in codes.tolist())
                    # disambiguate collisions by walking residual free codes if needed
                    if code_t in used:
                        # simple disambiguation: increment last code until free
                        c0, c1, c2 = code_t
                        found = False
                        for alt in range(CODEBOOK_SIZE):
                            cand = (c0, c1, (c2 + alt) % CODEBOOK_SIZE)
                            if cand not in used:
                                code_t = cand
                                found = True
                                break
                        if not found:
                            for alt1 in range(CODEBOOK_SIZE):
                                for alt2 in range(CODEBOOK_SIZE):
                                    cand = (c0, (c1 + alt1) % CODEBOOK_SIZE, alt2)
                                    if cand not in used:
                                        code_t = cand
                                        found = True
                                        break
                                if found:
                                    break
                    used[code_t] = iid
                    sid_map[iid] = {
                        "codes": list(code_t),
                        "sid": format_sid(code_t),
                    }
                with open(out_dir / "sid_map.json", "w", encoding="utf-8") as f:
                    json.dump(sid_map, f, ensure_ascii=False, indent=2)
            else:
                stale += 1
                if stale >= early_collision_patience and best_collision < 0.3:
                    print(f"Early stop at epoch {epoch}, best_collision={best_collision:.4f}")
                    break

    with open(out_dir / "train_history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    print(f"Done. Artifacts in {out_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--emb_path", type=Path, required=True)
    parser.add_argument("--ids_path", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=5000)
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--latent_dim", type=int, default=32)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--beta", type=float, default=0.25)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--log_every", type=int, default=50)
    args = parser.parse_args()
    train_rqvae(
        args.emb_path,
        args.ids_path,
        args.out_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
        beta=args.beta,
        device=args.device,
        log_every=args.log_every,
    )


if __name__ == "__main__":
    main()
