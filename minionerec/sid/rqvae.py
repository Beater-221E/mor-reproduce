"""Residual Quantized VAE for Semantic IDs (paper §3.2).

Improvements vs naive VQ:
  - PCA dimensionality reduction before RQ
  - KMeans warm-start on a large sample (not a single mini-batch)
  - Periodic dead-code reset from high-residual samples
  - Per-layer codebook usage logging
  - Early stop on collision plateau
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader, TensorDataset

from minionerec.constants import CODEBOOK_SIZE, NUM_CODEBOOK_LAYERS
from minionerec.sid.codec import format_sid


class RQVAE(nn.Module):
    def __init__(
        self,
        in_dim: int,
        latent_dim: int = 64,
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
        """KMeans init on ``batch`` (preferably a large / full-data sample)."""
        z = self.encoder(batch)
        residual = z
        n = residual.size(0)
        for layer in range(self.num_layers):
            x_np = residual.detach().cpu().numpy()
            k = min(self.codebook_size, n)
            km = KMeans(n_clusters=k, n_init=10, random_state=42 + layer)
            km.fit(x_np)
            centers = torch.tensor(km.cluster_centers_, device=batch.device, dtype=batch.dtype)
            if k < self.codebook_size:
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
            vq_loss = vq_loss + F.mse_loss(residual.detach(), quantized) + self.beta * F.mse_loss(
                residual, quantized.detach()
            )
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

    @torch.no_grad()
    def layer_residuals(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Return residual vectors entering each codebook layer."""
        z = self.encoder(x)
        residual = z
        residuals = []
        for layer in range(self.num_layers):
            residuals.append(residual.clone())
            dists = torch.cdist(residual, self.codebooks[layer])
            idx = dists.argmin(dim=-1)
            residual = residual - self.codebooks[layer][idx]
        return residuals

    @torch.no_grad()
    def reset_dead_codes(self, x: torch.Tensor, usage: list[np.ndarray], min_count: int = 1) -> list[int]:
        """
        Replace under-used codebook entries with high-norm residuals from ``x``.
        Returns number of resets per layer.
        """
        residuals = self.layer_residuals(x)
        resets = []
        for layer, (res, counts) in enumerate(zip(residuals, usage)):
            dead = np.where(counts < min_count)[0]
            n_reset = 0
            if len(dead) == 0:
                resets.append(0)
                continue
            # prefer high-norm residuals (poorly fit points)
            norms = res.norm(dim=-1)
            # sample with replacement if needed
            order = torch.argsort(norms, descending=True)
            pick = order[: max(len(dead), 1)]
            for i, code_id in enumerate(dead):
                src = pick[i % len(pick)]
                noise = 0.01 * torch.randn_like(res[src])
                self.codebooks[layer][int(code_id)].copy_(res[src] + noise)
                n_reset += 1
            resets.append(n_reset)
        return resets


def _apply_pca(emb: np.ndarray, pca_dim: int) -> tuple[np.ndarray, dict]:
    """L2-normalize → PCA → L2-normalize. Returns features and serializable PCA stats."""
    norms = np.linalg.norm(emb, axis=1, keepdims=True).clip(min=1e-8)
    emb_n = emb / norms
    dim = min(int(pca_dim), emb_n.shape[0], emb_n.shape[1])
    pca = PCA(n_components=dim, random_state=42)
    z = pca.fit_transform(emb_n)
    z_norms = np.linalg.norm(z, axis=1, keepdims=True).clip(min=1e-8)
    z = z / z_norms
    meta = {
        "pca_dim": dim,
        "explained_variance_ratio_sum": float(pca.explained_variance_ratio_.sum()),
        "original_dim": int(emb.shape[1]),
    }
    return z.astype(np.float32), meta


def _export_sid_map(item_ids: list, indices: np.ndarray) -> dict:
    sid_map: dict = {}
    used: dict = {}
    for iid, codes in zip(item_ids, indices):
        code_t = tuple(int(c) for c in codes.tolist())
        if code_t in used:
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
        sid_map[iid] = {"codes": list(code_t), "sid": format_sid(code_t)}
    return sid_map


def _enforce_unique_last_code(indices: np.ndarray, codebook_size: int = CODEBOOK_SIZE) -> np.ndarray:
    """
    Keep semantic prefix (c0, c1); reassign c2 inside each prefix bucket so SIDs are unique.
    Safe when max bucket size <= codebook_size (true for Amazon23 Industrial with K=256).
    """
    from collections import defaultdict

    out = indices.copy()
    groups: dict[tuple[int, int], list[int]] = defaultdict(list)
    for i, (a, b, c) in enumerate(indices):
        groups[(int(a), int(b))].append(i)
    for (a, b), members in groups.items():
        if len(members) <= 1:
            continue
        members = sorted(members, key=lambda i: (int(indices[i, 2]), i))
        if len(members) > codebook_size:
            # Extremely rare: spill into neighboring b codes.
            for j, i in enumerate(members):
                out[i, 2] = j % codebook_size
                out[i, 1] = (b + j // codebook_size) % codebook_size
        else:
            for j, i in enumerate(members):
                out[i, 2] = j
    return out


def train_residual_kmeans(
    emb_path: Path,
    ids_path: Path,
    out_dir: Path,
    pca_dim: int = 256,
    codebook_size: int = CODEBOOK_SIZE,
    num_layers: int = NUM_CODEBOOK_LAYERS,
    seed: int = 42,
    enforce_unique: bool = True,
) -> None:
    """
    Classic residual quantization via layered KMeans (no neural encoder).
    Optional last-code uniquify keeps (c0,c1) semantics while driving collision→0.
    """
    emb = np.load(emb_path)
    with open(ids_path, encoding="utf-8") as f:
        item_ids = json.load(f)
    assert len(item_ids) == emb.shape[0]

    emb_pca, pca_meta = _apply_pca(emb, pca_dim)
    print(
        f"[rq-kmeans] PCA {pca_meta['original_dim']}→{pca_meta['pca_dim']} "
        f"var_explained={pca_meta['explained_variance_ratio_sum']:.4f}"
    )

    residual = emb_pca.copy()
    index_cols: list[np.ndarray] = []
    codebooks: list[np.ndarray] = []
    for layer in range(num_layers):
        k = min(codebook_size, residual.shape[0])
        km = KMeans(n_clusters=k, n_init=20, random_state=seed + layer, max_iter=300)
        km.fit(residual)
        centers = km.cluster_centers_
        if k < codebook_size:
            rng = np.random.default_rng(seed + layer)
            pad = centers[rng.integers(0, k, size=codebook_size - k)]
            pad = pad + 0.01 * rng.normal(size=pad.shape)
            centers = np.concatenate([centers, pad], axis=0)
            dists = ((residual[:, None, :] - centers[None, :, :]) ** 2).sum(-1)
            idx = dists.argmin(axis=1)
        else:
            idx = km.predict(residual)
        residual = residual - centers[idx]
        index_cols.append(idx.astype(np.int64))
        codebooks.append(centers.astype(np.float32))
        used = len(np.unique(idx))
        print(f"[rq-kmeans] layer={layer} used={used}/{codebook_size}")

    indices = np.stack(index_cols, axis=1)
    keys = [tuple(row.tolist()) for row in indices]
    unique = len(set(keys))
    collision = 1.0 - unique / len(keys)
    usage = [int(len(np.unique(indices[:, i]))) for i in range(num_layers)]
    print(f"[rq-kmeans] raw collision={collision:.4f} unique={unique}/{len(keys)} usage={usage}")

    if enforce_unique and num_layers >= 3:
        indices = _enforce_unique_last_code(indices, codebook_size=codebook_size)
        keys_u = [tuple(row.tolist()) for row in indices]
        unique = len(set(keys_u))
        collision = 1.0 - unique / len(keys_u)
        usage = [int(len(np.unique(indices[:, i]))) for i in range(num_layers)]
        print(f"[rq-kmeans] enforced collision={collision:.4f} unique={unique}/{len(keys_u)} usage={usage}")

    metrics = {
        "collision": collision,
        "unique": unique,
        "n": len(keys),
        "usage": usage,
        "method": "residual_kmeans",
        "enforce_unique": enforce_unique,
        "codebook_size": codebook_size,
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    sid_map = _export_sid_map(item_ids, indices)
    with open(out_dir / "sid_map.json", "w", encoding="utf-8") as f:
        json.dump(sid_map, f, ensure_ascii=False, indent=2)
    with open(out_dir / "pca_meta.json", "w", encoding="utf-8") as f:
        json.dump(pca_meta, f, indent=2)
    with open(out_dir / "best_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    np.savez(out_dir / "codebooks.npz", **{f"layer{i}": cb for i, cb in enumerate(codebooks)})
    print(f"Done. artifacts in {out_dir}")


def train_rqvae(
    emb_path: Path,
    ids_path: Path,
    out_dir: Path,
    epochs: int = 4000,
    batch_size: int = 2048,
    lr: float = 3e-4,
    latent_dim: int = 64,
    hidden_dim: int = 256,
    beta: float = 0.25,
    device: str = "cuda:0",
    log_every: int = 50,
    early_collision_patience: int = 40,
    pca_dim: int = 256,
    dead_code_every: int = 100,
    warm_start_size: int = 8192,
) -> None:
    emb = np.load(emb_path)
    with open(ids_path, encoding="utf-8") as f:
        item_ids = json.load(f)
    assert len(item_ids) == emb.shape[0]

    emb_pca, pca_meta = _apply_pca(emb, pca_dim)
    print(
        f"[rqvae] PCA {pca_meta['original_dim']}→{pca_meta['pca_dim']} "
        f"var_explained={pca_meta['explained_variance_ratio_sum']:.4f}"
    )

    x = torch.tensor(emb_pca, dtype=torch.float32)
    ds = TensorDataset(x)
    bs = min(batch_size, len(ds))
    loader = DataLoader(ds, batch_size=bs, shuffle=True, drop_last=False)

    model = RQVAE(in_dim=emb_pca.shape[1], latent_dim=latent_dim, hidden_dim=hidden_dim, beta=beta).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs))

    # warm-start on a large random subset (or full data if small)
    n_warm = min(warm_start_size, len(ds))
    warm_idx = torch.randperm(len(ds))[:n_warm]
    warm_batch = x[warm_idx].to(device)
    model.warm_start(warm_batch)
    print(f"[rqvae] warm-start on {n_warm} items, latent_dim={latent_dim}, beta={beta}")

    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "pca_meta.json", "w", encoding="utf-8") as f:
        json.dump(pca_meta, f, indent=2)

    best_collision = 1.0
    stale = 0
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.0
        for (batch,) in loader:
            batch = batch.to(device)
            loss, _, _, _ = model(batch)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()
        sched.step()

        # periodic dead-code reset
        if dead_code_every > 0 and epoch % dead_code_every == 0:
            model.eval()
            with torch.no_grad():
                all_idx = []
                for (batch,) in DataLoader(ds, batch_size=bs, shuffle=False):
                    all_idx.append(model.encode_indices(batch.to(device)).cpu())
                indices_tmp = torch.cat(all_idx, dim=0).numpy()
            usage = []
            for layer in range(NUM_CODEBOOK_LAYERS):
                counts = np.bincount(indices_tmp[:, layer], minlength=CODEBOOK_SIZE)
                usage.append(counts)
            # reset using a shuffled subset for diversity
            reset_batch = x[torch.randperm(len(ds))[: min(4096, len(ds))]].to(device)
            resets = model.reset_dead_codes(reset_batch, usage, min_count=1)
            if any(resets):
                print(f"[rqvae] epoch={epoch} dead-code reset per layer={resets}")

        if epoch % log_every == 0 or epoch == 1:
            model.eval()
            with torch.no_grad():
                all_idx = []
                for (batch,) in DataLoader(ds, batch_size=bs, shuffle=False):
                    all_idx.append(model.encode_indices(batch.to(device)).cpu())
                indices = torch.cat(all_idx, dim=0).numpy()
            keys = [tuple(row.tolist()) for row in indices]
            unique = len(set(keys))
            collision = 1.0 - unique / len(keys)
            avg_loss = total / max(1, len(loader))
            usage_counts = [
                int(np.bincount(indices[:, layer], minlength=CODEBOOK_SIZE).astype(bool).sum())
                for layer in range(NUM_CODEBOOK_LAYERS)
            ]
            row = {
                "epoch": epoch,
                "loss": avg_loss,
                "collision": collision,
                "unique": unique,
                "usage": usage_counts,
                "lr": float(sched.get_last_lr()[0]),
            }
            history.append(row)
            print(
                f"epoch={epoch} loss={avg_loss:.6f} collision={collision:.4f} "
                f"unique={unique}/{len(keys)} usage={usage_counts} lr={row['lr']:.2e}"
            )
            if collision < best_collision - 1e-6:
                best_collision = collision
                stale = 0
                torch.save(
                    {
                        "model": model.state_dict(),
                        "in_dim": emb_pca.shape[1],
                        "latent_dim": latent_dim,
                        "pca_meta": pca_meta,
                    },
                    out_dir / "rqvae.pt",
                )
                sid_map = _export_sid_map(item_ids, indices)
                with open(out_dir / "sid_map.json", "w", encoding="utf-8") as f:
                    json.dump(sid_map, f, ensure_ascii=False, indent=2)
                with open(out_dir / "best_metrics.json", "w", encoding="utf-8") as f:
                    json.dump(row, f, indent=2)
            else:
                stale += 1
                if stale >= early_collision_patience and best_collision < 0.5:
                    print(f"Early stop at epoch {epoch}, best_collision={best_collision:.4f}")
                    break

    with open(out_dir / "train_history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    print(f"Done. best_collision={best_collision:.4f} artifacts in {out_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--emb_path", type=Path, required=True)
    parser.add_argument("--ids_path", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--method", type=str, default="residual_kmeans", choices=["residual_kmeans", "rqvae"])
    parser.add_argument("--epochs", type=int, default=4000)
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--latent_dim", type=int, default=64)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--beta", type=float, default=0.25)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument("--early_collision_patience", type=int, default=40)
    parser.add_argument("--pca_dim", type=int, default=256)
    parser.add_argument("--dead_code_every", type=int, default=100)
    parser.add_argument("--warm_start_size", type=int, default=8192)
    parser.add_argument(
        "--enforce_unique",
        type=int,
        default=1,
        help="For residual_kmeans: reassign last code within (c0,c1) buckets (1=on).",
    )
    args = parser.parse_args()
    if args.method == "residual_kmeans":
        train_residual_kmeans(
            args.emb_path,
            args.ids_path,
            args.out_dir,
            pca_dim=args.pca_dim,
            enforce_unique=bool(args.enforce_unique),
        )
        return
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
        early_collision_patience=args.early_collision_patience,
        pca_dim=args.pca_dim,
        dead_code_every=args.dead_code_every,
        warm_start_size=args.warm_start_size,
    )


if __name__ == "__main__":
    main()
