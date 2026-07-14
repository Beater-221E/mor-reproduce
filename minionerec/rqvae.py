"""Residual Quantized VAE for Semantic IDs (paper §3.2).

Critical for stability on unit-norm embeddings:
- LayerNorm on encoder output (otherwise cluster spacing << Adam step)
- Classic VQ loss with Adam on codebook (not broken EMA-from-zero)
- Dead-code reset each step
- Default lr=3e-4
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader, TensorDataset

from minionerec.util import CODEBOOK_SIZE, NUM_CODEBOOK_LAYERS, format_sid


class RQVAE(nn.Module):
    def __init__(
        self,
        in_dim: int,
        latent_dim: int = 128,
        hidden_dim: int = 512,
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
            nn.LayerNorm(latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, in_dim),
        )
        self.codebooks = nn.ParameterList(
            [nn.Parameter(torch.randn(codebook_size, latent_dim) * 0.02) for _ in range(num_layers)]
        )

    @torch.no_grad()
    def warm_start(self, data: torch.Tensor, max_samples: int = 20000) -> None:
        """K-means init on LayerNorm-ed encoder outputs (large sample)."""
        self.eval()
        n = data.size(0)
        sample = data[torch.randperm(n)[:max_samples]] if n > max_samples else data
        device = next(self.parameters()).device
        zs = []
        for i in range(0, sample.size(0), 2048):
            zs.append(self.encoder(sample[i : i + 2048].to(device)).cpu())
        residual = torch.cat(zs, dim=0).numpy()
        for layer in range(self.num_layers):
            k = min(self.codebook_size, residual.shape[0])
            km = MiniBatchKMeans(
                n_clusters=k,
                batch_size=min(2048, residual.shape[0]),
                n_init=5,
                max_iter=100,
                random_state=42 + layer,
            )
            km.fit(residual)
            centers = torch.tensor(km.cluster_centers_, dtype=torch.float32)
            if k < self.codebook_size:
                pad = centers[torch.randint(0, k, (self.codebook_size - k,))]
                pad = pad + 0.01 * torch.randn_like(pad)
                centers = torch.cat([centers, pad], dim=0)
            self.codebooks[layer].data.copy_(centers.to(device))
            assign = km.predict(residual)
            residual = residual - centers.numpy()[assign]
            print(f"warm_start layer={layer} unique={len(set(assign.tolist()))}/{k}")
        self.train()

    @torch.no_grad()
    def reset_dead_codes(self, z: torch.Tensor, indices: torch.Tensor) -> None:
        """Replace unused codebook entries with residuals from the batch."""
        residual = z
        for layer in range(self.num_layers):
            idx = indices[:, layer]
            counts = torch.bincount(idx, minlength=self.codebook_size)
            dead = (counts == 0).nonzero(as_tuple=False).view(-1)
            if dead.numel() > 0 and residual.size(0) > 0:
                rep = residual[torch.randint(0, residual.size(0), (dead.numel(),), device=residual.device)]
                self.codebooks[layer].data[dead] = rep
            residual = residual - self.codebooks[layer][idx]

    def quantize(self, z: torch.Tensor):
        residual = z
        indices = []
        quantized_sum = torch.zeros_like(z)
        vq_loss = z.new_zeros(())
        for layer in range(self.num_layers):
            cb = self.codebooks[layer]
            idx = torch.cdist(residual, cb).argmin(dim=-1)
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
        return recon_loss + vq_loss, recon_loss.detach(), vq_loss.detach(), indices, z.detach()

    @torch.no_grad()
    def encode_indices(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        _, indices, _ = self.quantize(z)
        return indices


def _export_sid_map(item_ids: list[str], indices: np.ndarray):
    sid_map = {}
    used: dict[tuple[int, int, int], str] = {}
    n_collide = 0
    for iid, codes in zip(item_ids, indices):
        code_t = tuple(int(c) for c in codes.tolist())
        raw = code_t
        if code_t in used:
            n_collide += 1
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
            "raw_codes": list(raw),
        }
    return sid_map, n_collide


def fit_rq_kmeans(
    emb: np.ndarray,
    num_layers: int = NUM_CODEBOOK_LAYERS,
    codebook_size: int = CODEBOOK_SIZE,
    pca_dim: int = 256,
    random_state: int = 42,
):
    x = emb.astype(np.float64)
    norms = np.linalg.norm(x, axis=1, keepdims=True).clip(min=1e-8)
    x = x / norms

    pca = None
    if pca_dim and pca_dim < x.shape[1]:
        pca = PCA(n_components=pca_dim, random_state=random_state)
        x = pca.fit_transform(x)
        print(f"PCA {emb.shape[1]} -> {pca_dim}, var_explained={pca.explained_variance_ratio_.sum():.4f}")

    residual = x.copy()
    codebooks: list[np.ndarray] = []
    indices_layers: list[np.ndarray] = []
    for layer in range(num_layers):
        k = min(codebook_size, residual.shape[0])
        km = MiniBatchKMeans(
            n_clusters=k,
            batch_size=min(4096, residual.shape[0]),
            n_init=10,
            max_iter=200,
            random_state=random_state + layer,
            reassignment_ratio=0.01,
        )
        km.fit(residual)
        centers = km.cluster_centers_
        if k < codebook_size:
            rng = np.random.default_rng(random_state + layer)
            pad = centers[rng.integers(0, k, size=codebook_size - k)]
            pad = pad + 0.01 * rng.standard_normal(pad.shape)
            centers = np.concatenate([centers, pad], axis=0)
            dists = ((residual[:, None, :] - centers[None, :, :]) ** 2).sum(-1)
            idx = dists.argmin(axis=1)
        else:
            idx = km.predict(residual)
        codebooks.append(centers.astype(np.float32))
        indices_layers.append(idx.astype(np.int64))
        residual = residual - centers[idx]
        print(f"rq-kmeans layer={layer} unique={len(set(idx.tolist()))}/{codebook_size}")

    return np.stack(indices_layers, axis=1), codebooks, pca


def train_rq_kmeans(emb_path: Path, ids_path: Path, out_dir: Path, pca_dim: int = 256) -> None:
    emb = np.load(emb_path)
    with open(ids_path, encoding="utf-8") as f:
        item_ids = json.load(f)
    assert len(item_ids) == emb.shape[0]

    print(f"RQ-KMeans n_items={len(item_ids)} in_dim={emb.shape[1]}")
    indices, codebooks, pca = fit_rq_kmeans(emb, pca_dim=pca_dim)
    keys = [tuple(row.tolist()) for row in indices]
    unique = len(set(keys))
    collision = 1.0 - unique / len(keys)
    usage = [len(set(indices[:, i].tolist())) for i in range(indices.shape[1])]
    print(f"collision={collision:.4f} unique={unique}/{len(keys)} usage={usage}")

    out_dir.mkdir(parents=True, exist_ok=True)
    sid_map, n_disambig = _export_sid_map(item_ids, indices)
    with open(out_dir / "sid_map.json", "w", encoding="utf-8") as f:
        json.dump(sid_map, f, ensure_ascii=False, indent=2)
    np.savez(out_dir / "codebooks.npz", **{f"layer{i}": cb for i, cb in enumerate(codebooks)})
    if pca is not None:
        np.savez(out_dir / "pca.npz", components=pca.components_, mean=pca.mean_)
    with open(out_dir / "train_history.json", "w", encoding="utf-8") as f:
        json.dump(
            [{"method": "rq_kmeans", "collision": collision, "unique": unique, "usage": usage, "disambiguated": n_disambig}],
            f,
            indent=2,
        )
    print(f"Done. disambiguated={n_disambig} Artifacts in {out_dir}")


def train_rqvae(
    emb_path: Path,
    ids_path: Path,
    out_dir: Path,
    epochs: int = 10000,
    batch_size: int = 2048,
    lr: float = 3e-4,
    latent_dim: int = 128,
    hidden_dim: int = 512,
    beta: float = 0.25,
    device: str = "cuda:0",
    log_every: int = 50,
    early_collision_patience: int = 40,
    target_collision: float = 0.05,
) -> None:
    emb = np.load(emb_path)
    with open(ids_path, encoding="utf-8") as f:
        item_ids = json.load(f)
    assert len(item_ids) == emb.shape[0]

    norms = np.linalg.norm(emb, axis=1, keepdims=True).clip(min=1e-8)
    emb = emb / norms
    x = torch.tensor(emb, dtype=torch.float32)
    ds = TensorDataset(x)
    bs = min(batch_size, len(ds))
    loader = DataLoader(ds, batch_size=bs, shuffle=True, drop_last=False)

    model = RQVAE(in_dim=emb.shape[1], latent_dim=latent_dim, hidden_dim=hidden_dim, beta=beta).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    print(f"RQ-VAE in_dim={emb.shape[1]} n_items={len(ds)} latent={latent_dim} lr={lr}")
    model.warm_start(x)

    out_dir.mkdir(parents=True, exist_ok=True)
    best_collision = 1.0
    stale = 0
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.0
        for (batch,) in loader:
            batch = batch.to(device)
            loss, _, _, indices, z = model(batch)
            model.reset_dead_codes(z, indices)
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
                indices_np = torch.cat(all_idx, dim=0).numpy()
            keys = [tuple(row.tolist()) for row in indices_np]
            unique = len(set(keys))
            collision = 1.0 - unique / len(keys)
            usage = [len(set(indices_np[:, i].tolist())) for i in range(indices_np.shape[1])]
            avg_loss = total / max(1, len(loader))
            history.append({"epoch": epoch, "loss": avg_loss, "collision": collision, "unique": unique, "usage": usage})
            print(
                f"epoch={epoch} loss={avg_loss:.6f} collision={collision:.4f} "
                f"unique={unique}/{len(keys)} usage={usage}"
            )
            if collision < best_collision - 1e-6:
                best_collision = collision
                stale = 0
                torch.save(
                    {
                        "model": model.state_dict(),
                        "in_dim": emb.shape[1],
                        "latent_dim": latent_dim,
                        "hidden_dim": hidden_dim,
                    },
                    out_dir / "rqvae.pt",
                )
                sid_map, n_disambig = _export_sid_map(item_ids, indices_np)
                with open(out_dir / "sid_map.json", "w", encoding="utf-8") as f:
                    json.dump(sid_map, f, ensure_ascii=False, indent=2)
                print(f"  saved checkpoint (disambiguated={n_disambig})")
            else:
                stale += 1
                if best_collision <= target_collision and stale >= early_collision_patience:
                    print(f"Early stop at epoch {epoch}, best_collision={best_collision:.4f}")
                    break
                if epoch >= 300 and best_collision > 0.5:
                    print(f"Abort: still high collision={best_collision:.4f}; try METHOD=rq_kmeans")
                    break

    with open(out_dir / "train_history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    print(f"Done. best_collision={best_collision:.4f} Artifacts in {out_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--emb_path", type=Path, required=True)
    parser.add_argument("--ids_path", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--method", type=str, default="rqvae", choices=["rq_kmeans", "rqvae"])
    parser.add_argument("--pca_dim", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=10000)
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--latent_dim", type=int, default=128)
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--beta", type=float, default=0.25)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--log_every", type=int, default=50)
    args = parser.parse_args()

    if args.method == "rq_kmeans":
        train_rq_kmeans(args.emb_path, args.ids_path, args.out_dir, pca_dim=args.pca_dim)
    else:
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
