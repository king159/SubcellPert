"""CUDA implementations of the classical baselines' complete-RNA geometry.

Keep the NumPy path as the reference. No RNA subsampling, changed quantiles,
or reduced precision: statistics and interpolation use float64 on CUDA.
"""

from __future__ import annotations

import numpy as np
import torch


def _group_statistics(xy, labels, counts, *, include_radial):
    groups = len(counts)
    mean = xy.new_zeros((groups, 2))
    second = xy.new_zeros((groups, 4))
    mean.index_add_(0, labels, xy)
    second.index_add_(0, labels, (xy[:, :, None] * xy[:, None, :]).reshape(-1, 4))
    denominator = counts.clamp_min(1)
    mean /= denominator[:, None]
    second = (second / denominator[:, None]).reshape(groups, 2, 2)
    radial = xy.new_zeros((groups, 16))
    if include_radial and len(xy):
        radius = torch.linalg.vector_norm(xy - mean[labels], dim=1)
        # Stable secondary sort groups globally sorted radii without padding
        # each gene to the largest molecule count.
        order = torch.argsort(radius, stable=True)
        order = order[torch.argsort(labels[order], stable=True)]
        ordered = radius[order]
        offsets = counts.cumsum(0) - counts
        q = torch.linspace(0, 1, 16, dtype=xy.dtype, device=xy.device)
        positions = (counts - 1).clamp_min(0)[:, None] * q
        low = positions.floor().long()
        high = positions.ceil().long()
        left = ordered[(offsets[:, None] + low).clamp_max(len(xy) - 1)]
        right = ordered[(offsets[:, None] + high).clamp_max(len(xy) - 1)]
        radial = left + (positions - low) * (right - left)
        radial[counts == 0] = 0
    return mean, second, radial


@torch.inference_mode()
def complete_cell_statistics_cuda(batch, *, device, source_fallback=False, include_radial=True):
    batch.validated()
    device = torch.device(device)
    if device.type != "cuda":
        raise ValueError("The CUDA statistics backend requires a CUDA device")
    b, g = batch.counts.shape
    counts = torch.as_tensor(batch.counts, dtype=torch.long, device=device)
    cells = torch.as_tensor(batch.molecule_cells, dtype=torch.long, device=device)
    genes = torch.as_tensor(batch.molecule_genes, dtype=torch.long, device=device)
    xy = torch.as_tensor(batch.coordinates, dtype=torch.float64, device=device)
    mean, second, radial = _group_statistics(xy, cells * g + genes, counts.flatten(), include_radial=include_radial)
    mean, second, radial = mean.reshape(b, g, 2), second.reshape(b, g, 2, 2), radial.reshape(b, g, 16)
    valid = counts > 0
    if source_fallback:
        fallback = _group_statistics(xy, cells, counts.sum(1), include_radial=include_radial)
        mean = torch.where(valid[..., None], mean, fallback[0][:, None])
        second = torch.where(valid[..., None, None], second, fallback[1][:, None])
        radial = torch.where(valid[..., None], radial, fallback[2][:, None])
        valid = torch.ones_like(valid)
    return tuple(value.cpu().numpy() for value in (mean, second, radial, valid))


@torch.inference_mode()
def apply_transform_cuda(transform, xy, genes, *, device):
    """Bound GPU workspace while transforming every generated molecule."""
    kind = transform["kind"]
    if kind not in {"affine", "radial"}:
        raise ValueError("Unknown Oracle spatial transform")
    values = {
        key: torch.as_tensor(value, dtype=torch.float64, device=device)
        for key, value in transform.items()
        if key != "kind"
    }
    result = np.empty((len(xy), 2), dtype=np.float64)
    for first in range(0, len(xy), 65536):
        sl = slice(first, first + 65536)
        points = torch.as_tensor(xy[sl], dtype=torch.float64, device=device)
        labels = torch.as_tensor(genes[sl], dtype=torch.long, device=device)
        if kind == "affine":
            params = values["parameters"][labels]
            transformed = torch.bmm(params[:, :4].reshape(-1, 2, 2), points[..., None]).squeeze(-1) + params[:, 4:]
        else:
            centered = points - values["centers"][labels]
            radius = torch.linalg.vector_norm(centered, dim=1)
            source_q, target_q = values["source_quantiles"][labels], values["target_quantiles"][labels]
            low = ((radius[:, None] >= source_q).sum(1) - 1).clamp(0, source_q.shape[1] - 2)
            rows = torch.arange(len(labels), device=device)
            fraction = (
                (radius - source_q[rows, low]) / (source_q[rows, low + 1] - source_q[rows, low]).clamp_min(1e-6)
            ).clamp(0, 1)
            mapped = target_q[rows, low] + fraction * (target_q[rows, low + 1] - target_q[rows, low])
            transformed = (
                values["centers"][labels] + values["shift"][labels]
                + centered * (mapped / radius.clamp_min(1e-6))[:, None]
            )
        result[sl] = transformed.cpu().numpy()
    return result
