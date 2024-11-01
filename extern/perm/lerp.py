from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import struct
from hair_roots import HairRoots
from misc import copy2cpu as c2c


def save_hair(path: str, data: np.ndarray) -> None:
    num_strands, num_points = data.shape[:2]
    with open(path, 'wb') as f:
        f.write(struct.pack('i', num_strands))
        for i in range(num_strands):
            f.write(struct.pack('i', num_points))
            f.write(struct.pack('f' * num_points * 3, *data[i].flatten().tolist()))


def project(data: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    """ Project data to the subspace spanned by bases.

    Args:
        data (torch.Tensor): Hair data of shape (batch_size, ...).
        basis (torch.Tensor): Blend shapes of shape (num_blend_shapes, ...).

    Returns:
        (torch.Tensor): Projected parameters of shape (batch_size, num_blend_shapes).
    """
    return torch.einsum('bn,cn->bc', data.flatten(start_dim=1), basis.flatten(start_dim=1))


def blend(coeff: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    """ Blend parameters and the corresponding blend shapes.

    Args:
        coeff (torch.Tensor): Parameters (blend shape coefficients) of shape (batch_size, num_blend_shapes).
        basis (torch.Tensor): Blend shapes of shape (num_blend_shapes, ...).

    Returns:
        (torch.Tensor): Blended results of shape (batch_size, ...).
    """
    return torch.einsum('bn,n...->b...', coeff, basis)


def sample(coords: torch.Tensor, blend_shape: torch.Tensor, mode: str = 'bilinear') -> torch.Tensor:
    """ Sample neural blend shapes with given coordinates.

    Args:
        coords (torch.Tensor): Sample coordinates of shape (batch_size, num_coords, 2) in [0, 1] x [0, 1].
        blend_shape (torch.Tensor): Blend shapes of shape (batch_size, feature_dim, height, width).
        mode (str): Interpolation mode for sampling.

    Returns:
        (torch.Tensor): Sampled neural features of shape (batch_size, num_coords, feature_dim).
    """
    grid = coords * 2 - 1  # (batch_size, num_coords, 2), in [-1, 1]
    samples = F.grid_sample(blend_shape, grid.unsqueeze(2), mode=mode, align_corners=True)  # (batch_size, feature_dim, num_coords, 1)
    samples = samples.squeeze(-1).mT  # (batch_size, num_coords, feature_dim)

    return samples


class StrandCodec(nn.Module):
    def __init__(self, model_path: str, num_coeff: Optional[int] = None, fft: bool = True):
        super().__init__()
        self.SAMPLES_PER_STRAND = 100

        data = np.load(model_path)
        self.register_buffer('mean_shape', torch.tensor(data['mean_shape'], dtype=torch.float32))

        if num_coeff is not None:
            num_coeff = min(num_coeff, data['blend_shapes'].shape[0])
        else:
            num_coeff = data['blend_shapes'].shape[0]
        self.register_buffer('blend_shapes', torch.tensor(data['blend_shapes'][:num_coeff], dtype=torch.float32))

        self.num_coeff = num_coeff
        self.fft = fft

    def encode(self, x):
        if self.fft:
            fourier = torch.fft.rfft(x, n=self.SAMPLES_PER_STRAND - 1, dim=-2, norm='ortho')
            x = torch.cat((fourier.real, fourier.imag), dim=-1)
        return project(x - self.mean_shape, self.blend_shapes)

    def decode(self, coeff):
        x = self.mean_shape + blend(coeff, self.blend_shapes)
        if self.fft:
            x = torch.fft.irfft(torch.complex(x[..., :3], x[..., 3:]), n=self.SAMPLES_PER_STRAND - 1, dim=-2, norm='ortho')
        return x

    def forward(self, x):
        coeff = self.encode(x)
        return self.decode(coeff)


class PermPCA(nn.Module):
    """A wrapper class for PCA version of Perm. """

    def __init__(
        self,
        model_path: str,
        head_mesh: str,
        scalp_bounds: Optional[Tuple[float]] = None,
    ):
        super().__init__()

        self.hair_roots = HairRoots(head_mesh, scalp_bounds=scalp_bounds)
        self.strand_codec = StrandCodec(model_path, num_coeff=64, fft=True)

    def forward(
        self,
        roots: torch.Tensor,
        texture: torch.Tensor
    ):
        coords = self.hair_roots.rescale(roots[..., :2])
        batch_size, num_coords = coords.shape[:2]
        coeff = sample(coords, texture, mode='nearest')
        position = self.strand_codec.decode(coeff.reshape(batch_size * num_coords, -1))
        position = position.reshape(batch_size, num_coords, -1, 3)
        position = F.pad(position, (0, 0, 1, 0), mode='constant', value=0)  # [batch_size, num_strands, 100, 3]
        position = position + self.hair_roots.spherical_to_cartesian(roots).unsqueeze(2)
        return position


device = torch.device('cuda:0')

perm = PermPCA(model_path='data/blend-shapes/fft-strands-blend-shapes.npz', head_mesh='data/head.obj', scalp_bounds=[0.1870, 0.8018, 0.4011, 0.8047])
perm = perm.eval().requires_grad_(False).to(device)

ALL_TEXTURE = []
ALL_ROOTS = []
for fname in ['Bob', 'Blowout', 'Jewfro', 'Jheri', 'Curly']:
    data = np.load(f'../../data/ct2hair/{fname}.npz')
    ALL_TEXTURE.append(torch.tensor(data['texture'][None, ...], dtype=torch.float32, device=device))
    ALL_ROOTS.append(torch.tensor(data['roots'][None, ...], dtype=torch.float32, device=device))


def perm_lerp(value: float) -> np.ndarray:
    value = np.clip(value, 0, len(ALL_TEXTURE) - 1)
    category = int(value)
    if category == len(ALL_TEXTURE) - 1:
        texture = ALL_TEXTURE[-1]
        roots = ALL_ROOTS[-1]
    else:
        weight = value - category
        texture1 = ALL_TEXTURE[category]
        texture2 = ALL_TEXTURE[category + 1]
        texture = torch.lerp(texture1, texture2, weight=weight)
        roots = ALL_ROOTS[category]
    strands = perm(texture=texture, roots=roots)
    return strands


value = 2.2
strands = perm_lerp(value)
save_hair(f'perm-interp-{value}.data', c2c(strands[0]))
