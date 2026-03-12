from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F


def pil_to_tensor(image: Image.Image) -> torch.Tensor:
    arr = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1).contiguous()
    return tensor


def mask_pil_to_tensor(mask: Image.Image) -> torch.Tensor:
    arr = np.asarray(mask.convert("L"), dtype=np.float32) / 255.0
    tensor = torch.from_numpy(arr)[None, ...].contiguous()
    return tensor


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    tensor = tensor.detach().clamp(0.0, 1.0)
    if tensor.dim() == 4:
        if tensor.size(0) != 1:
            raise ValueError("tensor_to_pil expects a single image or batch size 1")
        tensor = tensor[0]
    arr = (tensor.permute(1, 2, 0).cpu().numpy() * 255.0).round().astype(np.uint8)
    return Image.fromarray(arr, mode="RGB")


def ensure_batch(image: torch.Tensor) -> torch.Tensor:
    if image.dim() == 3:
        return image.unsqueeze(0)
    if image.dim() != 4:
        raise ValueError(f"Expected CHW or BCHW tensor, got shape {tuple(image.shape)}")
    return image


def ensure_mask_batch(mask: torch.Tensor, height: int, width: int, device: Optional[torch.device] = None) -> torch.Tensor:
    if mask.dim() == 2:
        mask = mask.unsqueeze(0).unsqueeze(0)
    elif mask.dim() == 3:
        if mask.size(0) == 1:
            mask = mask.unsqueeze(0)
        else:
            mask = mask.unsqueeze(1)
    elif mask.dim() != 4:
        raise ValueError(f"Expected HW, CHW or BCHW mask, got shape {tuple(mask.shape)}")
    if mask.shape[-2:] != (height, width):
        mask = F.interpolate(mask.float(), size=(height, width), mode="bilinear", align_corners=False)
    if device is not None:
        mask = mask.to(device)
    return mask.clamp(0.0, 1.0)


def gaussian_kernel1d(kernel_size: int, sigma: float, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    radius = kernel_size // 2
    coords = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    kernel = torch.exp(-(coords**2) / max(2 * sigma * sigma, 1e-6))
    kernel = kernel / kernel.sum()
    return kernel


def gaussian_blur(image: torch.Tensor, sigma: float = 3.0, kernel_size: Optional[int] = None) -> torch.Tensor:
    image = ensure_batch(image)
    if sigma <= 0:
        return image
    if kernel_size is None:
        kernel_size = int(max(3, round(sigma * 6))) | 1
    device = image.device
    dtype = image.dtype
    kernel_1d = gaussian_kernel1d(kernel_size, sigma, device, dtype)
    kernel_x = kernel_1d.view(1, 1, 1, -1)
    kernel_y = kernel_1d.view(1, 1, -1, 1)
    channels = image.shape[1]
    kernel_x = kernel_x.expand(channels, 1, 1, kernel_size)
    kernel_y = kernel_y.expand(channels, 1, kernel_size, 1)
    padded = F.pad(image, (kernel_size // 2, kernel_size // 2, kernel_size // 2, kernel_size // 2), mode="reflect")
    blurred = F.conv2d(padded, kernel_x, groups=channels)
    blurred = F.conv2d(blurred, kernel_y, groups=channels)
    return blurred


def lowpass(image: torch.Tensor, sigma: float = 5.0) -> torch.Tensor:
    return gaussian_blur(image, sigma=sigma)


def highpass(image: torch.Tensor, sigma: float = 5.0) -> torch.Tensor:
    image_b = ensure_batch(image)
    return image_b - lowpass(image_b, sigma=sigma)


def rgb_to_oklab(image: torch.Tensor) -> torch.Tensor:
    image = ensure_batch(image).clamp(0.0, 1.0)
    # linearize sRGB
    linear = torch.where(
        image <= 0.04045,
        image / 12.92,
        ((image + 0.055) / 1.055) ** 2.4,
    )
    m1 = torch.tensor(
        [
            [0.4122214708, 0.5363325363, 0.0514459929],
            [0.2119034982, 0.6806995451, 0.1073969566],
            [0.0883024619, 0.2817188376, 0.6299787005],
        ],
        device=image.device,
        dtype=image.dtype,
    )
    lms = torch.einsum("ij,bjhw->bihw", m1, linear)
    lms = torch.clamp(lms, min=1e-8) ** (1.0 / 3.0)
    m2 = torch.tensor(
        [
            [0.2104542553, 0.7936177850, -0.0040720468],
            [1.9779984951, -2.4285922050, 0.4505937099],
            [0.0259040371, 0.7827717662, -0.8086757660],
        ],
        device=image.device,
        dtype=image.dtype,
    )
    return torch.einsum("ij,bjhw->bihw", m2, lms)


def oklab_to_rgb(image: torch.Tensor) -> torch.Tensor:
    image = ensure_batch(image)
    m1_inv = torch.tensor(
        [
            [1.0, 0.3963377774, 0.2158037573],
            [1.0, -0.1055613458, -0.0638541728],
            [1.0, -0.0894841775, -1.2914855480],
        ],
        device=image.device,
        dtype=image.dtype,
    )
    lms_ = torch.einsum("ij,bjhw->bihw", m1_inv, image)
    lms = lms_ ** 3
    m2_inv = torch.tensor(
        [
            [4.0767416621, -3.3077115913, 0.2309699292],
            [-1.2684380046, 2.6097574011, -0.3413193965],
            [-0.0041960863, -0.7034186147, 1.7076147010],
        ],
        device=image.device,
        dtype=image.dtype,
    )
    rgb_linear = torch.einsum("ij,bjhw->bihw", m2_inv, lms)
    rgb = torch.where(
        rgb_linear <= 0.0031308,
        12.92 * rgb_linear,
        1.055 * torch.clamp(rgb_linear, min=0.0) ** (1.0 / 2.4) - 0.055,
    )
    return rgb.clamp(0.0, 1.0)


def make_full_mask(batch: int, height: int, width: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    return torch.ones((batch, 1, height, width), device=device, dtype=dtype)


def make_coord_grid(batch: int, height: int, width: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    ys = torch.linspace(-1.0, 1.0, height, device=device, dtype=dtype)
    xs = torch.linspace(-1.0, 1.0, width, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    grid = torch.stack([xx, yy], dim=0).unsqueeze(0).repeat(batch, 1, 1, 1)
    return grid


def image_center(mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    mask = ensure_batch(mask)
    b, _, h, w = mask.shape
    coords = make_coord_grid(b, h, w, mask.device, mask.dtype)
    weights = mask[:, 0] + 1e-6
    x = (coords[:, 0] * weights).sum(dim=(1, 2)) / weights.sum(dim=(1, 2))
    y = (coords[:, 1] * weights).sum(dim=(1, 2)) / weights.sum(dim=(1, 2))
    return x, y


def distance_like_field(mask: torch.Tensor) -> torch.Tensor:
    mask = ensure_batch(mask).clamp(0.0, 1.0)
    soft = gaussian_blur(mask, sigma=7.0)
    soft = soft / torch.clamp(soft.amax(dim=(-2, -1), keepdim=True), min=1e-6)
    return (soft * 2.0 - 1.0).clamp(-1.0, 1.0)


def build_feature_mask(mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    soft = gaussian_blur(mask, sigma=5.0).clamp(0.0, 1.0)
    dist = distance_like_field(mask)
    return soft, dist


@dataclass(slots=True)
class FrequencySplit:
    base: torch.Tensor
    detail: torch.Tensor


def split_frequencies(image: torch.Tensor, sigma: float = 5.0) -> FrequencySplit:
    image_b = ensure_batch(image)
    base = lowpass(image_b, sigma=sigma)
    detail = image_b - base
    return FrequencySplit(base=base, detail=detail)
