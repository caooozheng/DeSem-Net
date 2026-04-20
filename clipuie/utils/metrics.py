from __future__ import annotations

import math

import cv2
import numpy as np
import torch
from skimage.metrics import structural_similarity


def tensor_to_uint8(sample: torch.Tensor) -> np.ndarray:
    return (
        sample[0]
        .detach()
        .mul(255)
        .add_(0.5)
        .clamp_(0, 255)
        .permute(1, 2, 0)
        .to("cpu", torch.uint8)
        .numpy()
    )


def compute_psnr_numpy(image: np.ndarray, target: np.ndarray) -> float:
    mse = np.mean((image / 255.0 - target / 255.0) ** 2)
    if mse < 1e-10:
        return 100.0
    return 20 * math.log10(1.0 / math.sqrt(mse))


def compute_psnr_batch(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    mse = torch.mean((prediction - target) ** 2, dim=[1, 2, 3])
    mse = torch.clamp(mse, min=1e-10)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))


def calculate_uciqe(image: np.ndarray) -> float:
    image_lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    img_lum = image_lab[..., 0] / 255.0
    img_a = image_lab[..., 1] / 255.0
    img_b = image_lab[..., 2] / 255.0
    img_chr = np.sqrt(np.square(img_a) + np.square(img_b))
    img_sat = img_chr / np.sqrt(np.square(img_chr) + np.square(img_lum) + 1e-12)
    aver_sat = float(np.mean(img_sat))
    aver_chr = float(np.mean(img_chr))
    var_chr = float(np.sqrt(np.mean(np.abs(1 - np.square(aver_chr / (img_chr + 1e-12))))))
    hist, _ = np.histogram(img_lum, 256)
    cdf = np.cumsum(hist) / np.sum(hist)
    ilow = np.where(cdf > 0.0100)[0][0]
    ihigh = np.where(cdf >= 0.9900)[0][0]
    con_lum = (ihigh - 1) / 255.0 - (ilow - 1) / 255.0
    return 0.4680 * var_chr + 0.2745 * con_lum + 0.2576 * aver_sat


def _uicm(image: np.ndarray) -> float:
    image = image.astype(np.float64)
    red, green, blue = image[:, :, 0], image[:, :, 1], image[:, :, 2]
    rg = red - green
    yb = (red + green) / 2 - blue
    k = red.shape[0] * red.shape[1]
    rg1 = np.sort(rg.reshape(1, k))[0][int(0.1 * k + 1) : int(k * 0.9)]
    yb1 = np.sort(yb.reshape(1, k))[0][int(0.1 * k + 1) : int(k * 0.9)]
    n = k * 0.8
    mean_rg = np.sum(rg1) / n
    mean_yb = np.sum(yb1) / n
    delta_rg = np.sqrt(np.sum((rg1 - mean_rg) ** 2) / n)
    delta_yb = np.sqrt(np.sum((yb1 - mean_yb) ** 2) / n)
    return -0.0268 * np.sqrt(mean_rg ** 2 + mean_yb ** 2) + 0.1586 * np.sqrt(delta_yb ** 2 + delta_rg ** 2)


def _uiconm(image: np.ndarray) -> float:
    image = image.astype(np.float64)
    patch_size = 5
    total = 0.0
    for channel in range(3):
        channel_image = image[:, :, channel]
        height, width = channel_image.shape
        if height % patch_size != 0 or width % patch_size != 0:
            channel_image = cv2.resize(channel_image, (width - width % patch_size + patch_size, height - height % patch_size + patch_size))
            height, width = channel_image.shape
        accum = 0.0
        for i in range(0, height, patch_size):
            for j in range(0, width, patch_size):
                patch = channel_image[i : i + patch_size, j : j + patch_size]
                patch_max = np.max(patch)
                patch_min = np.min(patch)
                if (patch_max != 0 or patch_min != 0) and patch_max != patch_min:
                    ratio = (patch_max - patch_min) / (patch_max + patch_min)
                    accum += np.log(ratio) * ratio
        total += abs(accum) / ((height / patch_size) * (width / patch_size))
    return total


def _uism(image: np.ndarray) -> float:
    image = image.astype(np.float64)
    kernel_x = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    kernel_y = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    lambdas = [0.299, 0.587, 0.114]
    patch_size = 5
    total = 0.0
    for channel, weight in enumerate(lambdas):
        sobel = np.abs(cv2.filter2D(image[:, :, channel], -1, kernel_x) + cv2.filter2D(image[:, :, channel], -1, kernel_y))
        height, width = sobel.shape
        if height % patch_size != 0 or width % patch_size != 0:
            sobel = cv2.resize(sobel, (width - width % patch_size + patch_size, height - height % patch_size + patch_size))
            height, width = sobel.shape
        emer = 0.0
        for i in range(0, height, patch_size):
            for j in range(0, width, patch_size):
                patch = sobel[i : i + patch_size, j : j + patch_size]
                patch_max = np.max(patch)
                patch_min = np.min(patch)
                if patch_max != 0 and patch_min != 0:
                    emer += np.log(patch_max / patch_min)
        total += weight * (2 * abs(emer) / ((height / patch_size) * (width / patch_size)))
    return total


def calculate_uiqm(image: np.ndarray) -> float:
    return 0.0282 * _uicm(image) + 0.2953 * _uism(image) + 3.5753 * _uiconm(image)


class ImageMetricRunner:
    def __init__(self, compute_uiqm_flag: bool = True, compute_uciqe_flag: bool = True) -> None:
        self.compute_uiqm_flag = compute_uiqm_flag
        self.compute_uciqe_flag = compute_uciqe_flag

    def evaluate_pair(self, prediction: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
        pred_uint8 = tensor_to_uint8(prediction)
        target_uint8 = tensor_to_uint8(target)
        metrics = {
            "psnr_256": compute_psnr_numpy(pred_uint8, target_uint8),
            "ssim_256": structural_similarity(target_uint8, pred_uint8, channel_axis=2, data_range=255),
        }
        if self.compute_uiqm_flag:
            metrics["uiqm_256"] = float(calculate_uiqm(pred_uint8))
        if self.compute_uciqe_flag:
            metrics["uciqe_256"] = float(calculate_uciqe(pred_uint8))
        return metrics
