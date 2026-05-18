import os
import numpy as np
import cv2


def load_mask_from_npy(npy_path: str):
    mask = np.load(npy_path, allow_pickle=True)

    if isinstance(mask, np.ndarray) and mask.dtype == object:
        try:
            mask = mask.item()
        except Exception:
            pass

    if isinstance(mask, dict):
        for k in ["mask", "masks", "segmentation", "segments"]:
            if k in mask:
                mask = mask[k]
                break

    if not isinstance(mask, np.ndarray):
        raise ValueError("该 .npy 不是普通 ndarray，且无法自动解析")

    return mask


def to_2d_mask(mask: np.ndarray) -> np.ndarray:
    """
    用于保存黑白图。
    """
    if mask.ndim == 2:
        return mask

    elif mask.ndim == 3:
        if mask.shape[0] == 1:
            return mask[0]
        elif mask.shape[-1] == 1:
            return mask[:, :, 0]
        elif mask.shape[0] < mask.shape[1] and mask.shape[0] < mask.shape[2]:
            return mask.max(axis=0)
        else:
            return mask.max(axis=-1)

    else:
        raise ValueError(f"不支持的 mask 维度: {mask.shape}")


def normalize_to_uint8(mask_2d: np.ndarray) -> np.ndarray:
    mask_2d = mask_2d.astype(np.float32)

    min_val = mask_2d.min()
    max_val = mask_2d.max()

    if max_val - min_val < 1e-8:
        if max_val <= 1:
            return (mask_2d * 255).astype(np.uint8)
        return np.clip(mask_2d, 0, 255).astype(np.uint8)

    mask_2d = (mask_2d - min_val) / (max_val - min_val)
    return (mask_2d * 255).astype(np.uint8)


def binary_mask_to_component_label_map(mask_2d: np.ndarray, min_area: int = 80) -> np.ndarray:
    """
    关键修改：
    把黑白二值 mask 中的每一个连通区域都单独编号。

    原来:
    - 黑色是一类
    - 白色是一类

    现在:
    - 每一个白色连通块是一个类别
    - 每一个黑色连通块也是一个类别

    这样才能生成多颜色区域图。
    """
    mask_uint8 = normalize_to_uint8(mask_2d)

    # 二值化
    binary = (mask_uint8 > 127).astype(np.uint8)

    h, w = binary.shape
    label_map = np.zeros((h, w), dtype=np.int32)

    current_id = 1

    # 1. 给白色区域上不同标签
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)

    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < min_area:
            continue
        label_map[labels == i] = current_id
        current_id += 1

    # 2. 给黑色区域也上不同标签
    inv_binary = 1 - binary
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(inv_binary, connectivity=8)

    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < min_area:
            continue
        label_map[labels == i] = current_id
        current_id += 1

    return label_map


def masks_to_label_map(mask: np.ndarray) -> np.ndarray:
    """
    生成彩色图用的 label map。

    如果是二值 mask：
    - 使用连通域分割，让每个区域不同颜色。

    如果是多 mask 堆叠：
    - 每个 mask 作为一个实例区域。
    """
    if mask.ndim == 2:
        return binary_mask_to_component_label_map(mask)

    if mask.ndim == 3:
        # (1, H, W)
        if mask.shape[0] == 1:
            return binary_mask_to_component_label_map(mask[0])

        # (H, W, 1)
        if mask.shape[-1] == 1:
            return binary_mask_to_component_label_map(mask[:, :, 0])

        # (N, H, W)
        if mask.shape[0] < mask.shape[1] and mask.shape[0] < mask.shape[2]:
            masks = mask

        # (H, W, N)
        else:
            masks = np.transpose(mask, (2, 0, 1))

        n, h, w = masks.shape
        label_map = np.zeros((h, w), dtype=np.int32)

        areas = []
        for i in range(n):
            area = np.sum(masks[i] > 0)
            areas.append((i, area))

        # 大区域先画，小区域后画，避免小区域被覆盖
        areas = sorted(areas, key=lambda x: x[1], reverse=True)

        current_id = 1
        for i, area in areas:
            if area == 0:
                continue
            region = masks[i] > 0
            label_map[region] = current_id
            current_id += 1

        # 如果多 mask 堆叠后仍然只有一个区域，就再用连通域兜底
        if len(np.unique(label_map)) <= 2:
            merged = masks.max(axis=0)
            label_map = binary_mask_to_component_label_map(merged)

        return label_map

    raise ValueError(f"不支持的 mask 维度: {mask.shape}")


def pastel_colorize_label_map(label_map: np.ndarray, draw_boundary: bool = True) -> np.ndarray:
    """
    柔和 pastel 风格上色。
    """
    h, w = label_map.shape
    color_img = np.ones((h, w, 3), dtype=np.uint8) * 255

    # 更接近你发的那张示例图的柔和配色
    pastel_palette = np.array([
        [132, 176, 245],  # 蓝
        [244, 185, 188],  # 粉
        [202, 214, 181],  # 灰绿
        [255, 184, 110],  # 橙
        [185, 170, 239],  # 紫
        [95, 150, 220],   # 深一点蓝
        [206, 150, 150],  # 棕粉
        [160, 205, 220],  # 青蓝
        [230, 205, 160],  # 浅土黄
        [220, 190, 230],  # 淡紫粉
        [170, 215, 170],  # 淡绿
        [245, 205, 210],  # 浅粉
    ], dtype=np.uint8)

    unique_ids = np.unique(label_map)

    for idx in unique_ids:
        if idx == 0:
            continue

        region = label_map == idx
        color = pastel_palette[(idx - 1) % len(pastel_palette)]

        # 关键参数：和白色混合，降低饱和度
        # 如果还觉得太亮，把 0.70 改成 0.60
        # 如果觉得太淡，把 0.70 改成 0.80
        color = (0.72 * color + 0.28 * np.array([255, 255, 255])).astype(np.uint8)

        color_img[region] = color

    if draw_boundary:
        edges = np.zeros((h, w), dtype=np.uint8)

        for idx in unique_ids:
            if idx == 0:
                continue

            region = (label_map == idx).astype(np.uint8) * 255
            contours, _ = cv2.findContours(region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(edges, contours, -1, 255, 1)

        # 边界用浅灰，不用黑色，避免刺眼
        color_img[edges > 0] = [150, 150, 150]

    return color_img


def convert_npy_folder_to_png(
    input_dir,
    output_dir,
    save_colormap=True,
    save_gray=True,
    draw_boundary=True,
):
    os.makedirs(output_dir, exist_ok=True)

    npy_files = [f for f in os.listdir(input_dir) if f.endswith(".npy")]
    npy_files.sort()

    if len(npy_files) == 0:
        print(f"在目录中没有找到 .npy 文件: {input_dir}")
        return

    print(f"共找到 {len(npy_files)} 个 .npy 文件，开始转换...")

    for file_name in npy_files:
        npy_path = os.path.join(input_dir, file_name)

        try:
            mask = load_mask_from_npy(npy_path)
            base_name = os.path.splitext(file_name)[0]

            # 保存黑白图
            if save_gray:
                vis_mask = to_2d_mask(mask)
                vis_mask = normalize_to_uint8(vis_mask)

                gray_path = os.path.join(output_dir, base_name + ".png")
                cv2.imwrite(gray_path, vis_mask)

            # 保存柔和多区域彩色图
            if save_colormap:
                label_map = masks_to_label_map(mask)
                color_img_rgb = pastel_colorize_label_map(
                    label_map,
                    draw_boundary=draw_boundary
                )

                color_path = os.path.join(output_dir, base_name + "_pastel.png")
                cv2.imwrite(color_path, cv2.cvtColor(color_img_rgb, cv2.COLOR_RGB2BGR))

            print(f"[成功] {file_name} | 原始 shape={getattr(mask, 'shape', None)}")

        except Exception as e:
            print(f"[失败] {file_name}: {e}")

    print("转换完成！")


if __name__ == "__main__":
    input_dir = "mask_sam"
    output_dir = "mask_sam_png"

    save_gray = True
    save_colormap = True
    draw_boundary = True

    convert_npy_folder_to_png(
        input_dir=input_dir,
        output_dir=output_dir,
        save_colormap=save_colormap,
        save_gray=save_gray,
        draw_boundary=draw_boundary,
    )