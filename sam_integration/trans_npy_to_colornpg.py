import os
import numpy as np
import cv2


# =========================
# 1. 读取 npy
# =========================

def load_mask_from_npy(npy_path: str):
    """
    支持以下几种 .npy:
    1. 普通 ndarray: H,W / N,H,W / H,W,N
    2. dict: {"mask": ...} / {"masks": ...} / {"segmentation": ...}
    3. SAM 常见 list[dict]: [{"segmentation": ...}, ...]
    """
    data = np.load(npy_path, allow_pickle=True)

    # object 类型，尝试解析
    if isinstance(data, np.ndarray) and data.dtype == object:
        try:
            if data.shape == ():
                data = data.item()
            else:
                data = data.tolist()
        except Exception:
            pass

    # dict 类型
    if isinstance(data, dict):
        for k in ["mask", "masks", "segmentation", "segments"]:
            if k in data:
                data = data[k]
                break

    # SAM list[dict]
    if isinstance(data, (list, tuple)):
        masks = []

        for item in data:
            if isinstance(item, dict):
                for k in ["segmentation", "mask", "masks"]:
                    if k in item:
                        masks.append(np.asarray(item[k]))
                        break
            elif isinstance(item, np.ndarray):
                masks.append(item)

        if len(masks) == 0:
            raise ValueError("list/tuple 中没有找到可解析的 mask")

        data = np.stack(masks, axis=0)

    if not isinstance(data, np.ndarray):
        raise ValueError("该 .npy 无法解析为 ndarray mask")

    return data


# =========================
# 2. 基础工具
# =========================

def normalize_to_uint8(mask: np.ndarray) -> np.ndarray:
    """
    任意数值 mask 转 uint8，用于保存灰度图。
    """
    mask = np.asarray(mask)

    if mask.dtype == np.bool_:
        return mask.astype(np.uint8) * 255

    mask = mask.astype(np.float32)

    min_val = np.nanmin(mask)
    max_val = np.nanmax(mask)

    if max_val - min_val < 1e-8:
        if max_val <= 1:
            return (mask * 255).astype(np.uint8)
        return np.clip(mask, 0, 255).astype(np.uint8)

    mask = (mask - min_val) / (max_val - min_val)
    return (mask * 255).astype(np.uint8)


def to_2d_mask(mask: np.ndarray) -> np.ndarray:
    """
    用于保存黑白/灰度预览图。
    """
    mask = np.asarray(mask)

    if mask.ndim == 2:
        return mask

    if mask.ndim == 3:
        # (1, H, W)
        if mask.shape[0] == 1:
            return mask[0]

        # (H, W, 1)
        if mask.shape[-1] == 1:
            return mask[:, :, 0]

        # (N, H, W)
        if mask.shape[0] < mask.shape[1] and mask.shape[0] < mask.shape[2]:
            return mask.max(axis=0)

        # (H, W, N)
        return mask.max(axis=-1)

    raise ValueError(f"不支持的 mask 维度: {mask.shape}")


def unique_preview(arr: np.ndarray, max_show: int = 20):
    """
    打印 unique 信息，方便判断是不是多类别 mask。
    """
    arr = np.asarray(arr)

    try:
        vals = np.unique(arr)
        if len(vals) <= max_show:
            return vals.tolist()
        else:
            return f"{len(vals)} values, first {max_show}: {vals[:max_show].tolist()}"
    except Exception:
        return "无法统计 unique"


# =========================
# 3. 判断 mask 类型
# =========================

def is_binary_like(mask: np.ndarray) -> bool:
    """
    判断是否是二值 mask。
    """
    mask = np.asarray(mask)

    if mask.dtype == np.bool_:
        return True

    vals = np.unique(mask)

    if len(vals) <= 2:
        return True

    try:
        vals_set = set(vals.tolist())
        if vals_set.issubset({0, 1, 255}):
            return True
    except Exception:
        pass

    return False


def is_label_mask_2d(mask_2d: np.ndarray, max_classes: int = 512) -> bool:
    """
    判断 2D mask 是否是多类别 label mask。

    例如:
    0 = 背景
    1 = 水体
    2 = 地面
    3 = 物体

    这种不能二值化。
    """
    mask_2d = np.asarray(mask_2d)

    if mask_2d.ndim != 2:
        return False

    if is_binary_like(mask_2d):
        return False

    vals = np.unique(mask_2d)

    # 类别数量不太多，通常就是 label mask
    if len(vals) <= max_classes:
        return True

    return False


# =========================
# 4. 生成 label_map
# =========================

def binary_mask_to_component_label_map(mask_2d: np.ndarray, min_area: int = 20) -> np.ndarray:
    """
    仅用于真正的二值 mask。

    逻辑:
    - 白色连通区域分别编号
    - 黑色连通区域也分别编号

    注意:
    如果原始 npy 本身只有 0/1，
    那么它本来就没有语义类别信息，无法知道哪个是地面、哪个是物体。
    """
    mask_uint8 = normalize_to_uint8(mask_2d)
    binary = (mask_uint8 > 127).astype(np.uint8)

    h, w = binary.shape
    label_map = np.zeros((h, w), dtype=np.int32)

    current_id = 1

    for target_value in [1, 0]:
        region_binary = (binary == target_value).astype(np.uint8)

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            region_binary,
            connectivity=8
        )

        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]

            if area < min_area:
                continue

            label_map[labels == i] = current_id
            current_id += 1

    return label_map


def label_mask_to_component_label_map(mask_2d: np.ndarray, min_area: int = 20) -> np.ndarray:
    """
    用于多类别 label mask。

    例如原始 mask:
    0 = 背景
    1 = 水体
    2 = 地面
    3 = 物体

    本函数不会二值化，而是保留每个 label。
    同一个 label 如果分成多个连通块，也会分别编号。
    """
    mask_2d = np.asarray(mask_2d)

    h, w = mask_2d.shape
    label_map = np.zeros((h, w), dtype=np.int32)

    unique_vals = np.unique(mask_2d)
    current_id = 1

    for val in unique_vals:
        # 跳过 NaN
        try:
            if np.isnan(val):
                continue
        except Exception:
            pass

        region_binary = (mask_2d == val).astype(np.uint8)

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            region_binary,
            connectivity=8
        )

        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]

            if area < min_area:
                continue

            label_map[labels == i] = current_id
            current_id += 1

    return label_map


def rgb_mask_to_label_map(mask_rgb: np.ndarray, min_area: int = 20) -> np.ndarray:
    """
    如果 npy 里存的是 RGB 彩色 mask，则按颜色转 label。
    """
    mask_rgb = np.asarray(mask_rgb)

    if mask_rgb.shape[-1] == 4:
        mask_rgb = mask_rgb[:, :, :3]

    h, w, _ = mask_rgb.shape
    flat = mask_rgb.reshape(-1, 3)

    colors, inverse = np.unique(flat, axis=0, return_inverse=True)
    color_label = inverse.reshape(h, w)

    return label_mask_to_component_label_map(color_label, min_area=min_area)


def mask_slice_to_binary(mask_2d: np.ndarray) -> np.ndarray:
    """
    多实例 mask 中的单个 mask 转二值。
    """
    mask_2d = np.asarray(mask_2d)

    if mask_2d.dtype == np.bool_:
        return mask_2d.astype(np.uint8)

    if is_binary_like(mask_2d):
        return (mask_2d > 0).astype(np.uint8)

    # 概率图 0~1
    if mask_2d.max() <= 1.0 and mask_2d.min() >= 0.0:
        return (mask_2d > 0.5).astype(np.uint8)

    # 其他情况归一化后二值化
    mask_uint8 = normalize_to_uint8(mask_2d)
    return (mask_uint8 > 127).astype(np.uint8)


def stacked_masks_to_label_map(masks: np.ndarray, min_area: int = 20) -> np.ndarray:
    """
    用于 SAM 或实例分割输出:
    - N,H,W
    - H,W,N

    每一个 mask 单独编号。
    """
    masks = np.asarray(masks)

    # (N, H, W)
    if masks.shape[0] < masks.shape[1] and masks.shape[0] < masks.shape[2]:
        masks_nhw = masks
    else:
        # (H, W, N) -> (N, H, W)
        masks_nhw = np.transpose(masks, (2, 0, 1))

    n, h, w = masks_nhw.shape

    label_map = np.zeros((h, w), dtype=np.int32)

    areas = []
    binaries = []

    for i in range(n):
        binary = mask_slice_to_binary(masks_nhw[i])
        area = int(binary.sum())

        binaries.append(binary)
        areas.append((i, area))

    # 大区域先画，小区域后画，避免小目标被覆盖
    areas = sorted(areas, key=lambda x: x[1], reverse=True)

    current_id = 1

    for i, area in areas:
        if area < min_area:
            continue

        region = binaries[i] > 0
        label_map[region] = current_id
        current_id += 1

    return label_map


def masks_to_label_map(mask: np.ndarray, min_area: int = 20) -> np.ndarray:
    """
    总入口:
    自动判断 mask 类型并转成 label_map。
    """
    mask = np.asarray(mask)

    if mask.ndim == 2:
        if is_label_mask_2d(mask):
            return label_mask_to_component_label_map(mask, min_area=min_area)
        else:
            return binary_mask_to_component_label_map(mask, min_area=min_area)

    if mask.ndim == 3:
        # (1, H, W)
        if mask.shape[0] == 1:
            m = mask[0]
            if is_label_mask_2d(m):
                return label_mask_to_component_label_map(m, min_area=min_area)
            else:
                return binary_mask_to_component_label_map(m, min_area=min_area)

        # (H, W, 1)
        if mask.shape[-1] == 1:
            m = mask[:, :, 0]
            if is_label_mask_2d(m):
                return label_mask_to_component_label_map(m, min_area=min_area)
            else:
                return binary_mask_to_component_label_map(m, min_area=min_area)

        # RGB 彩色 mask
        if mask.shape[-1] in [3, 4]:
            # 如果不是 0/1/255 这种二值堆叠，就认为是 RGB mask
            if not is_binary_like(mask):
                return rgb_mask_to_label_map(mask, min_area=min_area)

        # 多实例 mask 堆叠
        return stacked_masks_to_label_map(mask, min_area=min_area)

    raise ValueError(f"不支持的 mask 维度: {mask.shape}")


# =========================
# 5. 上色
# =========================

def get_pastel_color(idx: int) -> np.ndarray:
    """
    给每个区域生成柔和颜色。
    前面几个颜色固定，后面自动生成，避免颜色不够用。
    """
    base_palette = np.array([
        [132, 176, 245],  # 蓝
        [244, 185, 188],  # 粉
        [202, 214, 181],  # 灰绿
        [255, 184, 110],  # 橙
        [185, 170, 239],  # 紫
        [95, 150, 220],   # 深蓝
        [206, 150, 150],  # 棕粉
        [160, 205, 220],  # 青蓝
        [230, 205, 160],  # 浅土黄
        [220, 190, 230],  # 淡紫粉
        [170, 215, 170],  # 淡绿
        [245, 205, 210],  # 浅粉
        [180, 220, 245],  # 浅蓝
        [245, 220, 160],  # 米黄
        [190, 230, 190],  # 草绿
        [230, 180, 210],  # 粉紫
        [170, 200, 230],  # 灰蓝
        [240, 200, 170],  # 肉橙
        [200, 190, 240],  # 淡紫
        [190, 220, 210],  # 青绿
    ], dtype=np.uint8)

    if idx <= len(base_palette):
        color = base_palette[idx - 1].astype(np.float32)
    else:
        rng = np.random.default_rng(idx * 10007)
        color = rng.integers(70, 235, size=3).astype(np.float32)

    # 和白色混合，变成 pastel 风格
    color = 0.72 * color + 0.28 * np.array([255, 255, 255], dtype=np.float32)

    return np.clip(color, 0, 255).astype(np.uint8)


def pastel_colorize_label_map(
    label_map: np.ndarray,
    draw_boundary: bool = True,
    boundary_color=(150, 150, 150)
) -> np.ndarray:
    """
    label_map -> RGB 彩色图。
    """
    h, w = label_map.shape
    color_img = np.ones((h, w, 3), dtype=np.uint8) * 255

    unique_ids = np.unique(label_map)

    for idx in unique_ids:
        if idx == 0:
            continue

        region = label_map == idx
        color = get_pastel_color(int(idx))
        color_img[region] = color

    if draw_boundary:
        edges = np.zeros((h, w), dtype=np.uint8)

        for idx in unique_ids:
            if idx == 0:
                continue

            region = (label_map == idx).astype(np.uint8) * 255
            contours, _ = cv2.findContours(
                region,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )
            cv2.drawContours(edges, contours, -1, 255, 1)

        color_img[edges > 0] = np.array(boundary_color, dtype=np.uint8)

    return color_img


# =========================
# 6. 文件夹批量转换
# =========================

def convert_npy_folder_to_png(
    input_dir: str,
    output_dir: str,
    save_gray: bool = True,
    save_colormap: bool = True,
    save_label_npy: bool = False,
    draw_boundary: bool = True,
    min_area: int = 20
):
    os.makedirs(output_dir, exist_ok=True)

    npy_files = [
        f for f in os.listdir(input_dir)
        if f.lower().endswith(".npy")
    ]

    npy_files.sort()

    if len(npy_files) == 0:
        print(f"在目录中没有找到 .npy 文件: {input_dir}")
        return

    print(f"共找到 {len(npy_files)} 个 .npy 文件，开始转换...")
    print("-" * 80)

    for file_name in npy_files:
        npy_path = os.path.join(input_dir, file_name)
        base_name = os.path.splitext(file_name)[0]

        try:
            mask = load_mask_from_npy(npy_path)

            print(f"[读取] {file_name}")
            print(f"  原始 shape: {mask.shape}")
            print(f"  原始 dtype : {mask.dtype}")
            print(f"  原始 unique: {unique_preview(mask)}")

            # 保存灰度图
            if save_gray:
                gray_mask = to_2d_mask(mask)
                gray_mask = normalize_to_uint8(gray_mask)

                gray_path = os.path.join(output_dir, base_name + ".png")
                cv2.imwrite(gray_path, gray_mask)

            # 保存彩色图
            if save_colormap:
                label_map = masks_to_label_map(mask, min_area=min_area)

                print(f"  label_map unique: {unique_preview(label_map)}")
                print(f"  有效区域数: {len(np.unique(label_map)) - 1}")

                color_img_rgb = pastel_colorize_label_map(
                    label_map,
                    draw_boundary=draw_boundary
                )

                color_path = os.path.join(output_dir, base_name + "_pastel.png")
                cv2.imwrite(
                    color_path,
                    cv2.cvtColor(color_img_rgb, cv2.COLOR_RGB2BGR)
                )

                if save_label_npy:
                    label_path = os.path.join(output_dir, base_name + "_label_map.npy")
                    np.save(label_path, label_map)

            print(f"[成功] {file_name}")
            print("-" * 80)

        except Exception as e:
            print(f"[失败] {file_name}: {e}")
            print("-" * 80)

    print("转换完成！")


# =========================
# 7. 主函数
# =========================

if __name__ == "__main__":
    input_dir = "../tmp/train/mask_sam"
    output_dir = "mask_sam_png"

    convert_npy_folder_to_png(
        input_dir=input_dir,
        output_dir=output_dir,
        save_gray=True,
        save_colormap=True,
        save_label_npy=False,
        draw_boundary=True,
        min_area=20
    )