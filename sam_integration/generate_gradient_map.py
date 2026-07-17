import cv2
import numpy as np


def generate_enhanced_gradient_map(image_path, save_path="gradient_map_enhanced.png"):
    img = cv2.imread(image_path)

    if img is None:
        raise FileNotFoundError(f"无法读取图片: {image_path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 轻微高斯平滑，减少噪声
    gray_blur = cv2.GaussianBlur(gray, (3, 3), 0)

    # Sobel 梯度
    grad_x = cv2.Sobel(gray_blur, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray_blur, cv2.CV_32F, 0, 1, ksize=3)

    grad = np.sqrt(grad_x ** 2 + grad_y ** 2)

    # 归一化
    grad = grad / (grad.max() + 1e-8)

    # gamma 增强，让弱边缘更明显
    gamma = 0.6
    grad = np.power(grad, gamma)

    grad = (grad * 255).astype(np.uint8)

    cv2.imwrite(save_path, grad)
    print(f"增强版 Gradient Map 已保存: {save_path}")


if __name__ == "__main__":
    generate_enhanced_gradient_map(
        image_path="GT.jpg",
        save_path="GT_grad.jpg"
    )