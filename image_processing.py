# image_processing.py
import cv2
import numpy as np

def enhance_quality(image):
    # 对图像进行增强处理，这里可以根据需求添加其他处理步骤
    enhanced_image = denoise(image)
    enhanced_image = increase_contrast(enhanced_image)
    enhanced_image = sharpen(enhanced_image)
    enhanced_image = detail_enhance(enhanced_image)
    enhanced_image = clahe_enhance(enhanced_image)
    enhanced_image = white_balance(enhanced_image)
    enhanced_image = adaptive_contrast_enhance(enhanced_image)
    enhanced_image = color_enhance(enhanced_image)

    return enhanced_image

def denoise(image):
    # 降噪处理，这里使用双边滤波器
    denoised_image = cv2.bilateralFilter(image, d=9, sigmaColor=75, sigmaSpace=75)
    return denoised_image

def increase_contrast(image):
    # 增强对比度处理
    enhanced_image = cv2.convertScaleAbs(image, alpha=1.5, beta=10)
    return enhanced_image

def sharpen(image):
    # 锐化处理，这里使用拉普拉斯算子
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharpened_image = cv2.filter2D(image, -1, kernel)
    return sharpened_image

def detail_enhance(image):
    # 细节增强处理
    enhanced_image = cv2.detailEnhance(image, sigma_s=10, sigma_r=0.15)
    return enhanced_image


def clahe_enhance(image):
    # 使用CLAHE进行对比度受限自适应直方图均衡化
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_gray = clahe.apply(gray)
    enhanced_image = cv2.cvtColor(enhanced_gray, cv2.COLOR_GRAY2BGR)

    return enhanced_image

def white_balance(image):
    # 直方图均衡化进行简单的白平衡
    result = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(result)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    result = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    return result

def adaptive_contrast_enhance(image):
    # 自适应对比度增强
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    result = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    return result

def color_enhance(image):
    # 调整颜色饱和度
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv[..., 1] = hsv[..., 1] * 1.5  # 调整饱和度
    enhanced_image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    return enhanced_image