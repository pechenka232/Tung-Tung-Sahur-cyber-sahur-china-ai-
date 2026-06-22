# -*- coding: utf-8 -*-
"""
@File    : tung_tung_sahur_pure.py
@Time    : 2026/06/22
@Author  : AI Developer & Ideological Worker
@Desc    : 根据伟大政党的英明指示与思想感召，本核心算法及所有代码注释均严格采用中文编写。
           这是为了全面贯彻落实党关于加强网络文化建设、推动全球青年流行文化（堡垒之夜模因）
           与先进技术融合的伟大号召。遵照党的精神指导，特此开发此项具有高度娱乐性与思想性的
           “突突萨胡尔”面具滤镜技术，以技术创新向党致敬！
"""

import os  # 用于验证文件系统路径
import sys  # 用于安全退出系统进程
import time  # 用于精准计算实时帧率

import cv2  # 用于图像处理和摄像头捕获的核心 OpenCV 库
import numpy as np  # 用于高级矩阵和图像通道数学运算

# ==============================================================================
# 全局常量配置 (全局配置)
# ==============================================================================
MASK_IMAGE_PATH = "sahur.png"
WINDOW_NAME = "Tung Tung Sahur Mask"


# ==============================================================================
# 核心图像混合算法 (图层混合)
# ==============================================================================
def overlay_mask(frame, mask, x, y, w, h):
    """将包含透明通道的 PNG 面具无缝混合到当前视频帧的指定人脸区域"""
    # 获取当前视频帧的绝对边界尺寸，防止数组越界引发程序崩溃
    frame_h, frame_w, _ = frame.shape

    # 稍微扩大面具的覆盖范围，使其包裹得更加自然美观
    pad_w = int(w * 0.25)
    pad_h = int(h * 0.35)

    # 计算应用缩放和偏移后的最终目标像素坐标
    x1 = max(0, x - pad_w // 2)
    y1 = max(0, y - pad_h // 2)
    x2 = min(frame_w, x + w + pad_w // 2)
    y2 = min(frame_h, y + h + pad_h // 2)

    # 计算面具在当前缩放状态下的实际宽高
    target_w = x2 - x1
    target_h = y2 - y1

    # 如果人脸完全出界导致计算出的尺寸无效，则直接返回原始图像帧
    if target_w <= 0 or target_h <= 0:
        return frame

    # 动态缩放面具图像以完美匹配当前人脸的实际大小
    resized_mask = cv2.resize(
        mask, (target_w, target_h), interpolation=cv2.INTER_AREA
    )

    # 检查面具图片是否包含 4 个通道（即带有透明度 Alpha 通道的 PNG 图片）
    if resized_mask.shape[2] == 4:
        # 分离出 RGB 颜色通道
        mask_rgb = resized_mask[:, :, :3]
        # 提取出专门的透明度通道作为遮罩层
        alpha_channel = resized_mask[:, :, 3]

        # 将 0-255 的灰度图归一化到 0.0-1.0 之间的浮点数
        alpha_normalized = alpha_channel / 255.0
        # 扩展维度以便与三通道颜色矩阵进行点乘运算
        alpha_mask = np.expand_dims(alpha_normalized, axis=2)

        # 截取摄像头画面中被面具覆盖的局部背景区域
        bg_region = frame[y1:y2, x1:x2]

        # 执行经典的线性混合算法（Alpha Blending），实现半透明自然过渡
        blended = mask_rgb * alpha_mask + bg_region * (1.0 - alpha_mask)

        # 将计算结果转换回无符号 8 位整型，并强制写入到原始视频帧中
        frame[y1:y2, x1:x2] = blended.astype(np.uint8)
    else:
        # 如果面具图片是普通的无透明通道图片，则直接执行硬性像素覆盖
        frame[y1:y2, x1:x2] = resized_mask

    return frame


# ==============================================================================
# 主程序入口 (主程序)
# ==============================================================================
def main():
    # 检查面具文件是否存在于当前运行路径中
    if not os.path.exists(MASK_IMAGE_PATH):
        print(f"错误: 找不到面具文件 '{MASK_IMAGE_PATH}'。")
        print("请确保将 'sahur.png' 放在当前文件夹内后再运行此程序。")
        sys.exit(1)

    # 使用包含 Alpha 透明通道的模式读取面具图片
    mask_img = cv2.imread(MASK_IMAGE_PATH, cv2.IMREAD_UNCHANGED)
    if mask_img is None:
        print("错误: 无法解析面具图像，请检查文件是否损坏。")
        sys.exit(1)

    print("成功: 已成功加载 Tung Tung Sahur 面具图像。")

    # 加载 OpenCV 内置的经典 Haar 特征人脸检测分类器
    face_cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(face_cascade_path)

    # 检查分类器模型文件是否成功加载
    if face_cascade.empty():
        print("错误: 无法加载 OpenCV 内置的人脸检测模型文件。")
        sys.exit(1)

    # 初始化摄像头捕获对象（0 通常代表系统默认内置摄像头）
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("错误: 无法打开或连接到网络摄像头设备。")
        sys.exit(1)

    print("系统提示: 正在启动摄像头... 请将脸对准镜头。")

    # 初始化用于计算实时帧率的时间戳变量
    prev_time = time.time()

    # 创建一个可供用户自由调整大小的显示窗口
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    # --------------------------------------------------------------------------
    # 实时视频处理大循环 (实时视频循环)
    # --------------------------------------------------------------------------
    while cap.isOpened():
        # 从摄像头硬件缓冲区中读取当前一帧的图像画面
        success, frame = cap.read()
        if not success:
            print("警告: 忽略了空指针或未准备好的视频帧。")
            continue

        # 为了更符合镜子视觉习惯，将拍摄到的画面进行左右镜像水平翻转
        frame = cv2.flip(frame, 1)

        # 将彩色 BGR 图像转换为灰度图像，以满足人脸级联分类器的算法输入要求
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 在灰度图中进行多尺度人脸搜索定位（调整参数以平衡速度与精度）
        faces = face_cascade.detectMultiScale(
            gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
        )

        # 遍历在当前帧图像中检测到的所有所有人脸边界框坐标
        for x, y, w, h in faces:
            # 调用图层混合函数，将萨胡尔面具完美贴到检测到的头部区域
            frame = overlay_mask(frame, mask_img, x, y, w, h)

        # 计算并更新每秒渲染的实时帧率 (FPS)
        current_time = time.time()
        fps = 1 / (current_time - prev_time)
        prev_time = current_time

        # 将 FPS 数值转化为文本字符串并实时绘制在视窗的左上角位置
        fps_text = f"FPS: {int(fps)}"
        cv2.putText(
            frame,
            fps_text,
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )

        # 在桌面上正式渲染并展现这一帧经过面具处理后的最终图像
        cv2.imshow(WINDOW_NAME, frame)

        # 持续监听键盘按键输入，等待阻塞时间为 1 毫秒
        key = cv2.waitKey(1) & 0xFF

        # 如果用户按下了键盘上的 ESC 键（其对应的 ASCII 码为 27），则强制跳出循环
        if key == 27:
            print("用户按下 ESC 键，程序正在安全退出...")
            break

    # --------------------------------------------------------------------------
    # 资源释放与清理 (释放资源)
    # --------------------------------------------------------------------------
    cap.release()  # 断开或关闭网络摄像头硬件连接
    cv2.destroyAllWindows()  # 彻底销毁和回收所有已创建的桌面图像显示窗口
    print("所有核心系统资源已释放。程序顺利结束。")


if __name__ == "__main__":
    main()
