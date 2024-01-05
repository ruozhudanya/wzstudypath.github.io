import cv2
import tkinter as tk
import os
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
from image_processing import enhance_quality
from Image_training_model import create_super_resolution_model

class ImageEnhancementGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("图像增强工具")

        # 创建画布用于显示图片/视频
        self.canvas_width = 800  # 设置画布的固定宽度
        self.canvas_height = 600  # 设置画布的固定高度
        self.canvas = tk.Canvas(root, width=self.canvas_width, height=self.canvas_height)
        self.canvas.pack()

        # 创建界面元素
        self.upload_button = tk.Button(root, text="上传文件", command=self.upload_file)
        self.show_button = tk.Button(root, text="展示文件", command=self.show_file)
        self.enhance_button = tk.Button(root, text="增强画质", command=self.enhance_quality)
        self.save_button = tk.Button(root, text="保存文件", command=self.save_file)
        self.quit_button = tk.Button(root, text="退出操作", command=root.destroy)

        # 设置按钮布局
        self.upload_button.pack(side=tk.LEFT, padx=5, anchor=tk.CENTER)
        self.show_button.pack(side=tk.LEFT, padx=5, anchor=tk.CENTER)
        self.enhance_button.pack(side=tk.LEFT, padx=5, anchor=tk.CENTER)
        self.save_button.pack(side=tk.LEFT, padx=5, anchor=tk.CENTER)
        self.quit_button.pack(side=tk.LEFT, padx=5, anchor=tk.CENTER)

        # 初始化文件路径变量和图片变量
        self.file_path = None
        self.image_before = None
        self.image_after = None

        # 创建超分辨率模型
        self.super_resolution_model = create_super_resolution_model()

    # 上传文件
    def upload_file(self):
        self.file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.png;*.jpeg;*.gif")], initialdir="C:/Users/18092/Pictures")
        if self.file_path:
            print(f"已上传文件: {self.file_path}")

    # 展示文件路径，并在画布上展示图片
    def show_file(self):
        if self.file_path:
            print(f"当前文件路径: {self.file_path}")

            # 读取图片/视频并在画布上展示
            try:
                self.image_before = self.load_image(self.file_path)
                if self.image_before:
                    self.display_image(self.image_before)
                else:
                    print("无法加载图片")
            except Exception as e:
                print(f"错误: {e}")
        else:
            print("请先上传文件")

    # 加载图片/视频
    def load_image(self, path):
        try:
            with open(path, 'rb') as f:
                img_array = np.frombuffer(f.read(), dtype=np.uint8)
            image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            if image is None or image.size == 0:
                raise Exception("无法读取文件")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # 缩放图片以适应画布大小
            img = Image.fromarray(image)
            img = self.resize_image(img, self.canvas_width, self.canvas_height)

            return img
        except Exception as e:
            print(f"加载图片时出错: {e}")
            return None

    # 在画布上展示图片
    def display_image(self, img):
        self.canvas.config(width=img.width, height=img.height)
        photo = ImageTk.PhotoImage(img)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=photo)
        self.canvas.image = photo

    # 缩放图片以适应画布大小
    def resize_image(self, img, max_width, max_height):
        width, height = img.size
        if width > max_width or height > max_height:
            ratio = min(max_width / width, max_height / height)
            new_width = int(width * ratio)
            new_height = int(height * ratio)
            img = img.resize((new_width, new_height), Image.LANCZOS)
        return img

    # 增强画质
    def enhance_quality(self):
        if self.file_path:
            try:
                # 读取原始图像
                with open(self.file_path, "rb") as f:
                    image_data = f.read()
                original_image = cv2.imdecode(np.frombuffer(image_data, dtype=np.uint8), -1)

                if original_image is None:
                    raise Exception("无法读取图像")

                # 调用图像处理模块中的函数
                enhanced_image = enhance_quality(original_image)
                # 显示增强后的图片
                self.image_after = Image.fromarray(cv2.cvtColor(enhanced_image, cv2.COLOR_BGR2RGB))
                self.display_image(self.image_after)
            except Exception as e:
                print(f"错误: {e}")
        else:
            print("请先上传文件")

    # 保存文件
    def save_file(self):
        if self.image_after:
            output_path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG files", "*.png")])
            if output_path:
                self.image_after.save(output_path)
                print(f"文件已保存至: {output_path}")
        else:
            print("请先进行画质增强")


# 创建主窗口
root = tk.Tk()
app = ImageEnhancementGUI(root)

# 运行主程序
root.mainloop()
