import tkinter as tk
from tkinter import ttk
from ttkthemes import ThemedTk
from tkinter import filedialog
from tkinter import messagebox
from PIL import Image, ImageTk
import os
import re
import subprocess


def find_matching_file(filename, directory):
    for file in os.listdir(directory):
        if os.path.splitext(file)[1] == ".jpg":
            match = re.match(r".+_" + filename + r"\.jpg", file)
            if match:
                return os.path.join(directory, file)
    return None


def delete_old(path: str):
    folder = f"{path}"  # 将此路径替换为要清除文件的文件夹路径

    # 列出文件夹中的所有文件
    files = os.listdir(folder)

    # 遍历文件并删除每个文件
    for file in files:
        file_path = os.path.join(folder, file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception as e:
            print(f"无法删除文件：{file_path}，错误：{e}")


def select_image():
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.png")])
    if file_path:
        input_image_label.config(text=file_path)

        # 显示输入图片
        img = Image.open(file_path)
        img.thumbnail((300, 300))
        img = ImageTk.PhotoImage(img)
        input_image_display.config(image=img)
        input_image_display.image = img


def get_latest_image(output_dir):
    images = [
        os.path.join(output_dir, f)
        for f in os.listdir(output_dir)
        if os.path.splitext(f)[-1].lower() in [".jpg", ".png", ".bmp", ".tiff"]
    ]
    if not images:
        return None

    latest_image = max(images, key=os.path.getctime)
    return latest_image


def generate():
    import os
    selected_weight = weight_var.get()
    input_image_path = input_image_label.cget("text")
    output_dir = os.path.abspath("C:\\PythonProject\\re_animegen2\\samples\\results")
    delete_old(output_dir)
    if not input_image_path or input_image_path == "No image selected":
        messagebox.showerror("Error", "Please select an input image.")
        return

    # 在这里编写生成CMD脚本并运行的代码

    # 获取图片文件所在的文件夹路径
    input_dir = os.path.dirname(input_image_path)
    input_image_name = os.path.basename(input_image_path)

    # input_dir = input_image_path
    # 构造 CMD 命令
    cmd = f'cd "C:\\PythonProject\\re_animegen2" && conda activate ldm && python main.py --input_dir "{input_dir}"  --checkpoint "C:\\PythonProject\\re_animegen2\\weights\\{selected_weight}.pt"'

    # 运行 CMD 命令
    subprocess.run(cmd, shell=True, check=True)

    # 获取生成图片文件所在的文件夹路径

    # 使用 get_latest_image 函数获取最新生成的图像
    latest_image_path = get_latest_image(output_dir)

    # 检查是否找到了图片文件
    if latest_image_path is None:
        messagebox.showerror("Error", "No output image found.")
        return

    img = Image.open(latest_image_path)
    img.thumbnail((300, 300))  # 缩放图像以适应显示区域
    img = ImageTk.PhotoImage(img)
    output_image_display.config(image=img, text="")
    output_image_display.image = img


root = ThemedTk(theme="arc")
root.title("Image Generation")

root.geometry("720x1080")

style = ttk.Style(root)
style.configure("TButton", relief="flat")

root.grid_columnconfigure(0, weight=1)
root.grid_columnconfigure(1, weight=1)
root.grid_rowconfigure(0, weight=1)
root.grid_rowconfigure(1, weight=1)
root.grid_rowconfigure(2, weight=1)

select_button = ttk.Button(root, text="Select Image", command=select_image)
select_button.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")

input_image_label = ttk.Label(root, text="No image selected", width=40, anchor="w")
input_image_label.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")

weight_var = tk.StringVar(root)
weight_var.set("celeba_distill")

weights = [
    "celeba_distill",
    "face_paint_512_v1",
    "face_paint_512_v2",
    "paprika"
]

weight_menu = ttk.OptionMenu(root, weight_var, *weights)
weight_menu.grid(row=1, column=0, padx=10, pady=10, sticky="nsew")

generate_button = ttk.Button(root, text="Generate", command=generate)
generate_button.grid(row=1, column=1, padx=10, pady=10, sticky="nsew")

# 显示输入和输出图像的标签
input_image_display = ttk.Label(root, text="Input image will be displayed here")
input_image_display.grid(row=2, column=0, padx=10, pady=10, sticky="nsew")

output_image_display = ttk.Label(root, text="Output image will be displayed here")
output_image_display.grid(row=2, column=1, padx=10, pady=10, sticky="nsew")

root.mainloop()
