import tkinter as tk
from tkinter import filedialog, messagebox, Canvas
from PIL import ImageTk, Image
import torch
from torchvision import transforms
from catsvsdogs import Net

# 设定图像预处理步骤
data_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 初始化Tkinter窗口
root = tk.Tk()
root.title("猫狗分类器")

# 加载模型
model = Net()
model.load_state_dict(torch.load('cats_vs_dogs_custom.pt'))
model.eval()


# 定义图像预测函数
def predict_image():
    global model
    file_path = filedialog.askopenfilename()
    try:
        image = Image.open(file_path)
        image = data_transform(image).unsqueeze(0)  # 添加批处理维度
        with torch.no_grad():
            outputs = model(image)
            _, predicted = torch.max(outputs, 1)
            if predicted.item() == 0:
                result_label.config(text="预测结果: 猫")
            else:
                result_label.config(text="预测结果: 狗")

            # 显示图像在Canvas上
            img = Image.open(file_path)
            img = img.resize((200, 200))  # 调整图像大小
            img_tk = ImageTk.PhotoImage(img)
            canvas.delete("all")
            canvas.create_image(0, 0, anchor=tk.NW, image=img_tk)
            canvas.image = img_tk  # 保持引用，防止被垃圾回收
    except Exception as e:
        messagebox.showerror("错误", f"无法加载图像：{e}")


# 创建GUI组件
root.title("猫狗识别")
root.geometry("300x150+700+300")
root.minsize(560, 545)
root.maxsize(560, 545)
root["background"] ="#CDCDB4"#
browse_button = tk.Button(root, text="选择图像", command=predict_image)
browse_button.pack(pady=20)

result_label = tk.Label(root, text="预测结果: ")
result_label.pack(pady=10)

canvas = Canvas(root, width=200, height=200)
canvas.pack(pady=10)

# 运行Tkinter主循环
root.mainloop()
