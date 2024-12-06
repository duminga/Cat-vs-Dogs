import os
import random
import matplotlib.pyplot as plt
from PIL import Image
import matplotlib
matplotlib.rc("font", family='LiSu')

# 定义数据集路径
base_dir = 'dataset'
sub_dirs = ['train', 'test']
animals = ['cats', 'dogs']

# 存储图片数量的字典
data_count = {'train': {'cats': 0, 'dogs': 0}, 'test': {'cats': 0, 'dogs': 0}}

# 计算每个类别的图片数量
for sub_dir in sub_dirs:
    for animal in animals:
        animal_dir = os.path.join(base_dir, sub_dir, animal)
        data_count[sub_dir][animal] = len(os.listdir(animal_dir))


# 数据可视化
fig, ax = plt.subplots(1, 2, figsize=(12, 6))

# 绘制训练集数据分布
ax[0].bar(data_count['train'].keys(), data_count['train'].values(), color=['blue', 'green'])
ax[0].set_title('训练集')
ax[0].set_xlabel('类别')
ax[0].set_ylabel('图片数量')

# 绘制测试集数据分布
ax[1].bar(data_count['test'].keys(), data_count['test'].values(), color=['blue', 'green'])
ax[1].set_title('测试集')
ax[1].set_xlabel('类别')
ax[1].set_ylabel('图片数量')

# 调整布局
plt.tight_layout()
plt.show()

# 定义函数来展示随机图片样本
def show_random_images(directory, title, n=10):
    filenames = random.sample(os.listdir(directory), n)
    plt.figure(figsize=(15, 5))
    for i, filename in enumerate(filenames):
        filepath = os.path.join(directory, filename)
        img = Image.open(filepath)
        plt.subplot(1, n, i+1)
        plt.imshow(img)
        plt.title(filename)
        plt.axis('off')
    plt.suptitle(title)
    plt.show()

# 展示随机图片样本
show_random_images(os.path.join(base_dir, 'train', 'cats'), '随机抽取训练集猫图片')
show_random_images(os.path.join(base_dir, 'train', 'dogs'), '随机抽取训练集狗图片')
show_random_images(os.path.join(base_dir, 'test', 'cats'), '随机抽取测试集猫图片')
show_random_images(os.path.join(base_dir, 'test', 'dogs'), '随机抽取测试集狗图片')
