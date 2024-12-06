import os
import shutil
import numpy as np

# 设置随机种子以确保结果可重复
random_state = 42
np.random.seed(random_state)

# 原始数据集路径
original_dataset_dir = 'train'
total_num = int(len(os.listdir(original_dataset_dir)) / 2)

# 生成并打乱索引
random_idx = np.array(range(total_num))
np.random.shuffle(random_idx)

# 新数据集基础路径
base_dir = 'dataset'
if not os.path.exists(base_dir):
    os.mkdir(base_dir)

# 定义训练集和测试集的划分比例
train_ratio = 0.8

# 划分训练集和测试集的索引
train_idx = random_idx[:int(total_num * train_ratio)]
test_idx = random_idx[int(total_num * train_ratio):]
numbers = [train_idx, test_idx]

# 定义子目录和类别
sub_dirs = ['train', 'test']
animals = ['cats', 'dogs']

# 创建文件夹并移动文件
for idx, sub_dir in enumerate(sub_dirs):
    dir = os.path.join(base_dir, sub_dir)
    if not os.path.exists(dir):
        os.mkdir(dir)
    for animal in animals:
        animal_dir = os.path.join(dir, animal)
        if not os.path.exists(animal_dir):
            os.mkdir(animal_dir)
        fnames = [animal[:-1] + '.{}.jpg'.format(i) for i in numbers[idx]]
        for fname in fnames:
            src = os.path.join(original_dataset_dir, fname)
            dst = os.path.join(animal_dir, fname)
            if os.path.exists(src):
                shutil.copyfile(src, dst)
            else:
                print(f"File not found: {src}")

        # # 打印每个类别的图像数量以验证划分结果
        # print(f"{animal_dir} total images: {len(os.listdir(animal_dir))}")

# 输出训练集的图片数量
train_cats_dir = os.path.join(base_dir, 'train', 'cats')
train_dogs_dir = os.path.join(base_dir, 'train', 'dogs')
print(f"{train_cats_dir} total images: {len(os.listdir(train_cats_dir))}")
print(f"{train_dogs_dir} total images: {len(os.listdir(train_dogs_dir))}")

# 输出测试集的图片数量
test_cats_dir = os.path.join(base_dir, 'test', 'cats')
test_dogs_dir = os.path.join(base_dir, 'test', 'dogs')
print(f"{test_cats_dir} total images: {len(os.listdir(test_cats_dir))}")
print(f"{test_dogs_dir} total images: {len(os.listdir(test_dogs_dir))}")
