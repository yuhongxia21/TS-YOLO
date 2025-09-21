import os
import random
import shutil

# 原数据集目录
root_dir = 'datasets/traffic'
# 划分比例
train_ratio = 0.8
valid_ratio = 0.1
test_ratio = 0.1

# 设置随机种子
random.seed(42)

# 拆分后数据集目录
split_dir = 'datasets/traffic'
os.makedirs(os.path.join(split_dir, 'train/images'), exist_ok=True)
os.makedirs(os.path.join(split_dir, 'train/labels'), exist_ok=True)
os.makedirs(os.path.join(split_dir, 'valid/images'), exist_ok=True)
os.makedirs(os.path.join(split_dir, 'valid/labels'), exist_ok=True)
os.makedirs(os.path.join(split_dir, 'test/images'), exist_ok=True)
os.makedirs(os.path.join(split_dir, 'test/labels'), exist_ok=True)


image_files = os.listdir(os.path.join(root_dir, 'images'))
label_files = os.listdir(os.path.join(root_dir, 'labels'))


combined_files = list(zip(image_files, label_files))
random.shuffle(combined_files)
image_files_shuffled, label_files_shuffled = zip(*combined_files)


train_bound = int(train_ratio * len(image_files_shuffled))
valid_bound = int((train_ratio + valid_ratio) * len(image_files_shuffled))


for i, (image_file, label_file) in enumerate(zip(image_files_shuffled, label_files_shuffled)):
    if i < train_bound:
        shutil.copy(os.path.join(root_dir, 'images', image_file), os.path.join(split_dir, 'train/images', image_file))
        shutil.copy(os.path.join(root_dir, 'labels', label_file), os.path.join(split_dir, 'train/labels', label_file))
    elif i < valid_bound:
        shutil.copy(os.path.join(root_dir, 'images', image_file), os.path.join(split_dir, 'valid/images', image_file))
        shutil.copy(os.path.join(root_dir, 'labels', label_file), os.path.join(split_dir, 'valid/labels', label_file))
    else:
        shutil.copy(os.path.join(root_dir, 'images', image_file), os.path.join(split_dir, 'test/images', image_file))
        shutil.copy(os.path.join(root_dir, 'labels', label_file), os.path.join(split_dir, 'test/labels', label_file))

