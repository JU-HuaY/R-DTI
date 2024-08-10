# import os
# import shutil
# import random
#
# # 原始数据集路径
# original_dataset_path = "prot_t5"
#
# # 目标文件夹路径
# dataset_1_path = "train"
# dataset_2_path = "validation"
#
# # 确保目标文件夹存在
# os.makedirs(dataset_1_path, exist_ok=True)
# os.makedirs(dataset_2_path, exist_ok=True)
#
# # 获取原始数据集中所有文件的列表
# files = os.listdir(original_dataset_path)
#
# # 随机打乱文件列表，确保分割的随机性
# random.shuffle(files)
#
# # 按照一定比例（例如50/50）将文件分配到两个新文件夹
# split_ratio = 0.9  # 可以根据需要调整
# split_index = int(len(files) * split_ratio)
#
# # 分割文件并复制到新文件夹
# for i, file in enumerate(files):
#     source_file_path = os.path.join(original_dataset_path, file)
#     if i < split_index:
#         destination_path = dataset_1_path
#     else:
#         destination_path = dataset_2_path
#
#     destination_file_path = os.path.join(destination_path, file)
#     shutil.copy2(source_file_path, destination_file_path)
#
# print("Files have been successfully split into two folders.")

import os

def count_files_in_directory(directory_path):
    """
    计算指定目录下（包括子目录）的文件数量。

    参数:
    directory_path (str): 待统计文件数量的目录路径。

    返回:
    int: 目录中文件的总数。
    """
    count = 0
    for root, dirs, files in os.walk(directory_path):
        count += len(files)  # 只累加当前目录下的文件数量，忽略子目录
    return count

# 使用示例
directory_to_count = "validation"
file_count = count_files_in_directory(directory_to_count)
print(f"The directory '{directory_to_count}' contains {file_count} files.")

# import os
# import shutil
# import random
#
# # 源文件夹路径
# source_dir = "validation"
#
# # 目标文件夹路径
# target_dir = "train"
#
# # 确保目标文件夹存在
# os.makedirs(target_dir, exist_ok=True)
#
# # 获取源文件夹中所有文件的列表
# files = [f for f in os.listdir(source_dir) if os.path.isfile(os.path.join(source_dir, f))]
#
# # 确保源文件夹中有足够的文件
# if len(files) < 7:
#     print("Error: Source directory does not contain at least three files.")
# else:
#     # 随机选择三个文件
#     selected_files = random.sample(files, 7)
#
#     # 移动选定的文件到目标文件夹
#     for file in selected_files:
#         source_file_path = os.path.join(source_dir, file)
#         target_file_path = os.path.join(target_dir, file)
#
#         shutil.move(source_file_path, target_file_path)
#
#     print("Seven files have been randomly moved to the target directory.")