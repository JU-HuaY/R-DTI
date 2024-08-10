import os
import shutil

# 指定源目录和目标目录
src_dir = "origin"
dst_dir = "protein_3d"

# 创建目标目录(如果不存在)
if not os.path.exists(dst_dir):
    os.makedirs(dst_dir)

# 遍历源目录下的所有文件
for root, dirs, files in os.walk(src_dir):
    for file in files:
        # 检查文件扩展名是否为.pdb
        if file.endswith(".pdb"):
            # 构建源文件路径和目标文件路径
            src_file = os.path.join(root, file)
            dst_file = os.path.join(dst_dir, file)

            # 复制文件
            shutil.copy(src_file, dst_file)
            print(f"Copied {src_file} to {dst_file}")