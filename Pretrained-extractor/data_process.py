import os
import shutil

src_dir = "origin"
dst_dir = "protein_3d"

if not os.path.exists(dst_dir):
    os.makedirs(dst_dir)

for root, dirs, files in os.walk(src_dir):
    for file in files:
        if file.endswith(".pdb"):
            src_file = os.path.join(root, file)
            dst_file = os.path.join(dst_dir, file)

            shutil.copy(src_file, dst_file)
            print(f"Copied {src_file} to {dst_file}")
