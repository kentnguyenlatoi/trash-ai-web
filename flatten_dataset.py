import os, shutil

SRC = "Dataset"
DST = "Dataset_flat"

splits = ["train", "val"]
super_classes = ["biodegradable", "non_biodegradable"]

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def copy_all_images(src_dir, dst_dir):
    ensure_dir(dst_dir)
    for root, _, files in os.walk(src_dir):
        for f in files:
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
                src_path = os.path.join(root, f)
                dst_path = os.path.join(dst_dir, f)

                # tránh trùng tên file
                if os.path.exists(dst_path):
                    base, ext = os.path.splitext(f)
                    i = 1
                    while True:
                        new_name = f"{base}_{i}{ext}"
                        dst_path = os.path.join(dst_dir, new_name)
                        if not os.path.exists(dst_path):
                            break
                        i += 1

                shutil.copy2(src_path, dst_path)

for split in splits:
    for sup in super_classes:
        sup_path = os.path.join(SRC, split, sup)
        if not os.path.isdir(sup_path):
            continue

        # các lớp con: food_waste, leaf_waste, ...
        for sub in os.listdir(sup_path):
            sub_path = os.path.join(sup_path, sub)
            if not os.path.isdir(sub_path):
                continue
            dst_class_dir = os.path.join(DST, split, sub)
            copy_all_images(sub_path, dst_class_dir)

print("✅ Done. New dataset at:", DST)
