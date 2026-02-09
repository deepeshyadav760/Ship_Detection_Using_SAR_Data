import os
import xml.etree.ElementTree as ET
from PIL import Image
from shutil import copyfile

CLASSES = ["ship"]

BASE_DIR = "data/raw/BBox_SSDD"
OUT_DIR = "data/yolo"

def convert_bbox(img_size, box):
    w, h = img_size
    xmin, ymin, xmax, ymax = box
    x_center = ((xmin + xmax) / 2) / w
    y_center = ((ymin + ymax) / 2) / h
    bw = (xmax - xmin) / w
    bh = (ymax - ymin) / h
    return x_center, y_center, bw, bh

def process_split(split_name, img_folder, yolo_split):
    ids_file = os.path.join(BASE_DIR, "ImageSets", "Main", f"{split_name}.txt")
    img_dir = os.path.join(BASE_DIR, img_folder)
    ann_dir = os.path.join(BASE_DIR, "Annotations")

    out_img_dir = os.path.join(OUT_DIR, "images", yolo_split)
    out_lbl_dir = os.path.join(OUT_DIR, "labels", yolo_split)

    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(out_lbl_dir, exist_ok=True)

    with open(ids_file) as f:
        image_ids = f.read().strip().split()

    for img_id in image_ids:
        img_path = os.path.join(img_dir, f"{img_id}.jpg")
        ann_path = os.path.join(ann_dir, f"{img_id}.xml")

        if not os.path.exists(img_path):
            continue

        copyfile(img_path, os.path.join(out_img_dir, f"{img_id}.jpg"))

        tree = ET.parse(ann_path)
        root = tree.getroot()

        img = Image.open(img_path)
        w, h = img.size

        with open(os.path.join(out_lbl_dir, f"{img_id}.txt"), "w") as f_out:
            for obj in root.iter("object"):
                cls = obj.find("name").text.lower()
                if cls not in CLASSES:
                    continue

                cls_id = CLASSES.index(cls)
                box = obj.find("bndbox")

                xmin = float(box.find("xmin").text)
                ymin = float(box.find("ymin").text)
                xmax = float(box.find("xmax").text)
                ymax = float(box.find("ymax").text)

                bb = convert_bbox((w, h), (xmin, ymin, xmax, ymax))
                f_out.write(f"{cls_id} {' '.join(map(str, bb))}\n")

if __name__ == "__main__":
    process_split("train", "JPEGImages_train", "train")
    process_split("test", "JPEGImages_test", "val")