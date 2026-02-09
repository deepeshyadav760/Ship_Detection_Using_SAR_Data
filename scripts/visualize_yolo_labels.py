import os
import cv2
import matplotlib.pyplot as plt

IMG_DIR = "../data/yolo/images/train"
LBL_DIR = "../data/yolo/labels/train"
OUT_DIR = "results/label_check"

os.makedirs(OUT_DIR, exist_ok=True)

img_files = sorted(os.listdir(IMG_DIR))[:5]  # check first 5 images

for img_name in img_files:
    img_path = os.path.join(IMG_DIR, img_name)
    lbl_path = os.path.join(LBL_DIR, img_name.replace(".jpg", ".txt"))

    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w, _ = img.shape

    if os.path.exists(lbl_path):
        with open(lbl_path) as f:
            for line in f:
                cls, xc, yc, bw, bh = map(float, line.split())

                x1 = int((xc - bw / 2) * w)
                y1 = int((yc - bh / 2) * h)
                x2 = int((xc + bw / 2) * w)
                y2 = int((yc + bh / 2) * h)

                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    plt.figure(figsize=(6, 6))
    plt.imshow(img)
    plt.title(img_name)
    plt.axis("off")

    save_path = os.path.join(OUT_DIR, img_name)
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()

print("Saved label visualization images to:", OUT_DIR)