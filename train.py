import cv2
import os
import random
import numpy as np
from ultralytics import YOLO

from deep_sort.deep_sort.tracker import Tracker as DeepSortTracker
from deep_sort.tools import generate_detections as gdet
from deep_sort.deep_sort import nn_matching
from deep_sort.deep_sort.detection import Detection

print("Imports done.")

from ultralytics import YOLO
import os
import random
import shutil
import xml.etree.ElementTree as ET

IMAGES_DIR = "train_data_raw/train_images"  # Directory containing training images.          
XML_FILE   = "train_data_raw/annotations.xml" #
OUTPUT_DIR = "train_data"
TRAIN_RATIO = 0.8                   

# Class mapping for your dataset (label_name -> class_index).
CLASS_MAP = {
    "car": 0,
    "bus": 1,
    "bicycle": 2,
    "person": 3,
    "motorbike": 4,
    "electric scooter": 5
}


if __name__ == "__main__":

    def convert_to_yolo_bbox(xtl, ytl, xbr, ybr, img_width, img_height):
        """
        Convert CVAT bounding box [xtl, ytl, xbr, ybr]
        to YOLO bounding box [x_center, y_center, width, height] in relative coords.
        """
        x_center = ((xtl + xbr) / 2.0) / img_width
        y_center = ((ytl + ybr) / 2.0) / img_height
        w = (xbr - xtl) / img_width
        h = (ybr - ytl) / img_height
        return x_center, y_center, w, h

    # Create train/val folders for images and labels
    train_img_dir = os.path.join(OUTPUT_DIR, "images", "train")
    val_img_dir   = os.path.join(OUTPUT_DIR, "images", "val")
    train_lbl_dir = os.path.join(OUTPUT_DIR, "labels", "train")
    val_lbl_dir   = os.path.join(OUTPUT_DIR, "labels", "val")

    os.makedirs(train_img_dir, exist_ok=True)
    os.makedirs(val_img_dir, exist_ok=True)
    os.makedirs(train_lbl_dir, exist_ok=True)
    os.makedirs(val_lbl_dir, exist_ok=True)


    tree = ET.parse(XML_FILE)
    root = tree.getroot()

    image_elements = root.findall('image')
    print("Number of images in XML:", len(image_elements))

    # Shuffle
    image_elements = list(image_elements)
    random.shuffle(image_elements)

    train_count = int(len(image_elements) * TRAIN_RATIO)
    train_elements = image_elements[:train_count]
    val_elements   = image_elements[train_count:]
    print(f"Train images: {len(train_elements)}, Val images: {len(val_elements)}")


    def process_images(image_subset, subset_name):
        """
        For each image in the subset, copy the image file to the appropriate train/val folder,
        create a YOLO label file with bounding boxes, and place it in labels/train or labels/val.
        """
        if subset_name == "train":
            img_out_dir = train_img_dir
            lbl_out_dir = train_lbl_dir
        else:
            img_out_dir = val_img_dir
            lbl_out_dir = val_lbl_dir

        for img_elem in image_subset:
            file_name = img_elem.attrib['name']
            width = float(img_elem.attrib['width'])
            height = float(img_elem.attrib['height'])

            src_img_path = os.path.join(IMAGES_DIR, file_name)
            dst_img_path = os.path.join(img_out_dir, file_name)

            if not os.path.exists(src_img_path):
                print(f"Warning: {src_img_path} not found. Skipping.")
                continue

            # Copy image
            shutil.copy2(src_img_path, dst_img_path)

            # Prepare label lines
            boxes = img_elem.findall('box')
            label_lines = []

            for b in boxes:
                label_str = b.attrib['label']
                xtl = float(b.attrib['xtl'])
                ytl = float(b.attrib['ytl'])
                xbr = float(b.attrib['xbr'])
                ybr = float(b.attrib['ybr'])

                # Convert label to class index
                if label_str not in CLASS_MAP:
                    print(f"Warning: Label '{label_str}' not in CLASS_MAP. Skipping.")
                    continue
                class_idx = CLASS_MAP[label_str]

                x_center, y_center, w, h = convert_to_yolo_bbox(
                    xtl, ytl, xbr, ybr, width, height
                )
                label_line = f"{class_idx} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}"
                label_lines.append(label_line)

            txt_file_name = os.path.splitext(file_name)[0] + ".txt"
            txt_out_path = os.path.join(lbl_out_dir, txt_file_name)
            with open(txt_out_path, "w") as f:
                for line in label_lines:
                    f.write(line + "\n")

    process_images(train_elements, "train")
    process_images(val_elements, "val")

    print("Finished processing images and labels.")

    sorted_classes = sorted(CLASS_MAP.items(), key=lambda x: x[1])
    data_yaml_path = os.path.join(OUTPUT_DIR, "dataset.yaml")
    with open(data_yaml_path, "w") as f:
        f.write(f"train: {os.path.abspath(train_img_dir)}\n")
        f.write(f"val: {os.path.abspath(val_img_dir)}\n")
        f.write("names:\n")
        for k, v in sorted_classes:
            f.write(f"  {v}: {k}\n")

    print(f"dataset.yaml created at {data_yaml_path}")

    model = YOLO('yolov8n.pt')
    results = model.train(
        data=data_yaml_path,
        epochs=1000,
        imgsz=640,
        batch=32,
        name="masa_model"
    )

    print("Training completed. Check 'runs/detect/masa_model' for logs and weights.")
