# fork from on https://github.com/yakhyo/YOLO2VOC
# Original code is altered to correct coordinate change errors when transforming from YOLO to VOC and the other way

"""Converter
Convert YOLO <-> VOC
"""
import argparse
import multiprocessing
import os
from xml.etree import ElementTree

from pascal_voc_writer import Writer
from PIL import Image

class config:
    LABEL_DIR = "\\data\\FLIR\\Campus\\Labels\\FLIR0002010" # path to text label files (for YOLO)
    IMAGE_DIR = "\\data\\FLIR\\Campus\\Images\\FLIR0002010"
    XML_DIR   = "\\data\\FLIR\\Campus\\Labels\\FLIR0002010_Pascal_VOC"

    names = [
        "drone"
    ]

# pylint: disable=consider-using-with


def yolo2voc(txt_file: str) -> None:
    """Convert YOLO to VOC
    Args:
        txt_file: str
    """
    w, h = Image.open(os.path.join(config.IMAGE_DIR, f"{txt_file[:-4]}.jpeg")).size
    writer = Writer(f"{txt_file[:-4]}.jpeg", w, h)
    with open(os.path.join(config.LABEL_DIR, txt_file)) as f:
        for line in f.readlines():
            label, x_center, y_center, width, height = line.rstrip().split(" ")
            x_min = int(round(w * max(float(x_center) - float(width) / 2, 1/w)))
            x_max = int(round(w * min(float(x_center) + float(width) / 2, 1)))
            y_min = int(round(h * max(float(y_center) - float(height) / 2, 1/h)))
            y_max = int(round(h * min(float(y_center) + float(height) / 2, 1)))
            writer.addObject(config.names[int(label)], x_min, y_min, x_max, y_max)
    writer.save(os.path.join(config.XML_DIR, f"{txt_file[:-4]}.xml"))


def voc2yolo(xml_file: str) -> None:
    """Convert VOC to YOLO
    Args:
        xml_file: str
    """
    with open(f"{config.XML_DIR}/{xml_file}") as in_file:
        tree = ElementTree.parse(in_file)
        size = tree.getroot().find("size")
        height, width = map(int, [size.find("height").text, size.find("width").text])

    class_exists = False
    for obj in tree.findall("object"):
        name = obj.find("name").text
        if name in config.names:
            class_exists = True

    if class_exists:
        with open(f"{config.LABEL_DIR}/{xml_file[:-4]}.txt", "w") as out_file:
            for obj in tree.findall("object"):
                difficult = obj.find("difficult").text
                if int(difficult) == 1:
                    continue
                xml_box = obj.find("bndbox")

                x_min = float(xml_box.find("xmin").text)
                y_min = float(xml_box.find("ymin").text)

                x_max = float(xml_box.find("xmax").text)
                y_max = float(xml_box.find("ymax").text)

                # according to darknet annotation (original)
                #box_x_center = (x_min + x_max) / 2.0 - 1
                #box_y_center = (y_min + y_max) / 2.0 - 1
                
                # HSU change
                box_x_center = (x_min + x_max) / 2.0 
                box_y_center = (y_min + y_max) / 2.0

                box_w = x_max - x_min
                box_h = y_max - y_min

                box_x = box_x_center * 1.0 / width
                box_w = box_w * 1.0 / width

                box_y = box_y_center * 1.0 / height
                box_h = box_h * 1.0 / height

                b = [box_x, box_y, box_w, box_h]

                cls_id = config.names.index(obj.find("name").text)
                out_file.write(str(cls_id) + " " + " ".join([str(f"{i:.6f}") for i in b]) + "\n")


def voc2yolo_a(xml_file: str) -> None:
    """Convert VOC to YOLO with absolute cordinates
    Args:
        xml_file: str
    """
    with open(f"{config.XML_DIR}/{xml_file}") as in_file:
        tree = ElementTree.parse(in_file)

    class_exists = False
    for obj in tree.findall("object"):
        name = obj.find("name").text
        if name in config.names:
            class_exists = True

    if class_exists:
        with open(f"{config.LABEL_DIR}/{xml_file[:-4]}.txt", "w") as out_file:
            for obj in tree.findall("object"):
                difficult = obj.find("difficult").text
                if int(difficult) == 1:
                    continue
                xml_box = obj.find("bndbox")
                x_min = round(float(xml_box.find("xmin").text))
                y_min = round(float(xml_box.find("ymin").text))

                x_max = round(float(xml_box.find("xmax").text))
                y_max = round(float(xml_box.find("ymax").text))

                b = [x_min, y_min, x_max, y_max]
                cls_id = config.names.index(obj.find("name").text)
                out_file.write(str(cls_id) + " " + " ".join([str(f"{i}") for i in b]) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--yolo2voc", action="store_true", help="YOLO to VOC")
    parser.add_argument("--voc2yolo", action="store_true", default=True, help="VOC to YOLO")
    parser.add_argument("--voc2yolo_a", action="store_true", help="VOC to YOLO absolute")
    args = parser.parse_args()

    if args.yolo2voc:
        print("YOLO to VOC")
        txt_files = [
            name for name in os.listdir(config.LABEL_DIR) if name.endswith(".txt")
        ]
        # check if xml_path exists, if not create it
        if not os.path.exists(config.XML_DIR):
            os.makedirs(config.XML_DIR)
            print(f"xml path was not existent so it was created (XML_DIR={config.XML_DIR})")
        
        with multiprocessing.Pool(os.cpu_count()) as pool:
            pool.map(yolo2voc, txt_files)
        pool.join()

    if args.voc2yolo:
        print("VOC to YOLO")
        xml_files = [
            name for name in os.listdir(config.XML_DIR) if name.endswith(".xml")
        ]
        with multiprocessing.Pool(os.cpu_count()) as pool:
            pool.map(voc2yolo, xml_files)
        pool.join()

    if args.voc2yolo_a:
        xml_files = [
            name for name in os.listdir(config.XML_DIR) if name.endswith(".xml")
        ]
        with multiprocessing.Pool(os.cpu_count()) as pool:
            pool.map(voc2yolo_a, xml_files)
        pool.close()
