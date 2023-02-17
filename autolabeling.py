import os
import xml.etree.ElementTree as ET
from PIL import Image
import torch
import argparse

def _pretty_print(current, parent=None, index=-1, depth=0):
    for i, node in enumerate(current):
        _pretty_print(node, current, i, depth + 1)
    if parent is not None:
        if index == 0:
            parent.text = '\n' + ('\t' * depth)
        else:
            parent[index - 1].tail = '\n' + ('\t' * depth)
        if index == len(parent) - 1:
            current.tail = '\n' + ('\t' * (depth - 1))

class AutoLabeling:

    def __init__(self, images_path:str, save_label_path:str, model_path:str, conf_thres:float):
        self.images_path = images_path
        self.save_label_path = save_label_path
        self.model_path = model_path
        self.conf_thres = conf_thres

    def auto_labeling(self) -> None:
        # 레이블링할 이미지 폴더의 경로와 xml 파일 저장할 폴더의 경로
        # images_path = "data/images/test"
        images_to_label = os.listdir(self.images_path)
        # save_label_path = "save_annotation1"

        print("length of dataset:", len(images_to_label))
        # image_practice = ["1-경기-38-거-3906.jpg"]

        # 학습시킨 모델 불러오기
        # model = torch.hub.load('.', 'custom', path='runs/train/trial/weights/best.pt', source='local')
        model = torch.hub.load('.', 'custom', path=self.model_path, source='local')

        wrong_row = 0

        for image in images_to_label:
            try:
                # 불러온 모델로 이미지 detect 수행
                detect_output = model(self.images_path + "/" + image)
                # detect한 바운딩 박스의 좌표
                print(image, "\n", detect_output.pandas().xyxy[0])
                # print(type(detect_output), type(detect_output.pandas()), type(detect_output.pandas().xyxy), type(detect_output.pandas().xyxy[0]))
                # print(len(detect_output.pandas().xyxy[0].columns))
                if len(detect_output.pandas().xyxy[0]) != 1:
                    print("wrong row number")
                    wrong_row += 1
                # confidence threshold 이상인 것만 포함
                for index in range(len(detect_output.pandas().xyxy[0]['confidence'])):
                    if detect_output.pandas().xyxy[0]['confidence'][index] > self.conf_thres:
                        output_bndbox = [detect_output.pandas().xyxy[0]['xmin'][0],
                                         detect_output.pandas().xyxy[0]['ymin'][0],
                                         detect_output.pandas().xyxy[0]['xmax'][0],
                                         detect_output.pandas().xyxy[0]['ymax'][0]]
                # xml 파일에 작성할 태그
                annotation = ET.Element("annotation")
                ET.SubElement(annotation, "filename").text = image
                size = ET.SubElement(annotation, "size")
                image_open = Image.open(self.images_path + "/" + image)
                ET.SubElement(size, "width").text = str(image_open.size[0])
                ET.SubElement(size, "height").text = str(image_open.size[1])
                ET.SubElement(size, "depth").text = "3"
                object_tag = ET.SubElement(annotation, "object")
                ET.SubElement(object_tag, "name").text = "Plate"
                ET.SubElement(object_tag, "pose").text = "Unspecified"
                ET.SubElement(object_tag, "truncated").text = "0"
                ET.SubElement(object_tag, "difficult").text = "0"
                bndbox = ET.SubElement(object_tag, "bndbox")
                ET.SubElement(bndbox, "xmin").text = str(int(output_bndbox[0]))
                ET.SubElement(bndbox, "ymin").text = str(int(output_bndbox[1]))
                ET.SubElement(bndbox, "xmax").text = str(int(output_bndbox[2]))
                ET.SubElement(bndbox, "ymax").text = str(int(output_bndbox[3]))
                # xml 파일에 작성
                _pretty_print(annotation)
                tree = ET.ElementTree(annotation)
                # xml 파일명
                annotated_filename = os.path.splitext(image)[0] + '.xml'
                with open(self.save_label_path + '/' + annotated_filename, "wb") as file:
                    tree.write(file, encoding='utf-8', xml_declaration=True)
            except Exception as e:
                print("error:", e)
        print("wrong row number:", wrong_row)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Auto Labeling")
    parser.add_argument('--images_path', type=str, help='The path of the image datasets to label')
    parser.add_argument('--save_label_path', type=str, help='The path to save the label xml files')
    parser.add_argument('--model_path', type=str, help='The path of the model to use in detection')
    parser.add_argument('--conf_thres', type=float, help='The confidence threshold used in detection', default=0.5)

    args = parser.parse_args()

    AutoLabeling(args.images_path, args.save_label_path, args.model_path, args.conf_thres).auto_labeling()
