from __future__ import division

from utils.utils import *
from utils.datasets import *
from utils.parse_config import *

import os

import argparse
import tqdm
import shutil
import cv2

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torch.autograd import Variable
import torch.optim as optim


def evaluate(model, path, iou_thres, conf_thres, nms_thres, img_size, batch_size):
    model.eval()

    # Get dataloader
    dataset = ListDataset(path, img_size=img_size, augment=False, multiscale=False)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=1, collate_fn=dataset.collate_fn
    )

    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    labels = []
    sample_metrics = []  # List of tuples (TP, confs, pred)
    for batch_i, (_, imgs, targets) in enumerate(tqdm.tqdm(dataloader, desc="Detecting objects")):
        # Extract labels
        labels += targets[:, 1].tolist()
        # Rescale target
        targets[:, 2:] = xywh2xyxy(targets[:, 2:])
        targets[:, 2:] *= img_size

        imgs = Variable(imgs.type(Tensor), requires_grad=False)

        with torch.no_grad():
            outputs = model(imgs)
            outputs = non_max_suppression(outputs, conf_thres=conf_thres, nms_thres=nms_thres)

        sample_metrics += get_batch_statistics(outputs, targets, iou_threshold=iou_thres)

    # Concatenate sample statistics
    true_positives, pred_scores, pred_labels = [np.concatenate(x, 0) for x in list(zip(*sample_metrics))]
    precision, recall, AP, f1, ap_class = ap_per_class(true_positives, pred_scores, pred_labels, labels)

    return precision, recall, AP, f1, ap_class


def prepareDatas(imagePath, labelPath):
    # print(imagePath, labelPath, imageName)

    # f = open(imageName)  # 返回一个文件对象
    # line = f.readline()  # 调用文件的 readline()方法

    # str1 = ''
    # while line:
    #     # print(line)
    #     str1 += 'data/custom/images/'+line
    #     line = f.readline()
    # f.close()
    # print(str1)

    # f = open('data/custom/valid.txt', 'w')
    # f.write(str1)
    # f.close()
    str1 = ''
    for filename in os.listdir(imagePath):  # listdir的参数是文件夹的路径
        path = os.path.join(imagePath, filename)

        # print(filename)
        str1 += 'data/custom/images/' + filename + '\n'
        shutil.copy(path, 'data/custom/images')

    f = open('data/custom/valid.txt', 'w')
    f.write(str1)
    f.close()

    for filename in os.listdir(labelPath):
        # print(filename)

        path = os.path.join(labelPath, filename)
        f = open(path, encoding='UTF-8')  # 返回一个文件对象
        line = f.readline()  # 调用文件的 readline()方法

        # str1 = imgPath + filename
        str1 = ''
        while line:
            # print
            # line,  # 后面跟 ',' 将忽略换行符
            arr = line.replace('\n', '').split(' ')
            # print(arr)
            # str1 += ' '
            # str1 += arr[2]
            # str1 += ','
            # str1 += arr[3]
            # str1 += ','
            # str1 += arr[4]
            # str1 += ','
            # str1 += arr[5]
            # str1 += ','
            x_Min = int(arr[2])
            y_Min = int(arr[3])
            x_Max = int(arr[4])
            y_Max = int(arr[5])

            boxW = x_Max - x_Min
            boxH = y_Max - y_Min
            x_center = x_Min + boxW / 2
            y_center = y_Min + boxH / 2

            img1 = cv2.imread('data/custom/images/' + filename.replace('.txt', '.jpg'))
            # print(img1)

            sp = img1.shape

            sz1 = sp[0]  # height(rows) of image
            sz2 = sp[1]  # width(colums) of image

            w_scale = boxW / sz2
            h_scale = boxH / sz1

            x_cen_sacle = x_center / sz2
            y_cen_scale = y_center / sz1

            if arr[1] == '带电芯充电宝':
                str1 += '0'
            elif arr[1] == '不带电芯充电宝':
                str1 += '1'
            else:
                # str1 += '2'
                print(filename)
                line = f.readline()

                continue
            str1 += ' '
            str1 += str('%.6f' % x_cen_sacle)
            str1 += ' '
            str1 += str('%.6f' % y_cen_scale)
            str1 += ' '
            str1 += str('%.6f' % w_scale)
            str1 += ' '
            str1 += str('%.6f' % h_scale)
            str1 += '\n'
            line = f.readline()

        f.close()

        f2 = open('data/custom/labels/' + filename.replace('.jpg', '.txt'), 'w')
        f2.write(str1)
        f2.close()
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=8, help="size of each image batch")
    parser.add_argument("--model_def", type=str, default="config/yolov3-custom.cfg",
                        help="path to model definition file")
    parser.add_argument("--data_config", type=str, default="config/custom.data", help="path to data config file")
    parser.add_argument("--weights_path", type=str, default="checkpoints/yolov3_ckpt_90_0.904.pth",
                        help="path to weights file")
    parser.add_argument("--class_path", type=str, default="data/custom/classes.names", help="path to class label file")
    parser.add_argument("--iou_thres", type=float, default=0.5, help="iou threshold required to qualify as detected")
    parser.add_argument("--conf_thres", type=float, default=0.001, help="object confidence threshold")
    parser.add_argument("--nms_thres", type=float, default=0.5, help="iou thresshold for non-maximum suppression")
    parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
    parser.add_argument("--img_path", type=str, default="test/image", help="path of image")
    parser.add_argument("--label_path", type=str, default="test/label", help="path of label")
    opt = parser.parse_args()
    print(opt)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_config = parse_data_config(opt.data_config)

    prepareDatas(opt.img_path, opt.label_path)

    valid_path = data_config["valid"]
    class_names = load_classes(data_config["names"])

    # Initiate model
    # model = Darknet(opt.model_def).to(device)
    # if opt.weights_path.endswith(".weights"):
    #     # Load darknet weights
    #     model.load_darknet_weights(opt.weights_path)
    # else:
    # Load checkpoint weights
    # model.load_state_dict(torch.load(opt.weights_path))

    model = torch.load(opt.weights_path, map_location='cpu')

    print("Compute mAP...")

    precision, recall, AP, f1, ap_class = evaluate(
        model,
        path=valid_path,
        iou_thres=opt.iou_thres,
        conf_thres=opt.conf_thres,
        nms_thres=opt.nms_thres,
        img_size=opt.img_size,
        batch_size=8,
    )

    print("Average Precisions:")
    for i, c in enumerate(ap_class):
        # print(f"+ Class '{c}' ({class_names[c]}) - AP: {AP[i]}")
        print("label:",c)
        print("class_name:",class_names[c])
        print("AP:", AP[i])

    print("mAP: ")
    print({AP.mean()})
