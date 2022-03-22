from __future__ import division

from models import *
from utils.utils import *
from utils.datasets import *
from utils.trafos import *
from utils.parse_config import *

import tqdm

import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable


def evaluate(model, path, iou_thres, conf_thres, nms_thres, img_size, batch_size, num_classes):
    model.eval()

    # Get dataloader
    dataset = ListDataset(path, img_size=img_size, multiscale=False, transform=None,
                          num_classes=num_classes)
    dataloader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=batch_size,
        shuffle=False,
        num_workers=1,
        collate_fn=dataset.collate_fn
    )

    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    labels = []
    sample_metrics = []  # List of tuples (TP, confs, pred)
    for batch_i, (_, imgs, targets) in enumerate(tqdm.tqdm(dataloader, desc="Detecting objects")):
        
        if targets is None:
            continue
            
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
    
    if len(sample_metrics) == 0:  # no detections over whole validation set.
        return None
    
    # Concatenate sample statistics
    true_positives, pred_scores, pred_labels = [np.concatenate(x, 0) for x in list(zip(*sample_metrics))]
    precision, recall, AP, f1, ap_class = ap_per_class(true_positives, pred_scores, pred_labels, labels)

    return precision, recall, AP, f1, ap_class


if __name__ == "__main__":
    opt = dict()
    opt["batch_size"] = 1
    opt["model_def"] = "config/yolov3-kitti-tiny_1cls.cfg"
    opt["data_config"] = "config/kitti_1cls.data"
    opt["weights_path"] = "weights/yolov3.pth"
    opt["class_path"] = "../data/kitti/classes.names"
    opt["iou_thres"] = 0.5
    opt["conf_thres"] = 0.5
    opt["nms_thres"] = 0.5
    opt["n_cpu"] = 8
    opt["img_size"] = 352

    for key in opt:
        print(f"{key}: {opt[key]}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_config = parse_data_config(opt["data_config"])
    valid_path = data_config["test"]
    class_names = load_classes(data_config["names"])
    num_classes = int(data_config["classes"])

    # Initiate model
    model = Darknet(opt["model_def"]).to(device)

    # Load checkpoint weights
    model.load_state_dict(torch.load(opt["weights_path"], map_location=torch.device("cpu")))

    print("Compute mAP...")

    precision, recall, AP, f1, ap_class = evaluate(
        model,
        path=valid_path,
        iou_thres=opt["iou_thres"],
        conf_thres=opt["conf_thres"],
        nms_thres=opt["nms_thres"],
        img_size=opt["img_size"],
        batch_size=opt["batch_size"],
        num_classes=num_classes
    )

    print("Average Precisions:")
    for i, c in enumerate(ap_class):
        print(f"+ Class '{c}' ({class_names[c]}) - AP: {AP[i]}")

    print(f"mAP: {AP.mean()}")
