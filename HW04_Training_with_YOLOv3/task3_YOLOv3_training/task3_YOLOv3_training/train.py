from __future__ import division

import datetime
import sys
import time

import tqdm
from torch.autograd import Variable
from torch.utils.data import DataLoader

from evaluate import evaluate
from models import *
from utils.augmentations import *
from utils.datasets import *
from utils.parse_config import *
from utils.utils import *

if __name__ == "__main__":
    opt = dict()
    opt["epochs"] = 100
    opt["batch_size"] = 8
    opt["gradient_accumulations"] = 2
    opt["model_def"] = "config/yolov3-kitti-tiny_1cls.cfg"
    opt["data_config"] = "config/kitti_1cls.data"
    opt["n_cpu"] = 8
    opt["img_size"] = 352
    opt["evaluation_interval"] = 1
    opt["compute_map"] = False
    opt["multiscale_training"] = True
    opt["verbose"] = False

    for key in opt:
        print(f"{key}: {opt[key]}")

    # Check if GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Get data configuration
    data_config = parse_data_config(opt["data_config"])
    train_path = data_config["train"]
    valid_path = data_config["test"]
    try:
        class_names = load_classes(data_config["names"])
    except Exception as e:
        print(e)
        print("Download kitti dataset from https://syncandshare.lrz.de/getlink/fiLHHZc1Atd2Voxv4qSqnvAU/kitti "
              "and keep its structure")
        sys.exit()
    num_classes = int(data_config["classes"])

    # Quit training if dataset folder is not in correct place
    try:
        assert len(os.listdir("../kitti/images/train/")) == 188, (
            "Download kitti dataset from https://syncandshare.lrz.de/getlink/fiLHHZc1Atd2Voxv4qSqnvAU/kitti "
            "and keep its structure")
        assert len(os.listdir("../kitti/images/test/")) == 34, (
            "Download kitti dataset from https://syncandshare.lrz.de/getlink/fiLHHZc1Atd2Voxv4qSqnvAU/kitti "
            "and keep its structure")
        assert len(os.listdir("../kitti/labels/train/")) == 188, (
            "Download kitti dataset from https://syncandshare.lrz.de/getlink/fiLHHZc1Atd2Voxv4qSqnvAU/kitti "
            "and keep its structure")
        assert len(os.listdir("../kitti/labels/test/")) == 34, (
            "Download kitti dataset from https://syncandshare.lrz.de/getlink/fiLHHZc1Atd2Voxv4qSqnvAU/kitti "
            "and keep its structure")
    except Exception as e:
        print(e)
        print("Download kitti dataset from https://syncandshare.lrz.de/getlink/fiLHHZc1Atd2Voxv4qSqnvAU/kitti "
              "and keep its structure")
        sys.exit()

    # Quit training if train.txt are not in dataset folder
    try:
        open(train_path)
    except Exception as e:
        print(e)
        print("No train.txt found in kitti directory. Place your created train.txt from task1 in the kitti folder!")
        sys.exit()

    # Initiate model
    model = Darknet(opt["model_def"], img_size=opt["img_size"]).to(device)
    model.apply(weights_init_normal)

    # Get dataloader
    dataset = ListDataset(train_path, multiscale=opt["multiscale_training"], img_size=opt["img_size"],
                          transform=AUGMENTATION_TRANSFORMS, num_classes=num_classes)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=opt["batch_size"],
        shuffle=True,
        num_workers=opt["n_cpu"],
        pin_memory=True,
        collate_fn=dataset.collate_fn,
    )

    # Define optimizer
    optimizer = torch.optim.Adam(model.parameters())

    # Define metrics
    metrics = [
        "grid_size",
        "loss",
        "x",
        "y",
        "w",
        "h",
        "conf",
        "cls",
        "cls_acc",
        "recall50",
        "recall75",
        "precision",
        "conf_obj",
        "conf_noobj",
    ]

    # Training loop
    for epoch in range(opt["epochs"]):
        model.train()
        start_time = time.time()

        # select smaller batches and train batch-by-batch
        for batch_i, (img_path, imgs, targets) in enumerate(tqdm.tqdm(dataloader,
                                                                      desc=f"Training Epoch {epoch}/{opt['epochs']}")):
            batches_done = len(dataloader) * epoch + batch_i

            imgs = Variable(imgs.to(device))
            targets = Variable(targets.to(device), requires_grad=False)

            # Inference and backpropagation
            loss, outputs = model(imgs, targets)
            loss.backward()

            if batches_done % opt["gradient_accumulations"] == 0:
                # Accumulates gradient before each step
                optimizer.step()
                optimizer.zero_grad()

            # ----------------
            #   Log progress
            # ----------------

            log_str = "\n---- [Epoch %d/%d, Batch %d/%d] ----\n" % (epoch, opt["epochs"], batch_i, len(dataloader))

            metric_table = [["Metrics", *[f"YOLO Layer {i}" for i in range(len(model.yolo_layers))]]]

            # Log metrics at each YOLO layer
            for i, metric in enumerate(metrics):
                formats = {m: "%.6f" for m in metrics}
                formats["grid_size"] = "%2d"
                formats["cls_acc"] = "%.2f%%"
                row_metrics = [formats[metric] % yolo.metrics.get(metric, 0) for yolo in model.yolo_layers]
                metric_table += [[metric, *row_metrics]]

            log_str += f"\nTotal loss {to_cpu(loss).item()}"

            # Tensorboard logging
            tensorboard_log = []
            for j, yolo in enumerate(model.yolo_layers):
                for name, metric in yolo.metrics.items():
                    if name != "grid_size":
                        tensorboard_log += [(f"train/{name}_{j + 1}", metric)]
            tensorboard_log += [("train/loss", to_cpu(loss).item())]

            # Determine approximate time left for epoch
            epoch_batches_left = len(dataloader) - (batch_i + 1)
            time_left = datetime.timedelta(seconds=epoch_batches_left * (time.time() - start_time) / (batch_i + 1))
            log_str += f"\n---- ETA {time_left}"

            if opt["verbose"]:
                print(log_str)

            model.seen += imgs.size(0)

        # Evaluate model after finishing an epoch
        if epoch % opt["evaluation_interval"] == 0:
            print("---- Evaluating Model ----")
            # Evaluate the model on the validation set
            metrics_output = evaluate(
                model,
                path=valid_path,
                iou_thres=0.5,
                conf_thres=0.5,
                nms_thres=0.5,
                img_size=opt["img_size"],
                batch_size=8,
                num_classes=num_classes
            )

            if metrics_output is not None:
                precision, recall, AP, f1, ap_class = metrics_output
                evaluation_metrics = [
                    ("validation/precision", precision.mean()),
                    ("validation/recall", recall.mean()),
                    ("validation/mAP", AP.mean()),
                    ("validation/f1", f1.mean()),
                ]

                print(f"---- AP class Car: {round(AP.mean(), 2)}")
            else:
                print("---- AP not measured (no detections found by model)")

            torch.save(model.state_dict(), "yolov3.pth")
            print(f"Epoch {epoch} finished! Saving model at yolov3.pth\n\n\n")
