# Standard imports
import sys
import os
import time
import torch
import pickle

try:
    import pkbar
except Exception:
    os.system("pip install pkbar")
    import pkbar
    print("missing package was installed sucessfully, run the script again")

import numpy as np
from torch.utils.data import DataLoader

repo_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(repo_path)

from bin.___ import ___, get_model_name

torch.manual_seed(0)


class OpenDDDataset(torch.utils.data.Dataset):
    def __init__(self, source_path):
        self.type_dict = {"Car": 1.0}
        if os.path.isdir(source_path):
            self.D = {}
            for root, _, files in os.walk(source_path):
                for file in files:
                    ff = os.path.join(root, file)
                    with open(ff, "rb") as fp:
                        temp_data = pickle.load(fp)
                        for key, values in temp_data.items():
                            if key not in self.D.keys():
                                self.D[key] = []
                            self.D[key] += values
            self.D["sampleID"] = list(
                range(len(self.D["sampleID"]))
            )  # concatenate all splitted data into one

        else:  # single file input
            with open(source_path, "rb") as fp:
                self.D = pickle.load(fp)

    def __len__(self):
        return len(self.D["hist"])

    def __getitem__(self, idx):

        # Get track history 'hist' = ndarray, and future track 'fut' = ndarray
        smpl_id = self.D["sampleID"][idx]
        hist = self.D["hist"][idx]
        fut = self.D["fut"][idx]
        lanes = self.D["lanes"][idx]
        classes = self.D["classes"][idx]
        ojbIDs = self.D["objID"][idx]

        return smpl_id, hist, fut, lanes, classes, ojbIDs

    # Collate function for dataloader
    def collate_fn(self, samples):

        len_in = len(samples[0][1])  # takes the length of hist of the first sample
        len_out = len(samples[0][2])  # takes the length of hist of the first sample
        num_lanes, len_lanes, _ = samples[0][3].shape

        # Initialize history, history lengths, future, output mask, lateral maneuver and longitudinal maneuver batches:
        hist_batch = torch.zeros(len_in, len(samples), 2)
        fut_batch = torch.zeros(len_out, len(samples), 2)
        lanes_batch = torch.zeros(num_lanes, len_lanes, len(samples), 2)
        class_batch = torch.zeros(1, len(samples), 1)
        objID_batch = torch.zeros(1, len(samples), 1)

        smpl_ids = []
        for dataID, (smpl_id, hist, fut, lanes, classes, objIDs) in enumerate(samples):

            # Set up history, future, lateral maneuver and longitudinalhsit maneuver batches:
            hist_batch[0: len(hist), dataID, 0] = torch.from_numpy(hist[:, 0])
            hist_batch[0: len(hist), dataID, 1] = torch.from_numpy(hist[:, 1])
            fut_batch[0: len(fut), dataID, 0] = torch.from_numpy(fut[:, 0])
            fut_batch[0: len(fut), dataID, 1] = torch.from_numpy(fut[:, 1])
            class_batch[0, dataID, 0] = self.type_dict.get(classes, 0.0)
            objID_batch[0, dataID, 0] = int(objIDs)
            for n in range(num_lanes):
                lanes_batch[n, 0:len_lanes, dataID, 0] = torch.from_numpy(
                    lanes[n, :, 0]
                )
                lanes_batch[n, 0:len_lanes, dataID, 1] = torch.from_numpy(
                    lanes[n, :, 1]
                )

            smpl_ids.append(smpl_id)

        return smpl_ids, hist_batch, fut_batch, lanes_batch, class_batch, objID_batch


def MSE(y_pred, y_gt):
    # If GT has not enough timesteps, shrink y_pred
    if y_gt.shape[0] < y_pred.shape[0]:
        y_pred = y_pred[: y_gt.shape[0], :, :]

    muX = y_pred[:, :, 0]
    muY = y_pred[:, :, 1]
    x = y_gt[:, :, 0]
    y = y_gt[:, :, 1]
    mse_det = torch.pow(x - muX, 2) + torch.pow(y - muY, 2)
    count = torch.sum(torch.ones(mse_det.shape))
    mse = torch.sum(mse_det) / count
    return mse, mse_det


def main_train(
    vel=None, modelin=None, learning_rate=0.0001, trainEpochs=8, full_train=False
):

    model_path, model_name = get_model_name(vel)

    # Batch size
    if not full_train:
        batch_size = 8  # for faster computing
        max_epochs = 4
        n_batches = 5
    else:
        batch_size = 64
        max_epochs = np.inf
        n_batches = np.inf

    # Initialize network
    if modelin is None:
        net = ___()
    else:
        net = modelin()

    # Get number of parameters
    pytorch_total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print("Model initialized with {} parameters".format(pytorch_total_params))

    # Initialize data loaders
    tic = time.time()
    train_path = os.path.join(repo_path, "data/training")
    trainings_set = OpenDDDataset(train_path)
    print("Data-Loading for training set took {:.2f} s".format(time.time() - tic))

    tic = time.time()
    validation_path = os.path.join(repo_path, "data/validation")
    validation_set = OpenDDDataset(validation_path)
    print("Data-Loading for validation set took {:.2f} s".format(time.time() - tic))

    tic = time.time()
    tr_dataloader = DataLoader(
        trainings_set,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=trainings_set.collate_fn,
    )
    val_dataloader = DataLoader(
        validation_set,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=validation_set.collate_fn,
    )

    print(
        "Data-Loading for DataLoader initialization took {:.2f} s".format(
            time.time() - tic
        )
    )

    best_val_loss = np.inf

    ####################################################
    ####################################################
    # ---------------- Training with MSE --------------#
    ####################################################
    ####################################################

    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
    print("Training with MSE loss")

    for epoch_num in range(trainEpochs):
        net.train_flag = True
        # Init progbar
        kbar = pkbar.Kbar(
            target=len(tr_dataloader),
            epoch=epoch_num,
            num_epochs=trainEpochs,
        )

        for i, data in enumerate(tr_dataloader):
            # Unpack data
            _, hist, fut, lanes, cl_type, _ = data

            # Feed forward
            if vel is not None:
                velocity = vel(sample=hist, train_flag=True)
                fut_pred = net(hist, lanes, cl_type, velocity)
            else:
                fut_pred = net(hist, lanes, cl_type)

            # Calculate loss
            loss, _ = MSE(fut_pred, fut)

            # Update status bar
            kbar.update(i + 1, values=[("MSE", loss)])

            # Backprop and update weights
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 10)  # Gradient clipping
            optimizer.step()

            if not full_train and i > n_batches:
                break

        # ---------------- Validation ----------------
        net.train_flag = False
        val_loss_list = []

        for i, data in enumerate(val_dataloader):

            # Unpack data
            _, hist, fut, lanes, cl_type, _ = data

            # Feed forward
            if vel is not None:
                velocity = vel(sample=hist, train_flag=True)
                fut_pred = net(hist, lanes, cl_type, velocity)
            else:
                fut_pred = net(hist, lanes, cl_type)

            loss, _ = MSE(fut_pred, fut)

            val_loss_list.append(loss.detach().cpu().numpy())

            if not full_train and i > n_batches:
                break

        val_loss = np.mean(val_loss_list)
        kbar.add(1, values=[("val_loss", val_loss)])

        # Save model if val_loss_improved
        if val_loss < best_val_loss:
            torch.save(net.state_dict(), model_path)
            best_val_loss = val_loss
            print("\nnew best model {} .. saved\n\n".format(model_name))
        else:
            print("\nno model improvements, keep on training\n\n")

        if not full_train and epoch_num > max_epochs:
            break

    return model_path


if __name__ == "__main__":

    # Training
    main_train()
