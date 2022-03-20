"""
Goal of Task 3:
    Implement the Decoder-GRU (type of RNN) and tune the hyper parameters to optimize the resulting loss.
"""

import torch
import torch.nn as nn
import numpy as np
from bin.eval_the_net import eval_the_net
from train import main_train


class YourNet(nn.Module):

    # Initialization
    def __init__(self):
        super(YourNet, self).__init__()

        # Model config
        model_config = {
            "dyn_embedding_size": 16,
            "input_embedding_size": 32,
            "encoder_size": 16,
            "num_layers_enc": 4,
            "decoder_size": 32,
            "num_layers_dec": 4,
            "batch_size": 64,
            "out_length": 50,  # do not modify the output length !!
        }

        # Sizes of network layers
        self.encoder_size = model_config["encoder_size"]
        self.decoder_size = model_config["decoder_size"]
        self.out_length = model_config["out_length"]
        self.dyn_embedding_size = model_config["dyn_embedding_size"]
        self.input_embedding_size = model_config["input_embedding_size"]
        self.num_layers_enc = model_config["num_layers_enc"]
        self.num_layers_dec = model_config["num_layers_dec"]

        # Input embedding layer
        self.ip_emb = torch.nn.Linear(2, self.input_embedding_size)
        self.lane_emb1 = torch.nn.Linear(2, self.input_embedding_size)
        self.lane_emb2 = torch.nn.Linear(2, self.input_embedding_size)
        self.lane_emb3 = torch.nn.Linear(2, self.input_embedding_size)

        # Encoder RNN
        self.enc_rnn_hist = torch.nn.GRU(
            self.input_embedding_size,
            self.encoder_size,
            self.num_layers_enc,
        )
        self.enc_rnn_lanes1 = torch.nn.GRU(
            self.input_embedding_size,
            self.encoder_size,
            self.num_layers_enc,
        )
        self.enc_rnn_lanes2 = torch.nn.GRU(
            self.input_embedding_size,
            self.encoder_size,
            self.num_layers_enc,
        )
        self.enc_rnn_lanes3 = torch.nn.GRU(
            self.input_embedding_size,
            self.encoder_size,
            self.num_layers_enc,
        )

        # Vehicle dynamics embedding
        self.dyn_emb = torch.nn.Linear(self.encoder_size, self.dyn_embedding_size)

        # Subtask 1:
        #   Write a Decoder RNN. We will use the GRU-Cell, which is State of the Art for Recurrent Neural Networks.
        # Hints:
        #   - For further details, check the doc:
        #   https://pytorch.org/docs/stable/generated/torch.nn.GRU.html#torch.nn.GRU
        #   - All parameters are defined in the model_config (see line 20 above)
        # model_config = {
        #     "dyn_embedding_size": 16,
        #     "input_embedding_size": 32,
        #     "encoder_size": 16,
        #     "num_layers_enc": 4,
        #     "decoder_size": 32,
        #     "num_layers_dec": 4,
        #     "batch_size": 64,
        #     "out_length": 50,  # do not modify the output length !!
        # }
        # ToDo: Implement a GRU, with the following specification:
        #   - input_size = 4 * self.dyn_embedding_size (input of 3 * lanes and 1 * object_history)
        #   - hidden_size = self.decoder_size (defined in model_config)
        #   - num_layers = self.num_layers_dec (defined in model_config)
        ########################
        #  Start of your code  #
        ########################
        input_size = 4 * self.dyn_embedding_size
        hidden_size = self.decoder_size
        num_layers = self.num_layers_dec

        self.dec_rnn = torch.nn.GRU(
            input_size,
            hidden_size,
            num_layers,
        )

        ########################
        #   End of your code   #
        ########################

        # Output layers:
        self.output_layer = torch.nn.Linear(self.decoder_size, 5)

        # Activations:
        self.relu = torch.nn.ReLU()
        self.tanh = torch.nn.Tanh()
        self.softmax = torch.nn.Softmax(dim=1)

    # Forward Pass
    def forward(self, hist, lanes, cl_type=None):
        """
        MODIFICATION OF THIS FUNCTION OWN YOUR OWN RISK - NOT RECOMMENDED (NOT PART OF THE HOMEWORK)
        """

        # Forward pass hist:
        _, hist_enc = self.enc_rnn_hist(self.relu(self.ip_emb(hist)))
        # Forward pass lane 1:
        _, l1_enc = self.enc_rnn_lanes1(self.relu(self.lane_emb1(lanes[0, :, :, :])))
        # Forward pass lane 2:
        _, l2_enc = self.enc_rnn_lanes2(self.relu(self.lane_emb2(lanes[1, :, :, :])))
        # Forward pass lanes 3:
        _, l3_enc = self.enc_rnn_lanes3(self.relu(self.lane_emb3(lanes[2, :, :, :])))

        # instead of tensor.view() strip first dimension
        hist_enc = hist_enc[-1, :, :]
        l1_enc = l1_enc[-1, :, :]
        l2_enc = l2_enc[-1, :, :]
        l3_enc = l3_enc[-1, :, :]
        hist_enc = self.relu(self.dyn_emb(hist_enc))
        l1_enc = self.relu(self.dyn_emb(l1_enc))
        l2_enc = self.relu(self.dyn_emb(l2_enc))
        l3_enc = self.relu(self.dyn_emb(l3_enc))

        # Concatenate encodings:
        enc = torch.cat((hist_enc, l1_enc, l2_enc, l3_enc), 1)
        fut_pred = self.decode(enc)
        return fut_pred

    def decode(self, enc):
        """
        MODIFICATION OF THIS FUNCTION OWN YOUR OWN RISK - NOT RECOMMENDED (NOT PART OF THE HOMEWORK)
        """

        enc = enc.repeat(self.out_length, 1, 1)
        h_dec, _ = self.dec_rnn(enc)
        h_dec = h_dec.permute(1, 0, 2)
        fut_pred = self.output_layer(h_dec)
        fut_pred = fut_pred.permute(1, 0, 2)
        fut_pred[:, :, 2:3] = torch.exp(fut_pred[:, :, 2:3])
        fut_pred[:, :, 3:4] = torch.exp(fut_pred[:, :, 3:4])
        fut_pred[:, :, 4:5] = torch.nn.Tanh()(fut_pred[:, :, 4:5])
        return fut_pred


def compare_metrics(your_model, best_final_rmse, worst_final_rmse, max_points=3):
    """
    This will be the exact same evaluation like after the submission.
    So you can check how good your net performs.
    Keep attention to output the correct dimensions, otherwise errors are thrown.
    """

    your_final_rmse = your_model["rmse"][-1]

    relative_value = (worst_final_rmse - your_final_rmse) / (
        worst_final_rmse - best_final_rmse
    )

    print("Best Model's final RMSE:\t{:.2f} m".format(best_final_rmse))
    print("Worst Model's final RMSE:\t{:.2f} m".format(worst_final_rmse))
    print("Your final RMSE:\t\t{:.2f} m".format(your_final_rmse))
    result = np.ceil(relative_value * max_points)

    return int(result)


if __name__ == "__main__":

    # Subtask 2:
    # ToDo: Modify learning rate and number of epochs to get the best result out of the training.
    ########################
    #  Start of your code  #
    ########################

    lr = 1e-2
    n_epochs = 100

    ########################
    #   End of your code   #
    ########################

    # You will be evaluated on a random test set,
    # but you can check your performance on your own data-set before.
    # See practice script for how to create a test-set.

    # Subtask 3:
    # ToDo: Successively execute the following training steps by umcommenting/commenting the respective code blocks.
    # 1. Start training in debugging mode to check if everything works fine
    # net_path = main_train(modelin=YourNet, learning_rate=lr, trainEpochs=n_epochs)

    # 2. After debugging, train your net on the whole dataset:
    net_path = main_train(modelin=YourNet, learning_rate=lr, trainEpochs=n_epochs, full_train=True)

    # 3. Next, uncomment the following lines and check your net against the 'best model'
    your_model = eval_the_net(model_type=YourNet, net_path=net_path, visualization=False)
    best_mean_rmse, best_final_rmse = eval_the_net(model_type="best_model", visualization=False)
    worst_mean_rmse, worst_final_rmse = eval_the_net(model_type="no_training")
    result = compare_metrics(your_model, best_final_rmse, worst_final_rmse)
    print("\nYour Score is {:d} (Maximum is 3)".format(result))
