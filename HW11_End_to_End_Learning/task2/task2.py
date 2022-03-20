# coding:utf-8

"""
Goal of Task 2:
    Get an overview of how an imitator policy can be implemented. The DAVE-2 architecture from NVIDIA is used for
    the imitator policy.
"""


import torch
from torch import nn


class Policy(nn.Module):
    def __init__(self, a_dim):
        super(Policy, self).__init__()
        """
        This function initializes the different layers of the architecture.
            
        input:
            a_dim (type: int): dimension of the output
        """

        # Subtask 1:
        # ToDo: Implement the convolutional layers of the DAVE-2 architecture.
        # Hint: Assume that we already have normalized inputs.
        ########################
        #  Start of your code  #
        ########################

        self.cnn1 = nn.Conv2d(3, 24, kernel_size=(5, 5), stride=(2, 2))
        self.cnn2 = nn.Conv2d(24, 36, kernel_size=(5, 5), stride=(2, 2))
        self.cnn3 = nn.Conv2d(36, 48, kernel_size=(5, 5), stride=(2, 2))
        self.cnn4 = nn.Conv2d(48, 64, kernel_size=(3, 3))
        self.cnn5 = nn.Conv2d(64, 64, kernel_size=(3, 3))

        ########################
        #   End of your code   #
        ########################

        # Subtask 2:
        # ToDo: Implement the fully connected layers of the DAVE-2 architecture.
        # Hints:
        #   - Don't care about flattening, it is already implemented in the forward method.
        #   - There is a small mistake in the architecture drawing from NVIDIA. Please use 1152 input neurons
        #     instead of 1164, as flatting 64@1x18 --> 1152 neurons
        #   - Number of output neurons necessary for vehicle control is a parameter.
        ########################
        #  Start of your code  #
        ########################

        self.fcn1 = nn.Linear(1152, 100)
        self.fcn2 = nn.Linear(100, 50)
        self.fcn3 = nn.Linear(50, 10)
        self.last_layer = nn.Linear(10, a_dim)

        ########################
        #   End of your code   #
        ########################

        # Subtask 3:
        # ToDo: Define the last activation function.
        # Hint: All of our vehicle actions are defined between [0, 1].
        ########################
        #  Start of your code  #
        ########################

        self.last_activation = nn.Sigmoid()

        ########################
        #   End of your code   #
        ########################

        self.relu = nn.ReLU(inplace=True)
        self.loss = nn.MSELoss(reduction='sum')
        print(self)

    def forward(self, x):

        x_cnn1 = self.relu(self.cnn1(x))
        x_cnn2 = self.relu(self.cnn2(x_cnn1))
        x_cnn3 = self.relu(self.cnn3(x_cnn2))
        x_cnn4 = self.relu(self.cnn4(x_cnn3))
        x_cnn5 = self.relu(self.cnn5(x_cnn4))
        x_flatten = x_cnn5.view(x_cnn5.shape[0], -1)
        x_fcn1 = self.relu(self.fcn1(x_flatten))
        x_fcn2 = self.relu(self.fcn2(x_fcn1))
        x_fcn3 = self.relu(self.fcn3(x_fcn2))
        vehicle_control = self.last_activation(self.last_layer(x_fcn3))

        return vehicle_control

    def criterion(self, a_imitator_, a_expert_):
        loss = self.loss(a_imitator_, a_expert_)
        return loss


if __name__ == '__main__':

    # Hyperparameters for testing and playing
    BATCH_SIZE = 1
    A_DIM = 2

    # Initialize policy with a_dim
    imitator_policy = Policy(a_dim=A_DIM)

    # Generate random test tensors for action and state
    a_expert = torch.rand(size=(BATCH_SIZE, 2))
    state_testing = torch.rand(size=(BATCH_SIZE, 3, 66, 200))

    # Calculate predicted action
    a_imitator = imitator_policy(state_testing)

    print('Predicted action: ', a_imitator)

    # Calculate MSE Loss
    loss = imitator_policy.criterion(a_imitator, a_expert)
    print('Calculated loss: ', loss)
