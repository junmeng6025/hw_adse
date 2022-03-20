"""
Goal of Task 1:
    Understand how easy it is to use data with python and pandas.
"""


import os
import pandas as pd
import matplotlib.pyplot as plt


class UseData:

    def __init__(self, datapath):
        """
        input:
            database (type: str): path to given data file
        """

        # Subtask 1:
        # ToDo: First read all the data from the given file.
        #   Then, extract the data for ax and ay and determine their length.
        # Hints:
        #   - use the parameter "datapath" to make the method generic
        #   - checkout the practice session
        ########################
        #  Start of your code  #
        ########################

        self.ax = pd.read_csv(datapath)['ax']
        self.ay = pd.read_csv(datapath)['ay']
        self.ax_len = len(self.ax)

        ########################
        #   End of your code   #
        ########################

    def avg_acc_x(self):
        """
        output:
            average (type: float)
        """

        # Subtask 2:
        # ToDo: Calculate the average of the acceleration ax.
        # Hint: remember the usage of "self"
        ########################
        #  Start of your code  #
        ########################

        average_x = sum(self.ax) / len(self.ax)
        return average_x

        ########################
        #   End of your code   #
        ########################


if __name__ == "__main__":

    cur_dir = os.path.dirname(os.path.abspath(__file__))
    datafile = os.path.join(cur_dir, 'data.csv')

    usedata = UseData(datafile)

    average = usedata.avg_acc_x()
    print("Calculated average x acceleration: ", average)

    fig = plt.figure()
    plt.step((range(usedata.ax_len)), usedata.ax, label='$a_x$', zorder=0)
    plt.hlines(average, 0, usedata.ax_len, label='$a_x$ $average$', color='red', zorder=1)
    plt.ylabel('Acceleration in m / $s^2$', fontsize=20)
    plt.xlabel('Number of measurements', fontsize=20)
    plt.title('IMU Measurements', fontsize=20)
    plt.legend(loc='best', prop={'size': 18})
    plt.show()
