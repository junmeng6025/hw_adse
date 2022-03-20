"""
Goal of Task 3:
    Implement a function that returns the lowest missing number of the list, starting from 0.
"""


def lowest_missing_number(list_in):
    """
    input:
        list_in (type: list): list of integers

    output:
        lowest_number (type: int)
    """

    # Task:
    # ToDo: Calculate and return the lowest missing number of the list, starting from 0.
    #       The usage of python packages is not allowed for this task.
    # Hint: e.g. L = [3, 6, 1, 0, 9, 7, 2] the function should return 4
    ########################
    #  Start of your code  #
    ########################

    lowest_num = -1
    i = 0
    while lowest_num == -1:
        if i in list_in:
            i += 1
        else:
            lowest_num = i
    return lowest_num

    ########################
    #   End of your code   #
    ########################


if __name__ == "__main__":
    assert lowest_missing_number([3, 6, 1, 0, 9, 7]) == 2
