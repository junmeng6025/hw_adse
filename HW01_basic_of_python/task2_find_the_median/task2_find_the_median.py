"""
Goal of Task 2:
    Implement a function that returns the median of a list.
"""


def median(list_in):
    """
    input:
        list_in (type: list): list of integers

    output:
        median (type: int or float)
    """

    # Task:
    # ToDo: Calculate and return the median of the given list.
    #       The usage of python packages is not allowed for this task.
    # Hints:
    #     - list might be unsorted
    #     - for an odd number of items the median is part of the list,
    #       e.g. for [0, 9, 2, 3, 1, 4, 7] the function should return 3.
    #     - for an even number of items the median is not part of the list but the mean of two middle number,
    #       e.g. for [0, 9, 2, 3, 1, 4, 7, 5] the function should return 3.5.
    ########################
    #  Start of your code  #
    ########################

    n = len(list_in)
    list_sorted = sorted(list_in)
    if n % 2:
        median_num = list_sorted[n // 2]
    else:
        median_num = (list_sorted[n // 2 - 1] + list_sorted[n // 2]) / 2
    return median_num

    ########################
    #   End of your code   #
    ########################


if __name__ == "__main__":
    # Example with even number of items
    assert median([0, 9, 2, 3, 1, 4, 7]) == 3

    # Example with odd number of items
    assert median([0, 9, 2, 3, 1, 4, 7, 5]) == 3.5
