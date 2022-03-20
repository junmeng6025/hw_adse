"""
Goal of Task 4:
    Implement a function that returns the n-th number of the Fibonacci Sequence.
"""


def fibonacci(n):
    """
    input:
        n (type: int)

    output:
        fibonacci_number (type: int)
    """

    # Task:
    # ToDo: Calculate and return the n-th number of the Fibonacci Sequence.
    #       The usage of python packages is not allowed for this task.
    # Hint: F_n = F_n-1 + F_n-2
    ########################
    #  Start of your code  #
    ########################

    fibonacci_number = int(-1)
    fi = [0, 1]
    if n < 2:
        return n
    else:
        for i in range(1, n):
            fi_n = fi[i - 1] + fi[i]
            fi.append(fi_n)
        fibonacci_number = fi[-1]
    return fibonacci_number

    ########################
    #   End of your code   #
    ########################


if __name__ == "__main__":
    assert fibonacci(9) == 34
