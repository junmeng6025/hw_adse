"""
Goal of Task 1:
    Implement a function that reverses the word order of a string.
"""


def reverse(sentence):
    """
    input:
        sentence (type: str)

    output:
        reversed_sentence (type: str)
    """

    # Task:
    # ToDo: Return the reversed word order of a string.
    #       The usage of python packages is not allowed for this task.
    ########################
    #  Start of your code  #
    ########################

    words = sentence.split(' ')
    rev_sentence = ' '.join(reversed(words))
    return rev_sentence

    ########################
    #   End of your code   #
    ########################


if __name__ == "__main__":
    assert reverse("this is a test") == "test a is this"
