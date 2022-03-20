# definition of the function
def lowest_missing_number(list_in):
    i = 1
    list_sorted=sorted(list_in)
    while i <= list_sorted[-1] + 1:
        if i in list_sorted:
            i += 1
        else:
            break
    return i


# test of the function
print(lowest_missing_number([3, 6, 1, 0, 9, 7]))
print(lowest_missing_number([2, 6, 1, 0, 9, 7]))
print(lowest_missing_number([3, 4, 1, 0, 9, 7]))