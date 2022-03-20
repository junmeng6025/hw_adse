# definition of the function
def fibonacci(n):
    # return the n-th number of the febonacci sequence
    fi=[0,1]
    if n<2:
        return n
    else:
        for i in range(1,n):
            fi_n=fi[i-1]+fi[i]
            fi.append(fi_n)
        return(fi[-1])
# test of the function
print(fibonacci(3))
print(fibonacci(9))