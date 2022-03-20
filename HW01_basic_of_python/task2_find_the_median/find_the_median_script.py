# definition of the function
def median(list_in):
    n=len(list_in)
    list_sorted=sorted(list_in)
    if n%2:
        median_num=list_sorted[n//2]
    else:
        median_num=(list_sorted[n//2-1]+list_sorted[n//2])/2
    print(median_num)

# test of the function
median([0,9,2,3,1,4,7])
median([0,9,2,3,1,4,7,5])