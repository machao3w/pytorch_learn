pi=3.1416
def sum(lst):
    total = lst[0]
    for value in lst[1:]:
        total += value
    return total
w = [0,1,2,3]
print(sum(w),pi)
