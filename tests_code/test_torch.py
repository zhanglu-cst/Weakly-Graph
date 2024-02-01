

x = [[1,2,3],
     [4,5,6],
     [7,8,9]]
aaa = list(map(list, zip(*x)))

print(len(aaa))
print(aaa[0])