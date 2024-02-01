s = ''
while True:
    item = input()
    if (item == '#'):
        break
    s += item
    s += ','

print(s)
