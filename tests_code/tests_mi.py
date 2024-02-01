from sklearn.metrics import mutual_info_score

x = [0.1, 0.2, 1, 2]
y = [0.1, 0.2, 1, 2]
s = mutual_info_score(x, y)
print(s)
