import pickle

with open('BA-2motif.pkl', 'rb') as f:
    adjs, features, labels = pickle.load(f)

print(adjs.shape)
print(features.shape)
