from ogb.graphproppred import GraphPropPredDataset
# from torch_geometric.data import DataLoader

# Download and process data at './dataset/ogbg_molhiv/'
dataset = GraphPropPredDataset(name = "ogbg-molhiv", root = 'dataset/')

print(len(dataset.labels))