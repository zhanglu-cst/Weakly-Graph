import matplotlib
matplotlib.use('Agg')
import os
os.environ['MPLBACKEND'] = 'Agg'
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Draw

# from rdkit.Chem.Draw import IPythonConsole #Needed to show molecules
from rdkit.Chem.Draw.MolDrawing import MolDrawing, DrawingOptions #Only needed if modifying defaults
from matplotlib import pyplot as plt
opts = DrawingOptions()
m = Chem.MolFromSmiles('OC1C2C1CC2')

Draw.ShowMol(m,size=(150,150), kekulize=False)

# opts.includeAtomNumbers=True
# opts.bondLineWidth=2.8
# draw = Draw.MolToImage(m, options=opts)
# draw.save('/Users/zeoy/st/drug_development/st_rdcit/img/mol10.jpg')
# draw.show()
# print(type(draw))
# MolDrawing.AddMol()
# plt.figure()
# plt.plot(draw)
# plt.show()