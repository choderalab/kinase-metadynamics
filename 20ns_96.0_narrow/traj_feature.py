"""
traj_feature.py
This is a tool to featurize kinase conformational changes using kinomodel.features.protein

"""
import featurize
import os
import numpy as np

# print full array without truncation
np.set_printoptions(threshold=np.nan)

(key_res, dihedrals, distances) = featurize.featurize(chain='A', coord='dcd', feature='conf', pdb='5UG9')
print(dihedrals[:,3])
#print(distances[:,4])

