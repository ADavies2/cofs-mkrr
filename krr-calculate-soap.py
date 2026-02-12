from dscribe.descriptors import SOAP
from ase.io import read
import numpy as np
from sparse import save_npz, load_npz
import os
import random

# List of functional group and linkers
R = ['NO2', 'CHO', 'H', 'EMEPO', 'DMPE', 'MEPO', 'OProp', 'OH', 'OMe', 'OEEPO', 'EPO', 'NH2', 'SO2H', 'PHEN',\
     'COOH', 'F', 'Cl', 'OCOCH3', 'Br', 'NO', 'SH', 'CH3', 'OEt', 'EEPO', 'CN', 'tBu', 'SO3H', 'CHS', 'I']

Linker = ['Benzene', '1,5-Naph', '2,6-Naph', '1,5-Anth', '2,6-Anth', 'Phenanthrene', 'Biphenyl', 'Terphenyl']

# Compile a list of names from linker-functionlgroup combinations
names = []
for l in Linker:
    for i in R:
        names.append(f'{l}-{i}')

print(len(names))

# Load the .xyz files for all structures in names
structures = []
for i in names:
    structures.append(read(f'all-xyz/{i}.xyz'))

print(len(structures))

species = []
for structure in structures:
    syms = np.unique(structure.get_chemical_symbols())
    species.extend([sym for sym in syms if sym not in species])
species.sort()

# Define the SOAP parameters
soap_params = {'r_cut': 7, 'sigma': 0.26, 'n_max': 11, 'l_max': 11,
               'rbf': 'gto', 'average': 'off', 'crossover': True}

# Build the SOAP object
soap = SOAP(
    species=species,
    periodic=True,
    sigma=soap_params['sigma'],
    r_cut=soap_params['r_cut'],
    n_max=soap_params['n_max'],
    l_max=soap_params['l_max'],
    rbf='gto',
    average='off',
    compression={'mode': 'crossover'},
    sparse=True
)

# Calculate SOAP fingerprints
for i, structure in enumerate(structures):
    soap_filename = os.path.join('npz/soap_'+names[i]+'.npz')
    if os.path.exists(soap_filename):
        continue
    soap_matrix = soap.create(structure)
    save_npz(soap_filename, soap_matrix)

# Get number of features
soap_matrix = soap.create(structures[0])

M = len(names)
N_features = np.shape(soap_matrix)[1]

print(M)
print(N_features)

# Prepare M average SOAPs
avg_soaps_M = np.zeros((M, N_features), dtype=np.float32)
for i in range(M):
    p = os.path.join('npz/soap_'+names[i]+'.npz')
    soap_temp = load_npz(p).todense()
    avg_soaps_M[i,:] = soap_temp.mean(axis=0)

print(np.shape(avg_soaps_M))

# Compute averaged kernel matrix
K = avg_soaps_M.dot(avg_soaps_M.T)
norm = np.sqrt(np.einsum('ii,jj->ij', K, K))

K = K/norm
print(len(K))

np.savetxt('soap-krr-kernel.csv', K.T, delimiter=',')
