from dscribe.descriptors import SOAP
from ase.io import read
import numpy as np
from sparse import save_npz, load_npz
import os
import random
import pandas as pd

def get_ids(filename):
    node, linker, r = [], [], []

    with open(filename, 'r') as f:
        current_section = None
        for line in f.readlines():
            # Check for ID after Node
            if 'Node' in line.strip():
                current_section = 'node'
                continue
            # Check for ID after Linker
            elif 'Linker' in line.strip():
                current_section = 'linker'
                continue
            # Check for ID after R
            elif 'R' in line.strip():
                current_section = 'r'
                continue
            # If string is empty
            elif line.strip() == "":
                current_section = None

            if current_section == 'node':
                node.append(int(line.strip()))
            if current_section == 'linker':
                linker.append(int(line.strip()))
            if current_section == 'r':
                r.append(int(line.strip()))

    return node, linker, r

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

# Read the ID files for all structures
ids = {}
for l in Linker:
    for j in R:
        node, linker, r = get_ids(f'{l}-FuncGroups/{j}/GFN1/ids.csv')
        ids[f'{l}-{j}'] = {'node': node, 'linker': linker, 'r': r}

# Check the ids dictionary. The length of keys should be equal to the number of structures, and there should be three keys per structure (node, linker, r-group)
print(len(ids.keys()))
print(len(ids['Benzene-NO2'].keys()))

species = []
for structure in structures:
    syms = np.unique(structure.get_chemical_symbols())
    species.extend([sym for sym in syms if sym not in species])
species.sort()

print(species)

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

# Calculate the averaged SOAP per building block with the atom IDs
for j in range(len(structures)):
    node_soap, linker_soap, r_soap = [], [], []
    p = os.path.join('npz/soap_'+names[j]+'.npz')
    soap_temp = load_npz(p).todense()
    for i in range(len(soap_temp)):
        if i in ids[names[j]]['node']:
            node_soap.append(soap_temp[i])
        if i in ids[names[j]]['linker']:
            linker_soap.append(soap_temp[i])
        if i in ids[names[j]]['r']:
            r_soap.append(soap_temp[i])
    node_avgs[j,:] = np.mean(node_soap, axis=0)
    linker_avgs[j,:] = np.mean(linker_soap, axis=0)
    r_avgs[j,:] = np.mean(r_soap, axis=0)

# Compute the Frobenius normed kernel for node, linker, and r-group
K_node = node_avgs.dot(node_avgs.T)
fro_norm_node = np.linalg.norm(K_node, 'fro')
K_node_fro = K_node/fro_norm_node

K_linker = linker_avgs.dot(linker_avgs.T)
fro_norm_linker = np.linalg.norm(K_linker, 'fro')
K_linker_fro = K_linker/fro_norm_linker

K_r = r_avgs.dot(r_avgs.T)
fro_norm_r = np.linalg.norm(K_r, 'fro')
K_r_fro = K_r/fro_norm_r

np.savetxt('soap-mkrr-node-kernel.csv', K_node_fro.T, delimiter=',')
np.savetxt('soap-mkrr-linker-kernel.csv', K_linker_fro.T, delimiter=',')
np.savetxt('soap-mkrr-rgroup-kernel.csv', K_r_fro.T, delimiter=',')
