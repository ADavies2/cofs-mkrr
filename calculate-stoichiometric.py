import numpy as np
import pandas as pd
from ase.io import read
import os
from scipy.stats.mstats import gmean
from collections import Counter
import json

def calculate_features(FRACTIONS, ALL_ELEMENTS, ELE_PROP):
# Calculate the average and deviation of a COF for a given elemental property
# The fractional amounts of each atom type in the unit cell,
# The column of all elements included in the Ele_Data spreadsheet
# And the column of elemental data to be tabulated for statistical values

    avg = []; dev = []

    # Calculate the mean property
    for tab_types in range(len(ALL_ELEMENTS)):
        for struc_types in FRACTIONS.keys():
            if ALL_ELEMENTS[tab_types] == struc_types:
                avg.append(FRACTIONS[struc_types]*ELE_PROP[tab_types])
    AVG = np.sum(avg)

    # Calculate the absolute deviation of the property
    for tab_types in range(len(ALL_ELEMENTS)):
        for struc_types in FRACTIONS.keys():
            if ALL_ELEMENTS[tab_types] == struc_types:
                dev.append(FRACTIONS[struc_types]*np.abs(ELE_PROP[tab_types]-AVG))
    DEV = np.sum(dev)

    return AVG, DEV

# Generate a list of structure names from all R-groups and Linkers
R = ['NO2', 'CHO', 'H', 'EMEPO', 'DMPE', 'MEPO', 'OProp', 'OH', 'OMe', 'OEEPO', 'EPO', 'NH2', 'SO2H', 'PHEN',\
     'COOH', 'F', 'Cl', 'OCOCH3', 'Br', 'NO', 'SH', 'CH3', 'OEt', 'EEPO', 'CN', 'tBu', 'SO3H', 'CHS', 'I']

Linker = ['Benzene', '1,5-Naph', '2,6-Naph', '1,5-Anth', '2,6-Anth', 'Phenanthrene', 'Biphenyl', 'Terphenyl']

names = []
for l in Linker:
    for i in R:
        names.append(f'{l}-{i}')

# Paths to files
xyz_path = os.path.join('xyz') # list of appended XYZs
tabulated_data_path = os.path.join('ElementalData.xlsx')

# Load the .xyz structures
structures = []
for i in names:
    structures.append(read(f'{xyz_path}/{i}.xyz'))

# Load the tabulated data from the Periodic Table
tabulated_data = pd.read_excel(tabulated_data_path, sheet_name='ElementalProperties', header=0)

FRACTIONS = {}
features = []
for i in range(len(names)):
    # Calculate the fractions of each atom type
    counts = Counter(structures[i].get_chemical_symbols())
    FRACTIONS[names[i]] = {symbol: (count / len(structures[i])) for symbol, count in counts.items()}

    # Calculate the statistical values for each property and append to a feature matrix
    # Each column is the feature values while each row is a distinct COF structured
    data = {}
    data = {'Structure': names[i]}
    for prop in tabulated_data.columns[1:]:
        AVG, DEV = calculate_features(FRACTIONS[names[i]], tabulated_data['AtomType'], tabulated_data[prop])
        new = {f'{prop}-AVG': AVG, f'{prop}-DEV': DEV}
        data = data | new
    features.append(data)

FEATURES = pd.DataFrame(features).set_index('Structure')

# Normalize the features with min-max scaling
FEATURES_NORM = {'Structure': names}

for i in FEATURES.columns[0:len(FEATURES)]:
    x_max = max(FEATURES[i])
    x_min = min(FEATURES[i])
    normalized = []
    for k in FEATURES[i]:
        x_norm = (k-x_min)/(x_max-x_min)
        normalized.append(x_norm)
    new = {f'{i}': normalized}
    FEATURES_NORM = FEATURES_NORM | new

FEATURES_NORM = pd.DataFrame(FEATURES_NORM).set_index('Structure')

FEATURES_NORM.to_csv('stoich-features.csv')
