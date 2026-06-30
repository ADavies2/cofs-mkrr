import numpy as np
import pandas as pd
import os
import random
from sklearn.model_selection import ShuffleSplit, cross_val_predict, KFold
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import pickle

from himalaya.kernel_ridge import WeightedKernelRidge

# Define paths to data
node_kernel_path = 'soap-mkrr-node-kernel.csv'
linker_kernel_path = 'soap-mkrr-linker-kernel.csv'
r_kernel_path = 'soap-mkrr-rgroup-kernel.csv'

y_path = os.path.join('all-bandgaps.csv')

# Read in data
K_node = pd.read_csv(node_kernel_path, header=None, delimiter=',').to_numpy()
K_linker = pd.read_csv(linker_kernel_path, header=None, delimiter=',').to_numpy()
K_fg = pd.read_csv(r_kernel_path, header=None, delimiter=',').to_numpy()

refcodes = pd.read_csv(y_path, header=None)[0].values
y = pd.read_csv(y_path, header=None)[1].values

print(np.shape(K_node))
print(np.shape(K_linker))
print(np.shape(K_fg))

print(len(refcodes))
print(len(y))

# Define hyperparameters
test_size = 0.3
seeds = [2873,317,3846,446,3555]
deltas = [22.20461977 20.65171198 19.45848805]
xi = 3

# Train the models for each seed
results, models = {}, {}
for i in seeds:
    # Set-up the training-testing split
    splitter = ShuffleSplit(n_splits=1, test_size=test_size, random_state=i)
    train_indices, test_indices = next(splitter.split(y))
    
    y_train = y[train_indices]
    y_test = y[test_indices]
    
    refcodes_train = refcodes[train_indices]
    refcodes_test = refcodes[test_indices]
    
    K_ntrain = K_node[train_indices,:][:,train_indices]
    K_ltrain = K_linker[train_indices,:][:,train_indices]
    K_rtrain = K_fg[train_indices,:][:,train_indices]
    
    K_ntest = K_node[test_indices, :][:, train_indices]
    K_ltest = K_linker[test_indices, :][:, train_indices]
    K_rtest = K_fg[test_indices, :][:, train_indices]
    
    # Calculate the Frobenius norm of the training kernels
    ntrain_fro = np.linalg.norm(K_ntrain, 'fro')
    ltrain_fro = np.linalg.norm(K_ltrain, 'fro')
    rtrain_fro = np.linalg.norm(K_rtrain, 'fro')
    
    # Normalize the train and test kernels by the train Frobenius
    K_ntrain_norm = (K_ntrain/ntrain_fro)**xi
    K_ltrain_norm = (K_ltrain/ltrain_fro)**xi
    K_rtrain_norm = (K_rtrain/rtrain_fro)**xi
    K_train = np.stack([K_ntrain_norm, K_ltrain_norm, K_rtrain_norm], axis=0)
    
    K_ntest_norm = (K_ntest/ntrain_fro)**xi
    K_ltest_norm = (K_ltest/ltrain_fro)**xi
    K_rtest_norm = (K_rtest/rtrain_fro)**xi
    K_test = np.stack([K_ntest_norm, K_ltest_norm, K_rtest_norm], axis=0)

    # Fit and predict
    mkl = WeightedKernelRidge(alpha=1, deltas=deltas, kernels='precomputed', random_state=42)
    mkl.fit(K_train, y_train)
    models[i] = mkl

    test_pred = mkl.predict(K_test)
    train_pred = mkl.predict(K_train)

    # Calculate the MAEs and MSEs
    train_mae = mean_absolute_error(y_train, train_pred)
    train_mse = mean_squared_error(y_train, train_pred)
    train_r2 = r2_score(y_train, train_pred)

    test_mae = mean_absolute_error(y_test, test_pred)
    test_mse = mean_squared_error(y_test, test_pred)
    test_r2 = r2_score(y_test, test_pred)


    # Save the results
    results[i] = {'train_codes': refcodes_train, 'test_codes': refcodes_test, 'train_labels': y_train, 'test_labels': y_test,\
                  'train_predictions': train_pred, 'test_predictions': test_pred, 'train_mae': train_mae,\
                 'test_mae': test_mae, 'train_r2': train_r2, 'test_r2': test_r2}

# Calculate the average performance over the five seeds
print('\n--------------------------------------------')
avg_train_mae = np.mean([results['train_mae'] for results in results.values()])
avg_test_mae = np.mean([results['test_mae'] for results in results.values()])
avg_train_r2 = np.mean([results['train_r2'] for results in results.values()])
avg_test_r2 = np.mean([results['test_r2'] for results in results.values()])

print('Training Error (MAE): ', round(avg_train_mae,4), ' +/- ',\
      round(np.std([results['train_mae'] for results in results.values()]),4))
print('Testing Error (MAE): ', round(avg_test_mae,4), ' +/- ',\
      round(np.std([results['test_mae'] for results in results.values()]),4))

print('Training Data r^2: ', round(avg_train_r2,4), ' +/- ',\
      round(np.std([results['train_r2'] for results in results.values()]),4))
print('Testing Data r^2: ', round(avg_test_r2,4), ' +/- ',\
      round(np.std([results['train_r2'] for results in results.values()]),4))
print('--------------------------------------------')

# Calculate the average prediction over each structure
pred_dict = {}
for i in seeds:
    codes = np.concatenate((results[i]['train_codes'],results[i]['test_codes']), axis=0).ravel()
    predictions = np.concatenate((results[i]['train_predictions'].flatten(),results[i]['test_predictions'].flatten()), axis=0)

    df = pd.DataFrame([codes, predictions]).T
    df.columns = ['Code', 'Prediction']
    df = df.set_index('Code')

    pred_dict[i] = df

labeled_preds = pd.concat(pred_dict, axis=1)
labeled_preds.columns = labeled_preds.columns.droplevel(1)
labeled_bg = pd.read_csv(y_path, header=None, names=['Structure', 'BandGap']).set_index('Structure')

avg_preds = {}
for i in labeled_preds.index:
    avg = np.mean(labeled_preds.loc[i])
    std = np.std(labeled_preds.loc[i])
    true_val = labeled_bg.loc[i]['BandGap']
    avg_preds[i] = {'True': true_val, 'Avg': avg, 'Std': std}

avg_preds = pd.DataFrame.from_dict(avg_preds, orient='index')
print(avg_preds)

Linker = ['Benzene', '1,5-Naph', '2,6-Naph', '1,5-Anth', '2,6-Anth', 'Phenanthrene', 'Biphenyl', 'Terphenyl']

ewg = ['DMPE', 'CHS', 'NO', 'I', 'Br', 'PHEN', 'Cl', 'EEPO', 'F', 'MEPO', 'CN', 'EMEPO', 'EPO', 'CHO', 'COOH', 'NO2', 'SO3H']
edg = ['NH2', 'SH', 'OH', 'OMe', 'OProp', 'OEt', 'OEEPO', 'OCOCH3', 'CH3', 'tBu', 'SO2H']

# Compute the contributions for each node, linker, and rgroups
contributions = {}
for i in seeds:
    train_pred = results[i]['train_predictions']

    splitter = ShuffleSplit(n_splits=1, test_size=test_size, random_state=i)
    train_indices, test_indices = next(splitter.split(y))
    
    K_ntrain = K_node[train_indices,:][:,train_indices]
    K_ltrain = K_linker[train_indices,:][:,train_indices]
    K_rtrain = K_fg[train_indices,:][:,train_indices]
    
    # Calculate the train Frobenius norm
    ntrain_fro = np.linalg.norm(K_ntrain, 'fro')
    ltrain_fro = np.linalg.norm(K_ltrain, 'fro')
    rtrain_fro = np.linalg.norm(K_rtrain, 'fro')
    
    # Normalize the train kernels by the train Frobenius
    K_ntrain_norm = (K_ntrain/ntrain_fro)**xi
    K_ltrain_norm = (K_ltrain/ltrain_fro)**xi
    K_rtrain_norm = (K_rtrain/rtrain_fro)**xi

    node_delta = np.exp(models[i].deltas_[0])
    linker_delta = np.exp(models[i].deltas_[1])
    r_delta = np.exp(models[i].deltas_[2])

    node_coefs = models[i].dual_coef_
    linker_coefs = models[i].dual_coef_
    r_coefs = models[i].dual_coef_

    node_vals = (node_delta*K_ntrain_norm).dot(node_coefs)
    linker_vals = (linker_delta*K_ltrain_norm).dot(linker_coefs)
    r_vals = (r_delta*K_rtrain_norm).dot(r_coefs)

    cof_contributions = {}
    index = 0
    for j in results[i]['train_codes']:
        cof_contributions[j] = {'cont,node': node_vals[index], 'cont,linker': linker_vals[index], 'cont,r': r_vals[index], 'bg,pred': train_pred[index]}
        index += 1
    contributions[i] = pd.DataFrame.from_dict(cof_contributions, orient='index')

# Sort values by linker
linker_ranks = {}
for i in seeds:
    avgs = {}
    for l in Linker:
        vals = []
        for j in range(len(contributions[i])):
            if contributions[i].index[j].startswith(l):
                vals.append(contributions[i].iloc[j]['cont,linker'])
        avgs[l] = {'Mean': np.mean(vals), 'Std': np.std(linker_vals)}

    df = pd.DataFrame.from_dict(avgs, orient='index')
    df['Rank'] = df['Mean'].rank(ascending=True).astype(int)
    linker_ranks[i] = df.sort_values(by='Mean', ascending=False)

    #print(df.sort_values(by='Mean', ascending=False))

# Normalize the mean contribution of each linker within each seed
for i in seeds:
    seed_group = linker_ranks[i]
    max_cont = max(seed_group['Mean'])
    min_cont = min(seed_group['Mean'])

    norm_means = []
    for j in range(len(seed_group)):
        norm_val = (seed_group.iloc[j]['Mean']-min_cont)/(max_cont-min_cont)
        norm_means.append(norm_val)
    linker_ranks[i]['Norm. Mean'] = norm_means

# Sort values by r-group
node_ranks, r_ranks = {}, {}
for i in seeds:
    node_avgs, r_avgs = {}, {}
    for r in R:
        node_vals, r_vals = [], []
        for j in range(len(contributions[i])):
            if contributions[i].index[j].endswith(f'-{r}'):
                node_vals.append(contributions[i].iloc[j]['cont,node'])
                r_vals.append(contributions[i].iloc[j]['cont,r'])
        node_avgs[r] = {'Mean': np.mean(node_vals), 'Std': np.std(node_vals)}
        r_avgs[r] = {'Mean': np.mean(r_vals), 'Std': np.std(r_vals)}

    node_df = pd.DataFrame.from_dict(node_avgs, orient='index')
    node_df['Rank'] = node_df['Mean'].rank(ascending=False).astype(int)
    node_ranks[i] = node_df.sort_values(by='Mean', ascending=True)

    r_df = pd.DataFrame.from_dict(r_avgs, orient='index')
    r_df['Rank'] = r_df['Mean'].rank(ascending=False).astype(int)
    r_ranks[i] = r_df.sort_values(by='Mean', ascending=True)
    
# Normalize the mean of each r-group within each seed
for i in seeds:
    seed_group = r_ranks[i]
    max_cont = max(seed_group['Mean'])
    min_cont = min(seed_group['Mean'])

    norm_means = []
    for j in range(len(seed_group)):
        norm_val = (seed_group.iloc[j]['Mean']-min_cont)/(max_cont-min_cont)
        norm_means.append(norm_val)
    r_ranks[i]['Norm. Mean'] = norm_means
