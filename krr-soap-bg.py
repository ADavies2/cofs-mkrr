import pandas as pd
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import train_test_split, KFold, cross_val_predict
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from scipy.stats import spearmanr
import numpy as np
import os
import random
from sklearn.inspection import permutation_importance, partial_dependence, PartialDependenceDisplay
import shap

# Define paths to data
fingerprint_path = 'stoich-features.csv' # fingerprints (length N)
y_path = os.path.join('all-bandgaps.csv') # band gaps (length N)

# Read in data
df_features = pd.read_csv(fingerprint_path, index_col=0)
df_BG = pd.read_csv(y_path, header=None, names=['BG(eV)'])['BG(eV)']
df = pd.concat([df_features, df_BG], axis=1)
all_data = df.dropna()
refcodes = all_data.index

# Define hyperparameters
kernel = 'rbf' # kernel function
test_size = 0.3
seeds = [2873,317,3846,446,3555]
alpha = 0.01
gamma = 1

# Drop features
features_to_drop = ['CovalentRadii(eV)-DEV', 'Atomic#-DEV', 'Valence#p-AVG', 'Valence#p-DEV', 'IonizationE(eV)-DEV',\
                   'Polarizability(au)-AVG', 'Period#-AVG', 'Density(g/cm3)-AVG', 'CovalentRadii(eV)-AVG', 'Atomic#-AVG']

all_data = all_data.drop(features_to_drop, axis=1)
print(len(all_data.columns))

# Train the models for each seed
results = {}
for i in seeds:
    # Split for train/test
    train_set, test_set = train_test_split(
        df, test_size=test_size, shuffle=True, random_state=i)
        
    X_train = train_set.loc[:, (df.columns != 'BG(eV)')]
    X_test = test_set.loc[:, (df.columns != 'BG(eV)')]
        
    refcodes_train = X_train.index
    refcodes_test = X_test.index
        
    y_train = train_set.loc[:, df.columns == 'BG(eV)'].to_numpy()
    y_test = test_set.loc[:, df.columns == 'BG(eV)'].to_numpy()

    # Fit and predict
    krr = KernelRidge(alpha=alpha, gamma=gamma, kernel=kernel)
    krr.fit(X_train, y_train)
    models[i] = krr

    y_train_pred = krr.predict(X_train)
    y_test_pred = krr.predict(X_test)

    # Calculate the MAEs and MSEs
    train_mae = mean_absolute_error(y_train, y_train_pred)
    train_mse = mean_squared_error(y_train, y_train_pred)
    train_r2 = r2_score(y_train, y_train_pred)

    test_mae = mean_absolute_error(y_test, y_test_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)
    test_r2 = r2_score(y_test, y_test_pred)

    # Save the results
    results[i] = {'train_codes': refcodes_train, 'test_codes': refcodes_test, 'train_labels': y_train, 'test_labels': y_test,\
                  'train_predictions': y_train_pred, 'test_predictions': y_test_pred, 'train_mae': train_mae,\
                 'test_mae': test_mae, 'train_r2': train_r2, 'test_r2': test_r2}

# Calculate the average performance over five seeds
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

perform_df = pd.DataFrame.from_dict(results, orient='index')
perform_df = perform_df.drop(['train_codes', 'test_codes', 'train_labels', 'test_labels', 'train_predictions', 'test_predictions'], axis=1)
print(perform_df)

# Calculate the average prediction for each structure
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

avg_preds = {}
for i in labeled_preds.index:
    avg = np.mean(labeled_preds.loc[i])
    std = np.std(labeled_preds.loc[i])
    true_val = df_BG.loc[i]
    avg_preds[i] = {'True': true_val, 'Avg': avg, 'Std': std}

avg_preds = pd.DataFrame.from_dict(avg_preds, orient='index')
print(avg_preds)

# Calculate the SHAP values for one seed
seed = seeds[2]
krr_model = models[seed]

train_set, test_set = train_test_split(
    all_data, test_size=test_size, shuffle=True, random_state=seed)

X_train = train_set.loc[:, (all_data.columns != 'BG(eV)')]
print(np.shape(X_train))

sample = shap.sample(X_train, 185, random_state=42)
explainer = shap.KernelExplainer(krr_model.predict, sample)
shap_values = explainer.shap_values(sample)
