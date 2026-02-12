import subprocess
import numpy as np
import pandas as pd
import scipy as sp 
import os
import pickle
import random
import time
import shap

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor

# Permutation importance method from sklearn
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

from sklearn.linear_model import LinearRegression
import itertools

# Read in data
FEATURE = pd.read_csv('stoich-features.csv', header=0, index_col=0)
BANDGAP = pd.read_csv('all-bandgaps.csv', header=None, index_col=0, names=['BandGap(eV)']).reindex(FEATURE.index)

ALL_DATA = FEATURE
ALL_DATA['BandGap(eV)'] = BANDGAP['BandGap(eV)']

ALL_FEATURES = ALL_DATA.drop(columns=['BandGap(eV)'])
ALL_FEATURES = ALL_FEATURES.columns.to_list()

if ALL_DATA.isnull().values.any() == True:
    print('There are NaN values in the FEATURE MATRIX. Please correct this and try again.')

# Define hyperparameters
test_size = 0.3
seeds = [2873,317,3846,446,3555]
params = {'n_estimators': 667, 'min_samples_split': 5, 'min_samples_leaf': 1,\
        'max_features': 'sqrt', 'max_depth': 10, 'bootstrap': False}

# Train the models for each seed
importances = []
results = {}
for i in seeds:
    # Split for training and testing
    train_set, test_set = train_test_split(
        ALL_DATA, test_size=test_size, shuffle=True, random_state=i)

    X_train = train_set.loc[:, (ALL_DATA.columns != 'BandGap(eV)')]
    X_test = test_set.loc[:, (ALL_DATA.columns != 'BandGap(eV)')]

    refcodes_train = X_train.index
    refcodes_test = X_test.index

    y_train = train_set.loc[:, ALL_DATA.columns == 'BandGap(eV)'].to_numpy()
    y_test = test_set.loc[:, ALL_DATA.columns == 'BandGap(eV)'].to_numpy()

    # Train the model and hyperparameterize with 3-fold cross-validation
    rf = RandomForestRegressor(n_estimators=params['n_estimators'], max_depth=params['max_depth'], min_samples_split=params['min_samples_split'],\
                               min_samples_leaf=params['min_samples_leaf'], max_features=params['max_features'], bootstrap=params['bootstrap'],\
                              random_state=42)
    rf.fit(X_train, y_train.ravel())

    # Evaluate the model
    y_train_pred = rf.predict(X_train)
    y_test_pred = rf.predict(X_test)

    train_mae = float(mean_absolute_error(y_train, y_train_pred))
    test_mae = float(mean_absolute_error(y_test, y_test_pred))

    train_r2 = float(r2_score(y_train, y_train_pred))
    test_r2 = float(r2_score(y_test, y_test_pred))

    # Calculate the important features through permutation
    permutation = permutation_importance(rf, X_test, y_test, n_repeats=10, random_state=42)
    import_dict = {'Mean':permutation.importances_mean, 'Std.':permutation.importances_std}
    import_df = pd.DataFrame.from_dict(import_dict, orient='columns')
    import_df.index = ALL_FEATURES
    importances.append(import_df)

    # Save the results
    results[i] = {'train_codes': refcodes_train, 'test_codes': refcodes_test, 'train_labels': y_train, 'test_labels': y_test,\
                  'train_predictions': y_train_pred, 'test_predictions': y_test_pred, 'train_mae': train_mae,\
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

perform_df = pd.DataFrame.from_dict(results, orient='index')
perform_df = perform_df.drop(['train_codes', 'test_codes', 'train_labels', 'test_labels', 'train_predictions', 'test_predictions'], axis=1)
print(perform_df)

# Calculate the average prediction for each structure
pred_dict = {}
for i in seeds:
    codes = np.concatenate((np.array(results[i]['train_codes']),np.array(results[i]['test_codes'])), axis=0).ravel()
    predictions = np.concatenate((results[i]['train_predictions'],results[i]['test_predictions']), axis=0)

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
    true_val = BANDGAP.loc[i]['BandGap(eV)']
    avg_preds[i] = {'True': true_val, 'Avg': avg, 'Std': std}

avg_preds = pd.DataFrame.from_dict(avg_preds, orient='index')
print(avg_preds)

# Calculate SHAP values for one seed
seed = seeds[2]

train_set, test_set = train_test_split(
    ALL_DATA, test_size=test_size, shuffle=True, random_state=seed)

X_train = train_set.loc[:, (ALL_DATA.columns != 'BandGap(eV)')]
print(np.shape(X_train))

explainer = shap.TreeExplainer(models[seed].best_estimator_)
shap_values = explainer(X_train)
