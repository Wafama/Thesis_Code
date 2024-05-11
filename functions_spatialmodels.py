from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score

def Objective_knn(trial, X_train, y_train, X_val, y_val):
    n_neighbors = trial.suggest_int('n_neighbors', 250, 312, step=1)
    weights = trial.suggest_categorical('weights', ['uniform', 'distance'])
    metric = trial.suggest_categorical('metric', ['minkowski', 'euclidean', 'manhattan'])
    algorithm = trial.suggest_categorical('algorithm', ["auto", "ball_tree", "kd_tree", "brute"])

    knn_opt = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights, metric=metric, algorithm=algorithm)
    
    knn_opt.fit(X_train, y_train)  
    
    y_pred = knn_opt.predict(X_val)  
    
    f1_macro = f1_score(y_val, y_pred, average='macro') 

    return f1_macro



# part of this code is from: 
# https://www.kaggle.com/code/mustafagerme/optimization-of-random-forest-model-using-optuna
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

criterion_options = ["gini", "entropy"]
min_samples_leaf_options = [5, 7, 9]
max_features_options = ["sqrt", "log2", None]
bootstrap_options = [True, False]

def Objective_rf(trial, X_train, y_train, X_val, y_val):
    random_state = 0
    n_estimators = trial.suggest_int("n_estimators", 1, 300, log=True)
    max_depth = trial.suggest_int("max_depth", 1, 40)
    min_samples_leaf = trial.suggest_categorical("min_samples_leaf", min_samples_leaf_options)
    max_features = trial.suggest_categorical("max_features", max_features_options)
    bootstrap = trial.suggest_categorical("bootstrap", bootstrap_options)
    criterion = trial.suggest_categorical("criterion", criterion_options)

    rf_opt = RandomForestClassifier(
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        bootstrap=bootstrap,
        n_estimators=n_estimators,
        criterion=criterion,
        random_state=random_state)  

    rf_opt.fit(X_train, y_train)
    y_pred = rf_opt.predict(X_val)
    f1_macro = f1_score(y_val, y_pred, average='macro')

    return f1_macro





# part of this code is is from : 
# https://medium.com/@mlxl/knime-xgboost-and-optuna-for-hyper-parameter-optimization-dcf0efdc8ddf


from xgboost import XGBClassifier
from sklearn.metrics import f1_score
import optuna


def Objective_xgb(trial, X_train, y_train, X_val, y_val):
    learning_rate = trial.suggest_float("learning_rate", 0.01, 1.0, log=True)
    n_estimators = trial.suggest_int("n_estimators", 100, 1000)
    max_depth = trial.suggest_int("max_depth", 3, 30)
    min_child_weight = trial.suggest_float("min_child_weight", 0.1, 200.0)
    gamma = trial.suggest_float("gamma", 0.0, 1.0)
    subsample = trial.suggest_float("subsample", 0.5, 1.0)
    colsample_bytree = trial.suggest_float("colsample_bytree", 0.1, 1.0)
    reg_alpha = trial.suggest_float("reg_alpha", 0.0, 30.0)
    reg_lambda = trial.suggest_float("reg_lambda", 0.0, 30.0)

    xgb_opt = XGBClassifier(
        learning_rate=learning_rate,
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_child_weight=min_child_weight,
        gamma=gamma,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        reg_alpha=reg_alpha,
        reg_lambda=reg_lambda,
        random_state=0,
        objective='multi:softmax', 
        num_class=5, 
        tree_method="hist",
        predictor="gpu_predictor"
    )

    xgb_opt.fit(X_train, y_train)
    y_pred = xgb_opt.predict(X_val)
    f1_macro = f1_score(y_val, y_pred, average='macro')

    return f1_macro

    xgb_opt.fit(X_train, y_train)
    y_pred = xgb_opt.predict(X_val)
    f1_macro = f1_score(y_val, y_pred, average='macro')

    return f1_macro


    
    # Fit the model
    xgb_opt.fit(X_train, y_train)

    # prediction based on validaton set
    y_pred = xgb_opt.predict(X_val)
    f1_macro = f1_score(y_val, y_pred, average='macro')
    
    return f1_macro





# part of this code isfrom: 
# https://www.kaggle.com/code/neilgibbons/tuning-tabnet-with-optuna
from pytorch_tabnet.tab_model import TabNetClassifier
import torch
from sklearn.metrics import f1_score

def Objective_tabnet(trial, X_train, y_train, X_val, y_val):
    mask_type = trial.suggest_categorical("mask_type", ["entmax", "sparsemax"])
    n_steps = trial.suggest_int("n_steps", 3, 15, step=1)
    n_d = trial.suggest_int("n_d", 2, 12, step=2)
    gamma = trial.suggest_float("gamma", 1, 1.6, step=0.2)
    n_shared = trial.suggest_int("n_shared", 1, 4)
    lambda_sparse = trial.suggest_float("lambda_sparse", 1e-6, 1e-3, log=True)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    tabnet_opt = TabNetClassifier(
        n_steps=n_steps,
        n_d=n_d,
        gamma=gamma,
        n_shared=n_shared,
        lambda_sparse=lambda_sparse,
        verbose=0,
        device_name=device, 
        mask_type=mask_type, 
        optimizer_fn=torch.optim.Adam,
        optimizer_params=dict(lr=1e-2),
        scheduler_params=dict(
            mode="min",
            patience=trial.suggest_int("patienceScheduler", low=3, high=10), 
            min_lr=2e-2,
            factor=0.5), scheduler_fn=torch.optim.lr_scheduler.ReduceLROnPlateau)   

    tabnet_opt.fit(X_train, y_train, batch_size=10024, virtual_batch_size=9000)

    y_pred = tabnet_opt.predict(X_val)

    f1_macro = f1_score(y_val, y_pred, average='macro')

    return f1_macro





import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, balanced_accuracy_score

def all_metrics(y_true, y_pred, y_prob, column_name):
    pres_mac = float(precision_score(y_true, y_pred, average='macro'))
    rec_mac = float(recall_score(y_true, y_pred, average='macro'))
    f1_macro = float(f1_score(y_true, y_pred, average='macro'))
    balanced_acc = float(balanced_accuracy_score(y_true, y_pred))
    logloss = float(log_loss(y_true, y_prob)) 
    # Put all metrics in a dictionary
    evaluation_metrics = {
        'Precision (Macro)': str(round(pres_mac * 100, 2)) + "%", 
        'Recall (Macro)': str(round(rec_mac * 100, 2)) + "%",  
        'F1-score (Macro)': str(round(f1_macro * 100, 2)) + "%",
        'Balanced Accuracy': str(round(balanced_acc * 100, 2)) + "%",
        'Log Loss': logloss  
    }
    
    # Convert dictionary into DataFrame
    metrics_df = pd.DataFrame.from_dict(evaluation_metrics, orient='index', columns=[column_name])
    
    return metrics_df



import numpy as np

def predict(classifier, params, X_train, y_train, X_test):
    # Predict class labels and probabilities for a given classifier.
    
    # Instantiate classifier
    clf = classifier(**params)
    
    # Fit classifier
    clf.fit(X_train, y_train)
    
    # Label Prediction
    y_pred = clf.predict(X_test)
    
    # Probability prediction
    y_pred_proba = clf.predict_proba(X_test)
    
    # Ensure minimum probability
    y_pred_proba = np.where(y_pred_proba > 0.00001, y_pred_proba, 0.00001)
    
    return y_pred, y_pred_proba



import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

def conf_matrix(y_true, y_pred, title, ax):
    mapping = {0: 'Class 0', 1: 'Class 1', 2: 'Class 2', 3: 'Class 3', 4: 'Class 4', 5: 'Class 5'}  # Update with your class labels mapping
    cm = pd.DataFrame(confusion_matrix(y_true, y_pred), index=mapping.keys(), columns=mapping.keys())
    sns.heatmap(cm, annot=True, fmt='g', cmap="Blues", ax=ax)
    ax.set_title(title)
    ax.set_ylabel('Actual Values')
    ax.set_xlabel('Predicted Values')



# function that calculated spatial lag based on rook 
import geopandas as gpd
from libpysal import weights
import pandas as pd 
import numpy as np


lst =["SurroundingAddressDensity","AvgDistToTrainStation","NumBusStops",
    'RoadNetworkDensity',"AvgDistToSupermarket",
   "NofBusinessEstablishments"] # built enviroment features


def spatial_lag_rook(gdf, df, lst, Lag_name):
    # gdf: geopandas containing spatial data: geometry coordinates and postal codes.
    # lst: list  containing the names of built environment features for which spatial lag features will be computed
    # df = This is training, validation, or test sets (same as the ones used for the baseline models)
    # lag_names: string specifying the type of spatial lag to be calculated ("KNN8","KNN15", "Queen", "Rook", "DistanceBand")

    # extract relevant features + postcode for merging
    relevant_columns = ["HomePostalCode"] + lst

    # Drop duplicated to avoid computational problems
    subset_df = df[relevant_columns].drop_duplicates('HomePostalCode')

    # merge the two datasets based on postcode 
    gdf = pd.merge(subset_df, gdf, left_on='HomePostalCode', right_on="postcode4", how='left')

    # convert into geopandas for spatial autocorrelation analysis
    gdf = gpd.GeoDataFrame(gdf, geometry='geometry')

    # to mitigate the skewness apply log transformation before calculating spatial lag features
    for col in gdf.columns:
        if col not in ["HomePostalCode", "geometry"]:
            gdf[col] = np.log(gdf[col] + 0.001) # add 0.001 to avoid inf log

    # applied rook metrix to calculate spatial autocorrelation
    rook = weights.Rook.from_dataframe(gdf, silence_warnings=True)
    rook.transform = "R"

    # add spatial lag features into the dataframe
    for feature in lst: 
        lag_feature = weights.lag_spatial(rook, gdf[feature].values)
        gdf[f"{feature}_{Lag_name}"] = lag_feature

    # select relevant features (spatial lag features)
    gdf_rook = gdf[["HomePostalCode", "geometry"] + [f"{feat}_{Lag_name}" for feat in lst]]

    # create new data frame containing spatial lag features, merge with the original dataset with spatial lag dataset
    df = pd.merge(df, gdf_rook, on='HomePostalCode', how='left')
    
    return df



import geopandas as gpd
from libpysal import weights
import pandas as pd 
import numpy as np
def spatial_lag_knn(gdf, df, k, lst, Lag_name):

    # extract only relevant columns to avoid computational problem
    relevant_columns = ["HomePostalCode"] + lst

    # drop duplicates based on home postcode to avoid computational problems
    subset_df = df[relevant_columns].drop_duplicates('HomePostalCode')

    # merge two datasets based on geometry
    gdf = pd.merge(subset_df, gdf, left_on='HomePostalCode', right_on="postcode4", how='left')
    gdf = gpd.GeoDataFrame(gdf, geometry='geometry')

    # transform built environment features into log transform
    for col in gdf.columns:
        if col not in ["HomePostalCode", "geometry"]:
            gdf[col] = np.log(gdf[col] + 0.001)

    # define knn spatial weight
    knn = weights.KNN.from_dataframe(gdf, k=k)
    knn.transform = "R"

    # construct spatial lag features of built environment features
    for feature in lst: 
        lag_feature = weights.lag_spatial(knn, gdf[feature].values)
        gdf[f"{feature}_{Lag_name}"] = lag_feature

    # extract lag features 
    gdf_knn = gdf[["HomePostalCode", "geometry"] + [f"{feat}_{Lag_name}" for feat in lst]]

    # merge with an original dataset based on postcode
    df = pd.merge(df, gdf_knn, on='HomePostalCode', how='left')
    
    return df

import geopandas as gpd
from libpysal import weights
import pandas as pd 
import numpy as np
def spatial_lag_queen(gdf, df, lst, Lag_name):
    relevant_columns = ["HomePostalCode"] + lst
    subset_df = df[relevant_columns].drop_duplicates('HomePostalCode')
    
    gdf = pd.merge(subset_df, gdf, left_on='HomePostalCode', right_on="postcode4", how='left')
    gdf = gpd.GeoDataFrame(gdf, geometry='geometry')
    
    for col in gdf.columns:
        if col not in ["HomePostalCode", "geometry"]:
            gdf[col] = np.log(gdf[col] + 0.001)
    
    queen = weights.Queen.from_dataframe(gdf, silence_warnings=True)
    queen.transform = "R"
    
    for feature in lst: 
        lag_feature = weights.lag_spatial(queen, gdf[feature].values)
        gdf[f"{feature}_{Lag_name}"] = lag_feature
        
    gdf_queen = gdf[["HomePostalCode", "geometry"] + [f"{feat}_{Lag_name}" for feat in lst]]
    
    df = pd.merge(df, gdf_queen, on='HomePostalCode', how='left')
    
    return df


def spatial_lag_distance_band(gdf, df, lst, Lag_name, threshold_distance):
    relevant_columns = ["HomePostalCode"] + lst
    subset_df = df[relevant_columns].drop_duplicates('HomePostalCode')
    
    gdf = pd.merge(subset_df, gdf, left_on='HomePostalCode', right_on="postcode4", how='left')
    gdf = gpd.GeoDataFrame(gdf, geometry='geometry')
    
    for col in gdf.columns:
        if col not in ["HomePostalCode", "geometry"]:
            gdf[col] = np.log(gdf[col] + 0.001)
    
    distance_band = weights.DistanceBand.from_dataframe(gdf, threshold_distance, silence_warnings=True)
    distance_band.transform = "R"
    
    for feature in lst: 
        lag_feature = weights.lag_spatial(distance_band, gdf[feature].values)
        gdf[f"{feature}_{Lag_name}"] = lag_feature
        
    gdf_distance_band = gdf[["HomePostalCode", "geometry"] + [f"{feat}_{Lag_name}" for feat in lst]]
    
    df = pd.merge(df, gdf_distance_band, on='HomePostalCode', how='left')
    
    return df


#https://medium.com/anolytics/all-you-need-to-know-about-encoding-techniques-b3a0af68338b#:~:text=Ordinal%20encoding%20is%20similar%20to,map%20them%20to%20integers%20accordingly.

from sklearn.preprocessing import OrdinalEncoder

from sklearn.preprocessing import RobustScaler, OneHotEncoder, StandardScaler

from sklearn.preprocessing import FunctionTransformer



def encoding_categorical_features(X_train_split, X_val, X_test):
    educ_order = ["Low", "Medium", "High"]
    household_order = ['1', '2', '3', '4', '5 or more']
    age_order = ['6 to 11 years', '12 to 14 years', '15 to 17 years', '18 to 19 years', '20 to 24 years', '25 to 29 years',
                 '30 to 34 years', '35 to 39 years', '40 to 44 years', '45 to 49 years', '50 to 54 years', '55 to 59 years',
                 '60 to 64 years', '65 to 69 years', '70 to 74 years', '75 to 79 years', '80 years or older']
    cars_order = ['0', '1', '2', '3', '4 or more']
    binary_mapping = {"Female": 0, "Male": 1, "Yes": 1, "No": 0,"weekday": 1, "weekend": 0}
    income_order = ['First 10% group','Second 10% group','Third 10% group','Fourth 10% group','Fifth 10% group',
               'Sixth 10% group','Seventh 10% group','Eighth 10% group','Ninth 10% group','Tenth 10% group',]
    
    
    ordinal_order = [ 'Never or almost never', 'Several times a year',
       'Several times a month', 'Several times a week',
       'Daily or almost daily']
    ourdoor_encoder = OrdinalEncoder(categories=[ordinal_order])
    X_train_split["FrequencyOfWalkingOutdoors"] = ourdoor_encoder.fit_transform(X_train_split[["FrequencyOfWalkingOutdoors"]]).flatten()
    X_val["FrequencyOfWalkingOutdoors"] = ourdoor_encoder.transform(X_val[["FrequencyOfWalkingOutdoors"]]).flatten()
    X_test["FrequencyOfWalkingOutdoors"] = ourdoor_encoder.transform(X_test[["FrequencyOfWalkingOutdoors"]]).flatten()
    pass_encoder = OrdinalEncoder(categories=[ordinal_order])
    X_train_split["FrequencyOfUseCcarAsAPassenger"] = pass_encoder.fit_transform(X_train_split[["FrequencyOfUseCcarAsAPassenger"]]).flatten()
    X_val["FrequencyOfUseCcarAsAPassenger"] = pass_encoder.transform(X_val[["FrequencyOfUseCcarAsAPassenger"]]).flatten()
    X_test["FrequencyOfUseCcarAsAPassenger"] = pass_encoder.transform(X_test[["FrequencyOfUseCcarAsAPassenger"]]).flatten()
    
    non_elec_encoder = OrdinalEncoder(categories=[ordinal_order])
    X_train_split["FrequencyOfUseOfNonEelectricBicycle"] = non_elec_encoder.fit_transform(X_train_split[["FrequencyOfUseOfNonEelectricBicycle"]]).flatten()
    X_val["FrequencyOfUseOfNonEelectricBicycle"] = non_elec_encoder.transform(X_val[["FrequencyOfUseOfNonEelectricBicycle"]]).flatten()
    X_test["FrequencyOfUseOfNonEelectricBicycle"] = non_elec_encoder.transform(X_test[["FrequencyOfUseOfNonEelectricBicycle"]]).flatten()
    
    def ordinal_encode(data, categories, column):
        encoder = OrdinalEncoder(categories=[categories])
        data[column] = encoder.fit_transform(data[[column]]).flatten()
        return data
    
    def binary_encode(data, mapping, column):
        data[column].replace(mapping, inplace=True)
        return data
    
    # encode educationLevel
    X_train_split = ordinal_encode(X_train_split, educ_order, "EducationLevel")
    X_val = ordinal_encode(X_val, educ_order, "EducationLevel")
    X_test = ordinal_encode(X_test, educ_order, "EducationLevel")
    
    # encode disposableIncome
    income_encoder = OrdinalEncoder(categories=[income_order])
    X_train_split["DisposableIncome"] = income_encoder.fit_transform(X_train_split[["DisposableIncome"]]).flatten()
    X_val["DisposableIncome"] = income_encoder.transform(X_val[["DisposableIncome"]]).flatten()
    X_test["DisposableIncome"] = income_encoder.transform(X_test[["DisposableIncome"]]).flatten()

    # encode householdSize
    X_train_split = ordinal_encode(X_train_split, household_order, "HouseholdSize")
    X_val = ordinal_encode(X_val, household_order, "HouseholdSize")
    X_test = ordinal_encode(X_test, household_order, "HouseholdSize")

    # ecode ageClass
    X_train_split = ordinal_encode(X_train_split, age_order, "AgeClass")
    X_val = ordinal_encode(X_val, age_order, "AgeClass")
    X_test = ordinal_encode(X_test, age_order, "AgeClass")

    # encode NumberOfCarsInHousehold
    X_train_split = ordinal_encode(X_train_split, cars_order, "NumberOfCarsInHousehold")
    X_val = ordinal_encode(X_val, cars_order, "NumberOfCarsInHousehold")
    X_test = ordinal_encode(X_test, cars_order, "NumberOfCarsInHousehold")

    

    # encode other binary features
    for column in ["Gender", "License", "Weekday", "ElectriBicycleIHousehold"]:
        X_train_split = binary_encode(X_train_split, binary_mapping, column)
        X_val = binary_encode(X_val, binary_mapping, column)
        X_test = binary_encode(X_test, binary_mapping, column)
    
    return X_train_split, X_val, X_test





from sklearn.preprocessing import RobustScaler, OneHotEncoder, StandardScaler
import numpy as np

from sklearn.preprocessing import StandardScaler, OneHotEncoder
import numpy as np

from sklearn.preprocessing import StandardScaler, OneHotEncoder, RobustScaler
import numpy as np

from sklearn.preprocessing import RobustScaler, OneHotEncoder
import numpy as np

from sklearn.preprocessing import RobustScaler, OneHotEncoder
import numpy as np

import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler, OneHotEncoder

def scaling_features(X_train_split, X_val, X_test, ordinal_features, numeric_features, categorical_features, binary_features, lag_features, pca_features):
    # Scale numerical features only (excluding ordinal features)
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train_split[numeric_features])
    X_val_scaled = scaler.transform(X_val[numeric_features])
    X_test_scaled = scaler.transform(X_test[numeric_features])

    # One-hot encoding for categorical features
    encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
    encoder.fit(X_train_split[categorical_features])
    one_hot_feature_names = encoder.get_feature_names_out(input_features=categorical_features)

    X_train_cat_encoded = encoder.fit_transform(X_train_split[categorical_features])
    X_val_cat_encoded = encoder.transform(X_val[categorical_features])
    X_test_cat_encoded = encoder.transform(X_test[categorical_features])

    # Concatenate encoded categorical features with binary features
    X_train_encoded = np.concatenate([X_train_cat_encoded, X_train_split[binary_features]], axis=1)
    X_val_encoded = np.concatenate([X_val_cat_encoded, X_val[binary_features]], axis=1)
    X_test_encoded = np.concatenate([X_test_cat_encoded, X_test[binary_features]], axis=1)

    # Concatenate scaled numerical features with encoded categorical and binary features, lag features, and PCA features
    X_train_final = np.concatenate([X_train_scaled, X_train_encoded, X_train_split[ordinal_features], 
                                     X_train_split[lag_features], X_train_split[pca_features]], axis=1)
    X_val_final = np.concatenate([X_val_scaled, X_val_encoded, X_val[ordinal_features], 
                                   X_val[lag_features], X_val[pca_features]], axis=1)
    X_test_final = np.concatenate([X_test_scaled, X_test_encoded, X_test[ordinal_features], 
                                    X_test[lag_features], X_test[pca_features]], axis=1)

    all_feature_names = numeric_features + list(one_hot_feature_names) + binary_features + ordinal_features + lag_features + pca_features

    return X_train_final, X_val_final, X_test_final, all_feature_names, scaler








from sklearn.metrics import precision_score, recall_score, f1_score, balanced_accuracy_score, log_loss
import pandas as pd
def all_metrics(y_true, y_pred, y_prob, column_name):
    pres_mac = float(precision_score(y_true, y_pred, average='macro'))
    rec_mac = float(recall_score(y_true, y_pred, average='macro'))
    f1_macro = float(f1_score(y_true, y_pred, average='macro'))
    balanced_acc = float(balanced_accuracy_score(y_true, y_pred))
    logloss = float(log_loss(y_true, y_prob)) 
    # Put all metrics in a dictionary
    evaluation_metrics = {
        'Precision (Macro)': str(round(pres_mac * 100, 2)) + "%", 
        'Recall (Macro)': str(round(rec_mac * 100, 2)) + "%",  
        'F1-score (Macro)': str(round(f1_macro * 100, 2)) + "%",
        'Balanced Accuracy': str(round(balanced_acc * 100, 2)) + "%",
        'Log Loss': logloss  
    }
    
    # Convert dictionary into DataFrame
    metrics_df = pd.DataFrame.from_dict(evaluation_metrics, orient='index', columns=[column_name])
    
    return metrics_df


