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

from xgboost import XGBClassifier
from sklearn.metrics import f1_score

from xgboost import XGBClassifier
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
        num_class=5, # 5 modes
        tree_method="hist",
        predictor="gpu_predictor"
    )

    xgb_opt.fit(X_train, y_train)
    y_pred = xgb_opt.predict(X_val)
    f1_macro = f1_score(y_val, y_pred, average='macro')

    return f1_macro






# part of this code is from: 
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




# https://github.com/JoseAngelMartinB/prediction-behavioural-analysis-ml-travel-mode-choice/blob/master/expermients_functions.py
import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, balanced_accuracy_score


from sklearn.metrics import precision_score, recall_score, f1_score, balanced_accuracy_score, log_loss
import pandas as pd

# function for different metrics
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
    mapping = {0: 'Class 0', 1: 'Class 1', 2: 'Class 2', 3: 'Class 3', 4: 'Class 4', 5: 'Class 5'} 
    cm = pd.DataFrame(confusion_matrix(y_true, y_pred), index=mapping.keys(), columns=mapping.keys())
    sns.heatmap(cm, annot=True, fmt='g', cmap="Blues", ax=ax)
    ax.set_title(title)
    ax.set_ylabel('Actual Values')
    ax.set_xlabel('Predicted Values')


