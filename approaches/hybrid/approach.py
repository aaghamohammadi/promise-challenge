import sys
import numpy as np
import pandas as pd
import csv
import random
import xgboost as xgb


from sklearn.ensemble import RandomForestClassifier


from imblearn.over_sampling import SMOTE

from joblib import dump

from utils import *

def approach():
    args = sys.argv

    data_path = args[1]
    score_path = args[2]
    approach_name = args[3]
    drop_months_end = int(args[4])
    num_test_commits = int(args[5])

    ######################################
    # Loop for within project prediction #
    # Loads only one project at a time   #
    ######################################

    for project_name in list_all_projects(path=data_path):
        print(project_name)
        data = load_project(path=data_path, project_name=project_name)

        train_df, test_df = prepare_within_project_data(data, drop_months_end=drop_months_end, num_test_commits=num_test_commits)

        #########################################
        # Build Classifier                      #
        # (should be adopted for your approach) #
        #########################################

        # extract columns with feature values from available data
        # we prepared the following feature lists for your convenience:
        #
        # ALL_FEATURES
        # STATIC_FEATURES
        # STATIC_FILE_FEATURES
        # STATIC_CLASS_FEATURES
        # STATIC_INTERFACE_FEATURES
        # STATIC_ENUM_FEATURES
        # STATIC_METHOD_FEATURES
        # FGJIT_FEATURES
        # JIT_FEATURES
        # WD_FEATURES
        # PMD_FEATURES
        #
        # please check the documentation to see which features are included in each list
        # https://github.com/smartshark/promise-challenge/blob/main/dataset.md
        # 
        # we use all available features for our baseline

        # train_df = train_df.sample(frac=0.2)

        with open('200_selected_features.csv', newline='') as csvfile:
            spamreader = csv.reader(csvfile)
            feature_names = next(spamreader)

        train_df_new = pd.concat([train_df[feat] for feat in feature_names], axis=1)
        test_df_new = pd.concat([test_df[feat] for feat in feature_names], axis=1)

        X_train = train_df_new.values
        X_test = test_df_new.values

        # binary labels are in the column 'is_inducing'
        y_train = train_df['is_inducing']
        y_test = test_df['is_inducing']

        # we recommend using a fixed random seed for reproducibility, but this is up to you
        RANDOM_SEED = 42
        np.random.seed(RANDOM_SEED)

        # we resample with SMOTE and build a random forest for our baseline
        X_res, y_res = SMOTE(random_state=RANDOM_SEED).fit_resample(X_train, y_train)
        # TODO: for
        preds_rf = []
        preds_xgb = []
        for r_seed in range(20):
            rf = RandomForestClassifier(random_state=r_seed, oob_score=True, n_estimators=200, max_features="log2", n_jobs=-1)
            rf.fit(X_res, y_res)

            xgbst = xgb.XGBClassifier(random_state=r_seed, n_jobs=-1, n_estimators=250, max_depth=5 + random.randint(-3,3), learning_rate=0.2)
            xgbst.fit(X_res, y_res)

            preds_rf.append(rf.predict_proba(X_test)[:,1])
            preds_xgb.append(xgbst.predict_proba(X_test)[:,1])

        # TODO: end of loop
        y_pred_rf = np.vstack(preds_rf).T.mean(axis=1)
        y_pred_xgb = np.vstack(preds_xgb).T.max(axis=1)

        y_pred = (y_pred_rf + y_pred_xgb) / 2

        # dump(rf, '200_important_features_rf.joblib')

        ######################################################
        # DO NOT TOUCH FROM HERE                             #
        # This is where the scores are calculated and stored #
        ######################################################

        scores = score_model(test_df, y_pred >= 0.5)
        print_summary(train_df, test_df, scores)
        write_scores(score_path, approach_name, project_name, scores)

        
if __name__ == '__main__':
    approach()