AVAILABLE_MODELS = {
    "logistic-regression-distance-only": {
        "model_type": "logreg",
        "model_desc": "Logistic Regression Distance Only",
        "comet_model_name": "logistic-regression-distance-only",
        "version": "1.0.2",
        "file_name": "LR_distance_only",
    },
    "logistic-regression-angle-only": {
        "model_type": "logreg",
        "model_desc": "Logistic Regression Angle Only",
        "comet_model_name": "logistic-regression-angle-only",
        "version": "1.0.3",
        "file_name": "LR_angle_only",
    },
    "logistic-regression-distance-and-angle": {
        "model_type": "logreg",
        "model_desc": "Logistic Regression Distance and Angle",
        "comet_model_name": "logistic-regression-distance-and-angle",
        "version": "1.0.2",
        "file_name": "LR_distance_and_angle",
    },
    "xgboost-lasso": {
        "model_type": "xgboost_lasso",
        "model_desc": "XGBoost Model with Lasso",
        "comet_model_name": "xgboost-lasso",
        "version": "1.0.1",
        "file_name": "xgboost_lasso",
    },
    "xgboost-shap": {
        "model_type": "xgboost_SHAP",
        "model_desc": "XGBoost Model with SHAP",
        "comet_model_name": "xgboost-shap",
        "version": "1.0.1",
        "file_name": "xgboost_SHAP",
    },
    "xgboost-feats-non-corr": {
        "model_type": "xgboost_non_corr",
        "model_desc": "XGBoost with Non Correlated Features",
        "comet_model_name": "xgboost-feats-non-corr",
        "version": "1.0.1",
        "file_name": "xgboost_feats_non_corr",
    },
    "nn-adv": {
        "model_type": "NN_MLP",
        "model_desc": "Neural Network - Advance Features",
        "comet_model_name": "nn-adv",
        "version": "1.0.1",
        "file_name": "NN_adv",
    },
    "lr-all-feats": {
        "model_type": "logreg_all",
        "model_desc": "logistic Regression with all Features in (Q4)",
        "comet_model_name": "lr-all-feats",
        "version": "1.0.0",
        "file_name": "lr_all_feats",
    },
    "lr-non-corr-feats": {
        "model_type": "logreg_non_corr_feats",
        "model_desc": "Logistic Regression without Correlated Features",
        "comet_model_name": "lr-non-corr-feats",
        "version": "1.0.0",
        "file_name": "lr_non_corr_feats",
    },
    "xgboost-SMOTE": {
        "model_type": "xgboost_SMOTE",
        "model_desc": "XGBoost with SMOTE Oversampling",
        "comet_model_name": "xgboost-SMOTE",
        "version": "1.0.0",
        "file_name": "xgboost_SMOTE",
    },
    "lr-SMOTE": {
        "model_type": "logreg_SMOTE",
        "model_desc": "Logistic Regression with SMOTE Oversampling",
        "comet_model_name": "lr-SMOTE",
        "version": "1.0.0",
        "file_name": "lr_SMOTE",
    },
}

PROJECT_NAME = "ift6758-hockey"
WORK_SPACE = "tim-k-lee"
