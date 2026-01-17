### Wstęp do Uczenia Maszynowego - Projekt
##### Bartosz Chudek 327426

import numpy as np

### select ###

params_tree_select = {
    "pre__selector__estimator__C": np.logspace(-3, 2, 10),
    "model__criterion": ["gini","entropy"],
    "model__max_depth": [None,1,3,5,7,9],
    "model__min_samples_leaf": [1,3,5,7,9],
    "model__splitter": ["best","random"],
    "model__min_samples_split": [2,4,6,8,10]
}

params_glm_select = {
    "pre__selector__estimator__C": np.logspace(-3, 2, 10),
    "model__solver": ["lbfgs","newton-cg","sag","saga"]
}

params_glm_l2_select = {
    "pre__selector__estimator__C": np.logspace(-3, 2, 10),
    "model__solver": ["lbfgs", "liblinear", "newton-cg", "sag", "saga"],
    "model__C": np.logspace(-3, 2, 10)
}

params_glm_l1_select = {
    "pre__selector__estimator__C": np.logspace(-3, 2, 10),
    "model__solver": ["liblinear", "saga"],
    "model__C": np.logspace(-3, 2, 10)
}

params_glm_en_select = {
    "pre__selector__estimator__C": np.logspace(-3, 2, 10),
    "model__C": np.logspace(-3, 2, 10),
    "model__l1_ratio": np.linspace(0.001, 0.999, 10)
}

params_svm_select = {
    "pre__selector__estimator__C": np.logspace(-3, 2, 10),
    "model__kernel": ["linear", "poly", "rbf", "sigmoid"],
    "model__C": np.logspace(-3, 2, 10),
    "model__gamma": ["scale", "auto"]
}

params_lda_select = {
    "pre__selector__estimator__C": np.logspace(-3, 2, 10)
}

params_qda_select = {
    "pre__selector__estimator__C": np.logspace(-3, 2, 10),
    "model__reg_param": [0.0, 0.01, 0.05, 0.1, 0.2, 0.5]
}

params_knn_select = {
    "pre__selector__estimator__C": np.logspace(-3, 2, 10),
    "model__n_neighbors": [5, 10, 20, 50, 80, 100, 150, 200],
    "model__weights": ["uniform", "distance"],
    "model__metric": ["euclidean", "manhattan"]
}

### PCA ###

params_tree_PCA = {
    "pre__selector__n_components": [20, 40, 60, 75, 80, 90, 95],
    "pre__selector__whiten": [True, False],
    "pre__selector__svd_solver": ["full", "randomized"],
    "model__criterion": ["gini","entropy"],
    "model__max_depth": [None,1,3,5,7,9],
    "model__min_samples_leaf": [1,3,5,7,9],
    "model__splitter": ["best","random"],
    "model__min_samples_split": [2,4,6,8,10]
}

params_glm_PCA = {
    "pre__selector__n_components": [20, 40, 60, 75, 80, 90, 95],
    "pre__selector__whiten": [True, False],
    "pre__selector__svd_solver": ["full", "randomized"],
    "model__solver": ["lbfgs","newton-cg","sag","saga"]
}

params_glm_l2_PCA = {
    "pre__selector__n_components": [20, 40, 60, 75, 80, 90, 95],
    "pre__selector__whiten": [True, False],
    "pre__selector__svd_solver": ["full", "randomized"],
    "model__solver": ["lbfgs", "liblinear", "newton-cg", "sag", "saga"],
    "model__C": np.logspace(-3, 2, 10)
}

params_glm_l1_PCA = {
    "pre__selector__n_components": [20, 40, 60, 75, 80, 90, 95],
    "pre__selector__whiten": [True, False],
    "pre__selector__svd_solver": ["full", "randomized"],
    "model__solver": ["liblinear", "saga"],
    "model__C": np.logspace(-3, 2, 10)
}

params_glm_en_PCA = {
    "pre__selector__n_components": [20, 40, 60, 75, 80, 90, 95],
    "pre__selector__whiten": [True, False],
    "pre__selector__svd_solver": ["full", "randomized"],
    "model__C": np.logspace(-3, 2, 10),
    "model__l1_ratio": np.linspace(0.001, 0.999, 10)
}

params_svm_PCA = {
    "pre__selector__n_components": [20, 40, 60, 75, 80, 90, 95],
    "pre__selector__whiten": [True, False],
    "pre__selector__svd_solver": ["full", "randomized"],
    "model__kernel": ["linear", "poly", "rbf", "sigmoid"],
    "model__C": np.logspace(-3, 2, 10),
    "model__gamma": ["scale", "auto"]
}

params_lda_PCA = {
    "pre__selector__n_components": [20, 40, 60, 75, 80, 90, 95],
    "pre__selector__whiten": [True, False],
    "pre__selector__svd_solver": ["full", "randomized"]
}

params_qda_PCA = {
    "pre__selector__n_components": [20, 40, 60, 75, 80, 90, 95],
    "pre__selector__whiten": [True, False],
    "pre__selector__svd_solver": ["full", "randomized"],
    "model__reg_param": [0.0, 0.01, 0.05, 0.1, 0.2, 0.5]
}

params_knn_PCA = {
    "pre__selector__n_components": [20, 40, 60, 75, 80, 90, 95],
    "pre__selector__whiten": [True, False],
    "pre__selector__svd_solver": ["full", "randomized"],
    "model__n_neighbors": [5, 10, 20, 50, 80, 100, 150, 200],
    "model__weights": ["uniform", "distance"],
    "model__metric": ["euclidean", "manhattan"]
}


################# komitety #######################

params_rf = {
    "model__n_estimators": np.arange(50,500,50),
    "model__criterion": ["gini","entropy"],
    # "model__max_depth": [None, 5, 10],
    # "model__min_samples_leaf": [1,5,9],
    # "model__min_samples_split": [2,6,10],
    "model__max_features": ["sqrt", 0.5]
}

params_bagging = {
    "model__n_estimators": np.arange(50,500,50)
    # "model__max_samples": [None, 0.6, 0.8],
    # "model__max_features": [0.5, 0.7, 0.9, 1.0],
}

params_et = {
    "model__n_estimators": np.arange(50,500,50),
    "model__criterion": ["gini","entropy"],
    "model__max_depth": [None, 10, 20],
    # "model__min_samples_leaf": [1,5,9],
    # "model__min_samples_split": [2,6,10],
    "model__max_features": ["sqrt", 0.5]
}

params_gbc = {
    "model__n_estimators": np.arange(50,500,50),
    "model__learning_rate": [0.01, 0.05, 0.1],
    "model__max_depth": [2, 3, 4],
    "model__min_samples_leaf": [1,5,9]
}

params_xgb = {
    "model__n_estimators": np.arange(50,500,50),
    "model__learning_rate": [0.01, 0.05, 0.1],
    "model__max_depth": [3, 5],
    "model__subsample": [0.8, 1.0],
    "model__colsample_bytree": [0.8, 1.0],
    "model__min_child_weight": [1, 5,10]
    # "model__gamma": [0, 0.1],
    # "model__reg_alpha": [0, 0.1],
    # "model__reg_lambda": [1, 5]
}

params_lgbm = {
    "model__n_estimators": np.arange(50,500,50),
    "model__learning_rate": [0.01, 0.05, 0.1],
    "model__num_leaves": [15,31,63],
    "model__max_depth": [-1, 5],
    "model__min_child_samples": [10, 20, 50],
    "model__subsample": [0.8, 1.0],
    "model__colsample_bytree": [0.8, 1.0]
}



param_grids = {
    # select
    "params_tree_select": params_tree_select,
    "params_glm_select": params_glm_select,
    "params_glm_l2_select": params_glm_l2_select,
    "params_glm_l1_select": params_glm_l1_select,
    "params_glm_en_select": params_glm_en_select,
    "params_svm_select": params_svm_select,
    "params_lda_select": params_lda_select,
    "params_qda_select": params_qda_select,
    "params_knn_select": params_knn_select,

    # PCA
    "params_tree_PCA": params_tree_PCA,
    "params_glm_PCA": params_glm_PCA,
    "params_glm_l2_PCA": params_glm_l2_PCA,
    "params_glm_l1_PCA": params_glm_l1_PCA,
    "params_glm_en_PCA": params_glm_en_PCA,
    "params_svm_PCA": params_svm_PCA,
    "params_lda_PCA": params_lda_PCA,
    "params_qda_PCA": params_qda_PCA,
    "params_knn_PCA": params_knn_PCA,

    # komitety
    "params_rf": params_rf,
    "params_bagging": params_bagging,
    "params_et": params_et,
    "params_gbc": params_gbc,
    "params_xgb": params_xgb,
    "params_lgbm": params_lgbm,
}

