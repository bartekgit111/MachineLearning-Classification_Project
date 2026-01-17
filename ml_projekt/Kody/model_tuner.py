### Wstęp do Uczenia Maszynowego - Projekt
##### Bartosz Chudek 327426

import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

from config import kfold

X_train = pd.read_csv("data/artifical_train_data.csv")
y_train = np.ravel(pd.read_csv("data/artifical_train_labels.csv")) - 1


def model_tuner(model, pre, params):

    pipeline = Pipeline([
        ("pre", pre if pre is not None else "passthrough"),
        ("model", model)
    ])

    grid_pipeline = GridSearchCV(pipeline,
                                    param_grid=params,
                                    cv = kfold,
                                    scoring="balanced_accuracy",
                                    n_jobs=-1
                                    )

    grid_pipeline.fit(X_train,y_train)

    best_idx = grid_pipeline.best_index_
    mean_ba = grid_pipeline.cv_results_["mean_test_score"][best_idx]
    std_ba = grid_pipeline.cv_results_["std_test_score"][best_idx]
  
    return grid_pipeline.best_estimator_, np.round(mean_ba, 6), np.round(std_ba, 6)
