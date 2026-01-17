from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline

from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import VarianceThreshold

from sklearn.linear_model import LogisticRegression

def preprocessor_select():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("const", VarianceThreshold(threshold=0.01)),
        ("selector", SelectFromModel(LogisticRegression(max_iter=10000, penalty="l1", solver="saga")))
    ])

def preprocessor_PCA():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("const", VarianceThreshold(threshold=0.01)),
        ("selector", PCA())
    ])

def preprocessor_rf():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("const", VarianceThreshold(threshold=0.01))
    ])

def preprocessor_et():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("const", VarianceThreshold(threshold=0.01))
    ])

def preprocessor_gbc():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("const", VarianceThreshold(threshold=0.01))
    ])

def preprocessor_xgb():
    return Pipeline([
        ("const", VarianceThreshold(threshold=0.1))
    ])

def preprocessor_lgbm():
    return Pipeline([
        ("const", VarianceThreshold(threshold=0.1))
    ])
