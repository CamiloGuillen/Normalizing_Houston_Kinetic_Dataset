from sklearn.ensemble import IsolationForest
from sklearn.covariance import EllipticEnvelope
from sklearn.neighbors import LocalOutlierFactor


class OutlierDetector:
    def __init__(self, method='isolation_forest'):
        if method == 'iForest':
            self.method = IsolationForest()
        elif method == 'MCD':
            self.method = EllipticEnvelope()
        elif method == 'LOF':
            self.method = LocalOutlierFactor()
        else:
            raise ValueError(f"Method: '{method}' is not supported.")

    def detect_outliers(self, x):
        if x.shape[0] > 0:
            y_hat = self.method.fit_predict(X=x)
            outliers_idx = [i for i, y in enumerate(y_hat) if y == -1]
        else:
            outliers_idx = list()

        return outliers_idx
