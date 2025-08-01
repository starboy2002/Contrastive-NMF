import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


def build_design_matrix(eeg, lags):
    """
    构建 time-lagged EEG 设计矩阵 R_lagged
    ----------
    eeg: np.array of shape (C, T)
        多通道 EEG
    lags: list of int
        以采样点为单位的延迟值（如 [0, 1, 2, ..., 64]）

    Returns
    -------
    R_lagged: (C * L, T - max_lag)
    """
    C, T = eeg.shape
    L = len(lags)
    max_lag = max(lags)
    R_lagged = []

    for lag in lags:
        if lag == 0:
            R_lagged.append(eeg[:, max_lag:])
        else:
            R_lagged.append(eeg[:, max_lag - lag:T - lag])

    R_lagged = np.concatenate(R_lagged, axis=0)  # shape = (C*L, T - max_lag)
    return R_lagged


def train_trf(R_lagged, H_target, alpha=0.1):
    """
    训练 TRF：Ridge 回归

    R_lagged: np.array, shape (C*L, T)
        EEG 延迟特征
    H_target: np.array, shape (K, T)
        NMF 得到的 target source 激活
    alpha: float
        正则化系数

    Returns
    -------
    G: (C*L, K)
        TRF 解码器权重矩阵
    """
    # 标准化
    scaler = StandardScaler()
    R_scaled = scaler.fit_transform(R_lagged.T).T

    # 多输出回归：逐列训练
    K = H_target.shape[0]
    G = np.zeros((R_lagged.shape[0], K))
    for k in range(K):
        model = Ridge(alpha=alpha)
        model.fit(R_scaled.T, H_target[k])
        G[:, k] = model.coef_

    return G, scaler


def predict_trf(R_lagged_test, G, scaler):
    """
    使用训练好的 TRF 对 EEG 预测 H_tilde

    R_lagged_test: (C*L, T)
    G: (C*L, K)
    Returns: (K, T)
    """
    R_scaled = scaler.transform(R_lagged_test.T).T
    H_pred = G.T @ R_scaled  # shape: (K, T)
    return H_pred
