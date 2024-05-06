from sklearn.preprocessing import scale,MinMaxScaler,Normalizer,StandardScaler
from sklearn.metrics import mean_squared_error,r2_score,mean_absolute_error
from sklearn.neural_network import MLPRegressor
import numpy as np


def ModelRgsevaluate(y_pred, y_true):

    mse = mean_squared_error(y_true,y_pred)
    R2  = r2_score(y_true,y_pred)
    mae = mean_absolute_error(y_true,y_pred)

    return np.sqrt(mse), R2, mae

def ModelRgsevaluatePro(y_pred, y_true, yscale):

    yscaler = yscale

    y_true = yscaler.inverse_transform(y_true)
    y_pred = yscaler.inverse_transform(y_pred)
    #y_true = y_true[:, 0]
    #y_pred = y_pred[:, 0]
    num_cols = y_true.shape[1]  # 获取数组的列数

    mse_list, r2_list, mae_list = [], [], []
    for i in range(num_cols):
        mse = mean_squared_error(y_true[:, i], y_pred[:, i])
        r2 = r2_score(y_true[:, i], y_pred[:, i])
        mae = mean_absolute_error(y_true[:, i], y_pred[:, i])

        mse_list.append(mse)
        r2_list.append(r2)
        mae_list.append(mae)
    # mse = mean_squared_error(y_true,y_pred)
    # R2  = r2_score(y_true,y_pred)
    # mae = mean_absolute_error(y_true, y_pred)

    return mse_list, r2_list, mae_list