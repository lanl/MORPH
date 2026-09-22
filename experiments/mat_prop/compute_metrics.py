import numpy as np

# Denormalize function
def denormalize_params(norm_params, min_params, max_params):
    for i in range(norm_params.shape[1]):
        norm_params[:,i] = norm_params[:,i] * (max_params[i] - min_params[i]) + min_params[i]
    return norm_params

# metrics calculation on denormalized values
def metrics(y_org_list, y_pred_list, 
            minval_list=None, maxval_list=None):
    y_org_arr = np.concatenate(y_org_list, axis=0)
    y_pred_arr = np.concatenate(y_pred_list, axis=0)

    if minval_list is not None and maxval_list is not None:
        # Denormalize the predictions and true values
        y_org_arr = denormalize_params(y_org_arr, minval_list, maxval_list)
        y_pred_arr = denormalize_params(y_pred_arr, minval_list, maxval_list)
    
    # calculate on every parameter
    r2_list, mse_list, mape_list = [], [], []

    for j in range(y_org_arr.shape[1]):
        # r2 score
        r2 = np.corrcoef(y_org_arr[:, j], y_pred_arr[:, j])[0, 1] ** 2
        r2_list.append(r2)
        # mse loss
        mse = np.mean((y_org_arr[:, j] - y_pred_arr[:, j])**2)
        mse_list.append(mse)
        # mape
        mape = np.mean(np.abs((y_org_arr[:, j] - y_pred_arr[:, j]) / y_org_arr[:, j])) * 100
        mape_list.append(mape)
        
    return mse_list, r2_list, mape_list