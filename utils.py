import pickle

import numpy as np
import os

import pandas as pd
from matplotlib import pyplot as plt
from openpyxl.workbook import Workbook
from paddlets.models.model_loader import load


def RSE(pred, true):
    return np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(np.sum((true - true.mean()) ** 2))


def CORR(pred, true):
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
    d = np.sqrt(((true - true.mean(0)) ** 2 * (pred - pred.mean(0)) ** 2).sum(0))
    return (u / d).mean(-1)


def MAE(pred, true):
    return np.mean(np.abs(pred - true))


def MSE(pred, true):
    return np.mean((pred - true) ** 2)


def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))


# def MAPE(pred, true):
#     return np.mean(np.abs((pred - true) / true)) * 100
def MAPE(pred, true):
    with np.errstate(divide='ignore', invalid='ignore'):
        try:
            mape = np.mean(np.abs((pred - true) / true)) * 100
            if np.isnan(mape):
                raise ValueError("Result is NaN")
            if np.isinf(mape):
                raise ValueError("Result is inf")
            return mape
        except ZeroDivisionError:
            print('Error: Division by zero occurred.')
            return None
        except ValueError:
            print('Warning: Result MAPE is NaN or inf.')
            non_zero_mask = true != 0
            mape = np.mean(np.abs((pred[non_zero_mask] - true[non_zero_mask]) / true[non_zero_mask])) * 100
            return mape


# def MSPE(pred, true):
#     return np.mean(np.square((pred - true) / true)) *100

def MSPE(pred, true):
    with np.errstate(divide='ignore', invalid='ignore'):
        try:
            mspe = np.mean(np.square((pred - true) / true)) * 100
            if np.isnan(mspe):
                raise ValueError("Result is NaN")
            if np.isinf(mspe):
                raise ValueError("Result is inf")
            return mspe
        except ZeroDivisionError:
            print('Error: Division by zero occurred.')
            return None
        except ValueError:
            print('Warning: Result MSPE is NaN or inf.')
            non_zero_mask = true != 0
            mspe = np.mean(np.square((pred[non_zero_mask] - true[non_zero_mask]) / true[non_zero_mask])) * 100
            return mspe


def MEAN(true):
    # true = true.to_numpy()
    return true.mean()


def mae_per_mean(pred, true):
    mae = MAE(pred, true)
    mean = MEAN(true)
    return (mae/mean) * 100
    pass
def metric(pred, true):

    pred = pred.to_numpy()
    true = true.to_numpy()

    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)
    mspe = MSPE(pred, true)
    maePerMean = mae_per_mean(pred, true)
    print('MAE loss:', mae)
    print('MAE/Mean %:', maePerMean)
    print('MAPE loss:', mape)

    return mae, mse, rmse, mape, mspe, maePerMean

# Save the MinMaxScaler object to a file
def save_scale(scaler, scale_path, flag=True):
    if flag:
        with open(scale_path, 'wb') as f:
            pickle.dump(scaler, f)
        pass
    else:
        return


# Load the MinMaxScaler object from the file
def load_scale(scale_path):
    with open('note.scaler_path', 'rb') as f:
        sc_loaded = pickle.load(f)
    return sc_loaded


def load_model(model_path):
    return load(model_path)
    pass


def inverse_scale(scaler, data):
    data_inv = scaler.inverse_transform(data)
    return data_inv

    pass


def append_to_excel(prediction, true, excel_path, target, out_chunk_len):
    if not os.path.exists(excel_path):
        wb = Workbook()
        wb.save(excel_path)

    with pd.ExcelWriter(excel_path, engine='openpyxl', mode='a') as writer:
        df = pd.DataFrame()
        # Append the new data to the existing DataFrame
        df[f'pred_{out_chunk_len}'] = prediction[target]
        df[f'true_{out_chunk_len}'] = true[target]
        print(df)
        # Write the DataFrame to the Excel file
        try:
            df.to_excel(writer, sheet_name=f'len_{out_chunk_len}', index=true)
        except ValueError:
            print(f"Sheet len_{out_chunk_len} already exists")

def paddlets_plot(data, data_added, fig_path, label="", title="", save_fig=True):
    ax = data.plot(add_data=data_added, labels=label, title=title)
    # Get the parent Figure object from the AxesSubplot
    fig = ax.get_figure()
    if save_fig:
        # Save the plot using the Figure object
        fig.savefig(f"./{fig_path}{title}.png")
        # Remove the previous ax.get_figure() object
        plt.close(fig)
    pass


def round_number(*number, n_decimal=2):
    tmp = []
    for n in number:
        tmp.append(round(n, n_decimal))
    if len(tmp) == 1:
        return tmp[0]
    else:
        return tmp


# Define a callback function
def print_loss(epoch, logs):
    loss = logs['loss']
    print(f'Epoch: {epoch + 1}, Loss: {loss}')

def check_valid_inputs(val_set, test_set, in_chunk_len, out_chunk_len):
    val_test_data = [val_set, test_set]
    len_val = len(val_set.target)
    for ds in val_test_data:
        len_ds = len(ds.target)
        if in_chunk_len > len_ds:
            print("Please reducing in_chunk_len or increase val_or_test_set len!")
            raise ValueError(f"Error: in_chunk_len_{in_chunk_len} cannot be larger than val_or_test_len_{len_ds}")
    if len_val < in_chunk_len + out_chunk_len:
        raise ValueError(f"Error: val_len_{len_val} cannot be smaller than in_chunk_len{in_chunk_len} + out_chunk_len_{out_chunk_len}")
    pass

class InitialFolder():
    def __init__(self, model_path, scale_path, back_test_path = "", next_24_steps_path = ""):
        self.model = model_path
        self.scale = scale_path
        self.back_test = back_test_path
        self.next_24_steps = next_24_steps_path

    def update_folder(self, folder_path):
        """
        Nhận folder path
        if folder đã tồn tại thì return
        else create new folder
        :return:
        """
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        else:
            return

    def update_all_folders(self):
        for attribute, value in vars(self).items():
            if value != "":
                self.update_folder(value)

if __name__ == "__main__":
    # ob = InitialFolder("mode1\\", "scaler1\\", "back1\\", "next1\\")
    # ob.update_folder(ob.model)
    path = 'save_figs/informer/next_24_timesteps'
    if os.path.exists(path):
        print("oke")
    print(os.getcwd())