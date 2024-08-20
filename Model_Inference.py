import argparse
from datetime import timedelta, datetime

import numpy as np
import paddle
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import os
from paddle.nn import L1Loss
from paddlets.models.common.callbacks import History, EarlyStopping

from utils import *
from paddlets.utils import backtest
from paddlets import TSDataset
from paddlets.datasets.repository import get_dataset
from paddlets.transform.sklearn_transforms import StandardScaler, MinMaxScaler
from paddlets.models.forecasting import MLPRegressor
from paddlets.models.forecasting.dl.informer import InformerModel
from paddlets.models.forecasting.dl.transformer import TransformerModel
from paddlets.models.model_loader import load
from paddlets.metrics import MAE as mae_paddlets

print("ok")
# data

data_A2M_TW79 = 'data/A2M_TW79/TW_0079_data_FillByERA5_Interpolate.csv'

# model path

model_A2M_TW79_out_1 = 'trained_models/informer/A2M_TW79/out_len_1/'
model_A2M_TW79_out_4 = 'trained_models/informer/A2M_TW79/out_len_4/'
model_A2M_TW79_out_6 = 'trained_models/informer/A2M_TW79/out_len_6/'

parser = argparse.ArgumentParser(description='[Informer-WindSpeedForecasting]')

parser.add_argument('--mode', type=str, default='a2m_79', help='mode')
parser.add_argument('--in_len', type=str, default='216', help='in chunk len')
parser.add_argument('--out_len', type=str, default='1', help='out chunk len')
parser.add_argument('--val_date', type=int, default=1, help='the date in the dataset that the model was evaluation, must range from 1 to 30')
args = vars(parser.parse_args())

# you can expan the data_parser if you have more dataset
data_parser = {'a2m_79': {'data': data_A2M_TW79,
                          'model': {"1": model_A2M_TW79_out_1 + f"Informer_{30 - args['val_date'] + 1}",
                                    "4": model_A2M_TW79_out_4 + f"Informer_{30 - args['val_date'] + 1}",
                                    "6": model_A2M_TW79_out_6 + f"Informer_{30 - args['val_date'] + 1}"}}}


end_train = datetime(2022, 10, 31)
end_val = datetime(2022, 11, 1)

date_dict = {}

for i in range(30):
    if i == 0:
        end_train = datetime(2022, 10, 31)
        end_val = datetime(2022, 11, 1)
    else:
        end_train = end_train + timedelta(days=(1))
        end_val = end_val + timedelta(days=1)
    date_dict[i+1] = (end_train.strftime('%Y/%m/%d'), end_val.strftime('%Y/%m/%d'))


print('args:', args)

if args['mode'] in data_parser.keys():
    data_info = data_parser[args['mode']]
    model_info = data_info['model']
    data_path = data_info['data']
    model_path = model_info[args['out_len']]
else:
    raise ValueError("mode is a2m_79")



np.random.seed(2023)
train_end = date_dict[args['val_date']][0]
val_end = date_dict[args['val_date']][1]

tw_df = pd.read_csv(data_path)
tw_dataset = TSDataset.load_from_dataframe(
    tw_df,
    time_col='datetime',
    target_cols='wind_speed',
    observed_cov_cols=['wind_direction', 'temp', 'hpa', 'wave_height'],
    freq='H'
)

inputs_hours = int(args['in_len'])
out_chunk_len = int(args['out_len'])
ts_train, ts_remaining = tw_dataset.split(train_end)
ts_val, compare_date = ts_remaining.split(val_end)
actuals, others = compare_date.split(str(pd.to_datetime(val_end) + timedelta(days=1)))
_, ts_inputs = ts_train.split( str(pd.to_datetime(train_end) - timedelta(hours=inputs_hours)))
ts_val = TSDataset.concat([ts_inputs, ts_val])

scaler = MinMaxScaler()
scaler.fit(ts_train)
ts_train_set_scaled = scaler.transform(ts_train)
ts_val_set_scaled = scaler.transform(ts_val)

model = load_model(model_path=model_path)

#  backtest
score, preds_data = backtest(
    data=ts_val_set_scaled,
    model=model,
    metric=mae_paddlets(),  # mae
    predict_window=out_chunk_len,
    stride=out_chunk_len,
    return_predicts=True)

preds_unscaled = scaler.inverse_transform(preds_data)
true = ts_val.target[inputs_hours:]

print("\nBack test\n")
print(score)

mae, mse, rmse, mape, mspe, maePerMean = metric(preds_unscaled, true)
mae, mse, rmse, mape, mspe, maePerMean = round_number(mae, mse, rmse, mape, mspe, maePerMean)


class Note():
    def __init__(self, note_path, val_date, out_len):
        self.validation_date = val_date
        self.prediction_length = out_len
        self.mae = 0
        self.maePerMean = 0
        self.mape = 0
        self.note_path = note_path
    def write_attributes_to_file(self):
        with open(self.note_path, 'a') as f:
            f.write("\n")
            for attribute, value in vars(self).items():
                f.write(f"{attribute}: {value}\n")

note = Note("./note_inference.txt", train_end, out_chunk_len)
note.mae = mae
note.maePerMean = maePerMean
note.mape = mape

print("preds:", preds_unscaled)
print("#"*30)
print("actuals:", true)
print("#"*30)
print("MAE: ", mae)
print("MAE/Mean: ", maePerMean)
print("MAPE: ", mape)
note.write_attributes_to_file()
print("ok")





