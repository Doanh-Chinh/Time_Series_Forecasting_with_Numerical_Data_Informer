import argparse
from datetime import timedelta, datetime

import numpy as np
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

class Training_BackTest:
    def __init__(self, train_end, val_end, sign):
        self.note_path = f"training_backtest/{sign}/history_informer_{args['out_len']}.txt"
        self.n_order = 1
        self.data_path = args['dp']
        self.data_fit = ''
        self.train_end = train_end
        self.val_end = val_end
        self.model_path = ""
        self.scaler_path = ""
        self.in_chunk_len = args['in_len']
        self.out_chunk_len = args['out_len']
        self.start_token_len = args['token_len']
        self.sampling_stride = 1
        self.batch_size = 64
        self.d_model = args['d_model']  # 512
        self.nhead = args['n_head']  # 8
        self.score_back_test = 0
        self.mae = 0
        self.maePerMean = 0
        self.mse = 0
        self.mape = 0
        self.result_path = ""

    def get_number_order(self):
        if os.path.isfile(self.note_path):
            with open(self.note_path, 'r') as f:
                lines = f.readlines()
                if len(lines) > 0:
                    last_line = lines[-1].strip()
                    if last_line.startswith(">>>>>>>>>>>>>>>>>>>>") and last_line.endswith("<<<<<<<<<<<<<<<<<<<<"):
                        self.n_order = int(last_line.split(">>>>>>>>>>>>>>>>>>>>")[-1].split("<<<<<<<<<<<<<<<<<<<<")[0])
                        self.n_order += 1

        return self.n_order

    def write_attributes_to_file(self, model_path, scale_path):
        self.model_path = model_path
        self.scaler_path = scale_path
        with open(self.note_path, 'a') as f:
            f.write("\n")
            for attribute, value in vars(self).items():
                f.write(f"{attribute}: {value}\n")
            f.write(f">>>>>>>>>>>>>>>>>>>>{self.n_order}<<<<<<<<<<<<<<<<<<<<\n")

def rolling_train(ts_train_set, ts_val_set, roll_time, sign):
    '''
    @:return: mape, mae
    '''
    global note
    n_order = note.get_number_order()

    scaler = MinMaxScaler()
    scaler.fit(ts_train_set)
    ts_train_set_scaled = scaler.transform(ts_train_set)
    ts_val_set_scaled = scaler.transform(ts_val_set)

    model = InformerModel(
        in_chunk_len=note.in_chunk_len,
        out_chunk_len=note.out_chunk_len,
        skip_chunk_len=0,
        sampling_stride=note.sampling_stride,
        eval_metrics=["mae", "mse"],
        batch_size=note.batch_size,
        max_epochs=150,
        patience=10,
        d_model=note.d_model,
        nhead=note.nhead,
        start_token_len=note.start_token_len,
        seed=2023,
        callbacks=[History()],
    )
    # model._init_callbacks()
    data_fit = model._update_fit_params([ts_train_set_scaled], [ts_val_set_scaled])
    note.data_fit = data_fit
    print('check tsdataset:', model._check_tsdataset(ts_train_set_scaled))
    print('data fit:', data_fit)

    model.fit(ts_train_set_scaled, ts_val_set_scaled)

    print(model._callbacks[0]._history)
    #  backtest
    score, preds_data = backtest(
        data=ts_val_set_scaled,
        model=model,
        metric=mae_paddlets(),  # mae
        predict_window=note.out_chunk_len,
        stride=note.out_chunk_len,
        return_predicts=True)

    preds_unscaled = scaler.inverse_transform(preds_data)
    true = ts_val_set.target[note.in_chunk_len:]

    print("\nBack test\n")
    print(score)

    mae, mse, rmse, mape, mspe, maePerMean = metric(preds_unscaled, true)
    mae, mse, rmse, mape, mspe, maePerMean = round_number(mae, mse, rmse, mape, mspe, maePerMean)

    note.mae = mae
    note.maePerMean = maePerMean
    note.mse = mse
    note.mape = mape
    note.score_back_test = score

    print("preds:", preds_unscaled)
    print("#"*30)
    print("actuals:", true)

    # paddlets_plot(ts_test, preds_unscaled, fig_path=back_test_f, label="backtest", title=f"{note.model_path[24:]}_MAE_{mae}_MSE_{mse}_MAPE_{mape}", save_fig=True)

    result_path = f'results/{sign}/rolling_out_chunk_{note.out_chunk_len}.xlsx'
    note.result_path = result_path
    append_to_excel(preds_unscaled, true=ts_val_set, excel_path=result_path, target=tw_dataset.target.columns[0], out_chunk_len=f'{note.out_chunk_len}_{roll_time + 1}')

    note.write_attributes_to_file(model_path=f"{model_f}Informer_OutChunk_{args['out_len']}_Rolling_{n_order}", scale_path=f"{scale_f}Informer_OutChunk_{args['out_len']}_Rolling_{n_order}")
    model.save(note.model_path)
    save_scale(scaler, note.scaler_path)

    print("ok")
    return mape, mae, maePerMean

if __name__=="__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dp", required=False, default="a2m_958", help="path to the dataset")
    ap.add_argument("--mp", required=False, default="trained_models/informer/", help="path to the model")
    ap.add_argument("--in_len", type=int, required=False, default=1 * 16,
                    help="number of timesteps to be included in the input, eg. 216 hours")
    ap.add_argument("--out_len", type=int, required=False, default=1,
                    help="out chunk len which model predicts, e.g. '1'")
    ap.add_argument("--token_len", type=int, required=False, default=1 * 8,
                    help="start token length of Informer decoder")
    ap.add_argument("--d_model", type=int, required=False, default=64,
                    help="the expected feature size for the input/output of the informer’s encoder/decoder")
    ap.add_argument("--n_head", type=int, required=False, default=32,
                    help="the number of heads in the multi-head attention mechanism")

    args = vars(ap.parse_args())
    print(args)

    data_A2M_TW79 = 'data/A2M_TW79/TW_0079_data_FillByERA5_Interpolate.csv'
    data_ERA5_30km = 'data/ERA5_30km/35_75_126_5_extract_5years.csv'
    data_A2M_958 = 'data/A2M_958/stn_958_1hour_filledmissing.csv'

    sign = args['dp']
    data_parser = {'era5_30': data_ERA5_30km,
                   'a2m_79': data_A2M_TW79,
                   'a2m_958': data_A2M_958}

    if args['dp'] in data_parser.keys():
        args['dp'] = data_parser[args['dp']]
    else:
        raise ValueError("dp is a2m_79 or era5_30 or a2m_958")


    split_point = ""
    lst_mape = []
    lst_mae = []
    lst_maePerMean = []

    sign = sign.upper()
    model_f = args['mp'] + f"{sign}/"
    scale_f = f"scalers/informer/{sign}/"
    # next_24_f = f"save_figs/informer/next_24_timesteps/{sign}"
    # back_test_f = f"save_figs/informer/back_test/{sign}"
    f = InitialFolder(model_path=model_f, scale_path=scale_f)
    f.update_all_folders()
    f.update_folder(f"training_backtest/{sign}")
    f.update_folder(f'results/{sign}')

    np.random.seed(2023)


    if sign.lower() == 'a2m_79':
        train_end = "2022-11-29 00:00"
        val_end = "2022-11-30 00:00"
        note = Training_BackTest(train_end=train_end, val_end=val_end, sign=sign)

        tw_df = pd.read_csv(note.data_path)
        print(tw_df)
        tw_dataset = TSDataset.load_from_dataframe(
            tw_df,
            time_col='datetime',
            target_cols='wind_speed',
            observed_cov_cols=['wind_direction', 'temp', 'hpa', 'wave_height'],
            freq='H'
        )
    if sign.lower() == 'a2m_958':
        train_end = "2023-06-14 00:00"
        val_end = "2023-07-15 00:00"
        note = Training_BackTest(train_end=train_end, val_end=val_end, sign=sign)

        tw_df = pd.read_csv(note.data_path)
        print(tw_df)
        tw_dataset = TSDataset.load_from_dataframe(
            tw_df,
            time_col='DATETIME',
            target_cols='WS',
            observed_cov_cols=['WD', 'TA'],
            freq='H'
        )
    if sign.lower() == 'era5_30':
        train_end = "2022-11-29 00:00"
        val_end = "2022-11-30 00:00"
        note = Training_BackTest(train_end=train_end, val_end=val_end, sign=sign)

        tw_df = pd.read_csv(note.data_path)
        print(tw_df)
        tw_dataset = TSDataset.load_from_dataframe(
            tw_df,
            time_col='datetime',
            target_cols='wind_speed',
            freq='H'
        )

    print(tw_dataset.columns)
    cols = [col for col in tw_dataset.columns.keys()]

    ts_train, ts_remaining = tw_dataset.split(note.train_end)
    ts_val, compare_date = ts_remaining.split(note.val_end)
    actuals, others = compare_date.split(str(pd.to_datetime(note.val_end) + timedelta(days=1)))
    inputs_hours = note.in_chunk_len

    # check_valid_inputs(ts_val, ts_test, in_chunk_len=note.in_chunk_len, out_chunk_len=note.out_chunk_len)

    print('train shape:', ts_train[cols].shape)
    print('valid shape:', ts_val[cols].shape)

    split_point_train = note.train_end
    split_point_val = note.val_end
    ts_train_part = ts_train
    ts_val_part = ts_val

    for i in range(30):
        if i == 0:
            split_point_train = str(pd.to_datetime(split_point_train))
            split_point_val = str(pd.to_datetime(split_point_val))
        else:
            split_point_train = str(pd.to_datetime(split_point_train) - timedelta(days=1))
            split_point_val = str(pd.to_datetime(split_point_val) - timedelta(days=1))
        print(split_point_val)
        note.train_end = split_point_train
        note.val_end = split_point_val

        ts_train_part, _ = ts_train_part.split(split_point_train)
        ts_val_part, _ = ts_val_part.split(split_point_val)
        _, ts_inputs = ts_train_part.split(str(pd.to_datetime(split_point_train) - timedelta(hours=inputs_hours)))
        ts_val_part = TSDataset.concat([ts_inputs, ts_val_part])
        mape, mae, maePerMean = rolling_train(ts_train_part, ts_val_part, roll_time=i, sign=sign)
        lst_mape.append(mape)
        lst_mae.append(mae)
        lst_maePerMean.append(maePerMean)
    avg_mape = np.mean(lst_mape)
    avg_mae = np.mean(lst_mae)
    avg_maePerMean = np.mean(lst_maePerMean)

    print(f"Average MAPE: {avg_mape}")
    print(f"Average MAE: {avg_mae}")
    print(f"Average MAE/Mean: {avg_maePerMean}")


