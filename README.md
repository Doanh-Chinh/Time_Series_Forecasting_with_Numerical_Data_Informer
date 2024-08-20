
# Time_Series_Forecasting_with_Numerical_Data_Informer
A Deep Learning Approach to Wind Speed Forecasting

## Requirements
- Python >= 3.7
- PaddlePaddle
- PaddleTS

## [Install Guide](https://paddlets.readthedocs.io/en/stable/source/installation/overview.html)

- conda create -n paddlets python=3.9
- python -m pip install paddlepaddle==2.5.0 -i https://pypi.tuna.tsinghua.edu.cn/simple
- python -m pip install paddlets

## Data
- data/A2M_TW79

## Main Files
- TW_ALL_Informer.py
- Model_Inference.py
- utils.py

## Reproducibility
- run Model_Inference.py with anaconda env following script below:
# A2M_TW79
- python Model_Inference.py --mode a2m_79 --out_len 6

## Train model
- run TW_ALL_Informer.py with anaconda env following script below:
# A2M_TW79
- python TW_ALL_Informer.py --dp a2m_79 --in_len 216 --out_len 24

### More parameter information please refer to `TW_ALL_Informer.py`. The detail is listed here:
- "--dp", required=False, default="a2m_79", help="path to the dataset"
- "--mp", required=False, default="trained_models/informer/", help="path to the model"
- "--in_len", type=int, required=False, default=216, help="number of timesteps to be included in the input, eg. 216 hours"
- "--out_len", type=int, required=False, default=24, help="out chunk len which model predicts, e.g. '1'"
- "--token_len", type=int, required=False, default=4*24, help="start token length of Informer decoder"
- "--d_model", type=int, required=False, default=32, help="the expected feature size for the input/output of the informer’s encoder/decoder"
- "--n_head", type=int, required=False, default=4, help="the number of heads in the multi-head attention mechanism"

## More detail about Informer model refering to:
- https://github.com/zhouhaoyi/Informer2020
- https://paddlets.readthedocs.io/en/stable/source/api/paddlets.models.forecasting.dl.informer.html

