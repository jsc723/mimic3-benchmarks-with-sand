# CS598 Deep Learning for Healthcare in Spring 2023 Final Project
## Introduction
The objective of this project is to reproduce some main results in the paper [Attend and Diagnose: Clinical TimeSeries Analysis Using Attention Models](https://arxiv.org/abs/1711.03905). This paper introduces a SAnD model for representing clinical time series data and use it for prediction tasks. The code for this project is develop on top of [YerevaNN/mimic3-benchmarks](https://github.com/YerevaNN/mimic3-benchmarks). The code that I implemented myself for trainig and evaluating the SAnD model can be found in the `mimic3models.in_hospital_mortality.SAnD` module. The implementation of the SAnD model itself is reused from [this repository](https://github.com/khirotaka/SAnD) and can be found at `mimic3models.torch_models.SAnD`.

## Dependencies
You will need to install the following packages before running the code
```
python 3.10.8 or above
CUDA 11.6
PyTorch 1.13.1 for CUDA 11.6
Tensorflow 2.11.1
```

## Download Data

To access the MIMIC-III datasets, you need to go to [Physinet](https://physionet.org/content/mimiciii/1.4/) and finish the [required training](https://physionet.org/about/citi-course/) to become a credential user. Then, you can download the data using the following command in terminal:
```
wget -r -N -c -np --user {your_username} --ask-password https://physionet.org/files/mimiciii/1.4/
```

## Preprocessing
cd to the data directory that contains `*.csv.gz` files, uncompress the csv's first
```
ls *.csv.gz | xargs -I "{}" gzip -d {}
```
Then go to your cloned repository and run the following preprocessing command for the in-hospital-mortality task. (It takes about 2 hours to run)
```
cd /path/to/repository

python -m mimic3benchmark.scripts.extract_subjects /path/to/directory/containing/csv data/root/

python -m mimic3benchmark.scripts.validate_events data/root/

python -m mimic3benchmark.scripts.extract_episodes_from_subjects data/root/

python -m mimic3benchmark.scripts.split_train_and_test data/root/

python -m mimic3benchmark.scripts.create_in_hospital_mortality data/root/ data/in-hospital-mortality/

python -m mimic3models.split_train_val data/in-hospital-mortality
```
## Train the baseline LSTM model
use the following command to train the baseline LSTM model
```
python -um mimic3models.in_hospital_mortality.main --network mimic3models/keras_models/lstm.py --dim 16 --timestep 1.0 --depth 2 --dropout 0.3 --mode train --batch_size 8 --output_dir mimic3models/in_hospital_mortality
```

## Evaluate the baseline LSTM model
After training the LSTM model, use the following command to evaluate it
```
python -um mimic3models.in_hospital_mortality.main --network mimic3models/keras_models/lstm.py --mode test --load_state  mimic3models/in_hospital_mortality/keras_states/{your_model.state}
```

## Train the SAnD model
use the following command to train the SAnD model
```
python -um mimic3models.in_hospital_mortality.SAnD.main
```
This by default train the model for 10 epochs and saves the results to mimic3models/in_hospital_mortality/SAnD/checkpoint_path.
To load from a checkpoint and continue training, use
```
python -um mimic3models.in_hospital_mortality.SAnD.main --use_cache true --load_checkpoint {epoch_number}
```
Note that `--use_cache true` can be used after the first time you train to reduce the time for loading input data. 

## Evaluate the SAnD model
```
python -um mimic3models.in_hospital_mortality.SAnD.main --mode test --load_checkpoint {epoch_number}
```

## Citations
- [Attend and Diagnose: Clinical TimeSeries Analysis Using Attention Models](https://arxiv.org/abs/1711.03905)
- [MIMIC-III, a freely accessible critical care database](https://www.nature.com/articles/sdata201635)
- [YerevaNN/mimic3-benchmarks](https://github.com/YerevaNN/mimic3-benchmarks)
- [khirotaka/SAnD](https://github.com/khirotaka/SAnD)


