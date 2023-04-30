from __future__ import absolute_import
from __future__ import print_function

import numpy as np
import argparse
import os
import imp
import re
import tqdm
import sys
import psutil
import time

from mimic3models.in_hospital_mortality import utils
from mimic3benchmark.readers import InHospitalMortalityReader

from mimic3models.preprocessing import Discretizer, Normalizer
from mimic3models import metrics
from mimic3models import keras_utils
from mimic3models import common_utils

from keras.callbacks import ModelCheckpoint, CSVLogger

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from sklearn.metrics import roc_auc_score, average_precision_score

from mimic3models.torch_models.SAnD.core.model import SAnD

pid = os.getpid()
process = psutil.Process(pid)
prev_memory_usage = process.memory_info().rss / 1e6
def get_memory_usage():
    global prev_memory_usage
    prev_memory_usage = process.memory_info().rss / 1e6
    return prev_memory_usage
def get_memory_usage_diff():
    prev = prev_memory_usage
    return get_memory_usage() - prev


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
common_utils.add_common_arguments(parser)
parser.add_argument('--data', type=str, help='Path to the data of in-hospital mortality task',
                    default=os.path.join(os.path.dirname(__file__), '../../../data/in-hospital-mortality/'))
parser.add_argument('--output_dir', type=str, help='Directory relative which all output files are stored',
                    default='.')
parser.add_argument('--use_cache', help='use cached numpy array from temp')

parser.add_argument('--load_checkpoint', default=None)
arg_list = sys.argv[1:] + ['--network', 'none']
args = parser.parse_args(arg_list)
print(args)
print('--------------------')

dir_name = os.path.dirname(os.path.dirname(__file__))
temp_dir = os.path.join(dir_name, 'temp')
checkpoint_path = os.path.join(dir_name, 'checkpoint_path')
checkpoint_prefix = 'cp_'

if not os.path.exists(temp_dir):
    os.makedirs(temp_dir)
if not os.path.exists(checkpoint_path):
    os.makedirs(checkpoint_path)


# Build readers, discretizers, normalizers
train_reader = InHospitalMortalityReader(dataset_dir=os.path.join(args.data, 'train'),
                                         listfile=os.path.join(args.data, 'train_listfile.csv'),
                                         period_length=48.0)

val_reader = InHospitalMortalityReader(dataset_dir=os.path.join(args.data, 'train'),
                                       listfile=os.path.join(args.data, 'val_listfile.csv'),
                                       period_length=48.0)

discretizer = Discretizer(timestep=float(args.timestep),
                          store_masks=True,
                          impute_strategy='previous',
                          start_time='zero')

discretizer_header = discretizer.transform(train_reader.read_example(0)["X"])[1].split(',')
cont_channels = [i for (i, x) in enumerate(discretizer_header) if x.find("->") == -1]

normalizer = Normalizer(fields=cont_channels)  # choose here which columns to standardize
normalizer_state = args.normalizer_state
if normalizer_state is None:
    normalizer_state = 'ihm_ts{}.input_str:{}.start_time:zero.normalizer'.format(args.timestep, args.imputation)
    normalizer_state = os.path.join(os.path.dirname(__file__), "..", normalizer_state)
normalizer.load_params(normalizer_state)

def get_data():
    if args.use_cache:
        train_x = np.load(os.path.join(temp_dir, 'train_x.npy'))
        train_y = np.load(os.path.join(temp_dir, 'train_y.npy'))
        val_x = np.load(os.path.join(temp_dir, 'val_x.npy'))
        val_y = np.load(os.path.join(temp_dir, 'val_y.npy'))
        print('loaded from ./temp/')


    else:
        # Read data
        train_raw = utils.load_data(train_reader, discretizer, normalizer, args.small_part)
        val_raw = utils.load_data(val_reader, discretizer, normalizer, args.small_part)

        train_x, train_y = train_raw[0], np.array(train_raw[1])
        val_x, val_y = val_raw[0], np.array(val_raw[1])
        
        np.save(os.path.join(temp_dir, 'train_x.npy'), train_x)
        np.save(os.path.join(temp_dir, 'train_y.npy'), train_y)
        np.save(os.path.join(temp_dir, 'val_x.npy'), val_x)
        np.save(os.path.join(temp_dir, 'val_y.npy'), val_y)
        print('data saved to ./temp/')
    
    train_x = torch.from_numpy(train_x).type(torch.FloatTensor)
    train_y = torch.from_numpy(train_y).type(torch.LongTensor)
    val_x = torch.from_numpy(val_x).type(torch.FloatTensor)
    val_y = torch.from_numpy(val_y).type(torch.LongTensor)
    return train_x, train_y, val_x, val_y

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def main():
    # Build model

    in_feature = 76
    seq_len = 48
    n_heads = 8
    factor = 12 #12
    num_class = 2
    num_layers = 4 #4
    d_model = 16 #256
    dropout_rate = 0.3
    BATCH_SIZE = 256

    model = SAnD(in_feature, seq_len, n_heads, factor, num_class, num_layers, d_model, dropout_rate)
    model = model.to(device)
    print(model)
    print("number of parameters: ", count_parameters(model))

    global checkpoint_prefix
    checkpoint_prefix = f'cp_n{num_layers}_m{factor}_d{d_model}'

    start_epoch = 1
    if args.load_checkpoint:
        start_epoch = int(args.load_checkpoint) + 1
        checkpoint_file = os.path.join(checkpoint_path, f'{checkpoint_prefix}_{start_epoch-1}.pt')
        state_dict = torch.load(checkpoint_file)
        model.load_state_dict(state_dict)
    num_of_epoch = 30
    save_freq = 5

    if args.mode == 'train':
        train_x, train_y, val_x, val_y = get_data()

        print(train_x.shape, train_y.shape)
        print(val_x.shape, val_y.shape)
        print(train_x.dtype, train_y.dtype)
        print(val_x.dtype, val_y.dtype)

        train_ds = TensorDataset(train_x, train_y)
        val_ds = TensorDataset(val_x, val_y)

        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE)
        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

        criterion = nn.CrossEntropyLoss()
        optimizer_config={"lr": 0.0005, "betas": (0.9, 0.98)}
        optimizer = optim.Adam(model.parameters(), **optimizer_config)

        train(model, train_loader, val_loader, criterion, optimizer,
            start_epoch, num_of_epoch, checkpoint_path, save_freq)

    elif args.mode == 'test':

        test_reader = InHospitalMortalityReader(dataset_dir=os.path.join(args.data, 'test'),
                                                listfile=os.path.join(args.data, 'test_listfile.csv'),
                                                period_length=48.0)
        ret = utils.load_data(test_reader, discretizer, normalizer, args.small_part,
                                return_names=True)

        test_x = ret["data"][0]
        test_y = np.array(ret["data"][1])

        test_x = torch.from_numpy(test_x).type(torch.FloatTensor)
        test_y = torch.from_numpy(test_y).type(torch.LongTensor)

        test_ds = TensorDataset(test_x, test_y)

        test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)

        test(model, test_loader)





def train(model, train_loader, val_loader, criterion, optimizer,
         start_epoch, epochs, save_to_checkpoint_path: str = None, save_freq = 5):

    len_of_train_dataset = len(train_loader.dataset)
    len_of_val_dataset = len(val_loader.dataset)
    max_epoch = start_epoch + epochs - 1
    prev_auroc = 0.0

    # Start the timer
    start_time = time.time()

    for epoch in range(start_epoch, max_epoch + 1):
        
        correct = 0.0
        total = 0.0
        
        model.train()
        total_loss = 0.0
        pbar = tqdm.tqdm(total=len_of_train_dataset)
        for x, y in train_loader:
            b_size = y.shape[0]
            total += y.shape[0]
            x = x.to(device) if isinstance(x, torch.Tensor) else [i.to(device) for i in x]
            y = y.to(device)

            pbar.set_description(
                "\033[36m" + "Training" + "\033[0m" + " - Epochs: {:03d}/{:03d}".format(epoch, max_epoch)
            )
            pbar.update(b_size)

            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

            _, predicted = torch.max(outputs, 1)
            correct += (predicted == y).sum().float().cpu().item()

            total_loss += loss.cpu().item()

        print(f'epoch: {epoch}, loss: {total_loss}, acc: {float(correct / total)}')
        print(f'memory usage: {get_memory_usage()} MB')
        print(f"Time used: {time.time() - start_time:.2f} seconds")

        # Validation
        with torch.no_grad():
            val_correct = 0.0
            val_total = 0.0

            y_true = []
            y_scores = []

            model.eval()
            for x_val, y_val in val_loader:
                val_total += y_val.shape[0]
                x_val = x_val.to(device) if isinstance(x_val, torch.Tensor) else [i_val.to(device) for i_val in x_val]
                y_val = y_val.to(device)

                val_output = model(x_val)
                val_loss = criterion(val_output, y_val)
                _, val_pred = torch.max(val_output, 1)
                val_correct += (val_pred == y_val).sum().float().cpu().item()

                y_true.append(y_val)
                val_score = torch.softmax(val_output, dim=1)[:, 1] # get score of the positive class
                y_scores.append(val_score)
            y_true = torch.cat(y_true)
            y_scores = torch.cat(y_scores)

            auroc = roc_auc_score(y_true.numpy(), y_scores.numpy())
            auprc = average_precision_score(y_true.numpy(), y_scores.numpy())

            print(f'epoch: {epoch}, val acc: {float(val_correct / val_total)}, auroc: {auroc}, auprc: {auprc}')

        pbar.close()

        if save_to_checkpoint_path and (epoch % save_freq == 0 or prev_auroc < auroc):
            path_to_save = os.path.join(save_to_checkpoint_path, f'{checkpoint_prefix}_{epoch}.pt')
            torch.save(model.state_dict(), path_to_save)
            print(f'saved to {path_to_save}')
        prev_auroc = auroc
    print(f"Total time used: {time.time() - start_time:.2f} seconds")


def test(model, test_loader):
    with torch.no_grad():
        test_correct = 0.0
        test_total = 0.0

        y_true = []
        y_scores = []

        model.eval()
        for x_val, y_val in test_loader:
            test_total += y_val.shape[0]
            x_val = x_val.to(device) if isinstance(x_val, torch.Tensor) else [i_val.to(device) for i_val in x_val]
            y_val = y_val.to(device)

            test_output = model(x_val)
            _, test_pred = torch.max(test_output, 1)
            test_correct += (test_pred == y_val).sum().float().cpu().item()

            y_true.append(y_val)
            test_score = torch.softmax(test_output, dim=1)[:, 1] # get score of the positive class
            y_scores.append(test_score)
        y_true = torch.cat(y_true)
        y_scores = torch.cat(y_scores)

        auroc = roc_auc_score(y_true.numpy(), y_scores.numpy())
        auprc = average_precision_score(y_true.numpy(), y_scores.numpy())

        print(f'test acc: {float(test_correct / test_total)}, auroc: {auroc}, auprc: {auprc}')
main()
