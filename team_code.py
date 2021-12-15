#!/usr/bin/env python

# Edit this script to add your team's training code.
# Some functions are *required*, but you can edit most parts of the required functions, remove non-required functions, and add your own functions.

################################################################################
#
# Imported functions and variables
#
################################################################################

# Import functions. These functions are not required. You can change or remove them.
from helper_code import *
import json
import copy
import numpy as np, os, sys, joblib
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
import argparse
from parse_config import ConfigParser
import torch.nn as nn
from model_training.training import *
from utils.loss import AsymmetricLossOptimized

import utils.lr_scheduler as custom_lr_scheduler
from utils.metric import ChallengeMetric
import classifier.se_resnet as module_arch_se_resnet
from model_training.utils import stratification, make_dirs, init_obj, get_logger, get_mnt_mode, to_np, save_checkpoint

# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True
np.random.seed(SEED)

# Setup Cuda
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

# model selection
files_models = {
    "inceptiontime": ['InceptionTimeV1', 'InceptionTimeV2'],
    "resnest": ['resnest50', 'resnest'],
    "resnet": ['resnet'],
    "swin_transformer": ['swin_transformer'],
    "beat_aligned_transformer": ['beat_aligned_transformer'],
    "beat_aligned_cnn_transformer": ['beat_aligned_cnn_transformer'],
    "beat_aligned_cnn": ['beat_aligned_cnn'],
    "nesT": ["nesT"],
    "se_resnet": ['se_resnet']
}

my_classes = []
log_step = 1

# Define the Challenge lead sets. These variables are not required. You can change or remove them.
twelve_leads = ('I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6')
six_leads = ('I', 'II', 'III', 'aVR', 'aVL', 'aVF')
four_leads = ('I', 'II', 'III', 'V2')
three_leads = ('I', 'II', 'V2')
two_leads = ('I', 'II')
lead_sets = (twelve_leads, six_leads, four_leads, three_leads, two_leads)


################################################################################
#
# Training model function
#
################################################################################

# Train your model. This function is *required*. You should edit this function to add your code, but do *not* change the arguments of this function.
def training_code(data_directory, model_directory):

    try:
        # split into training and validation
        split_idx = 'model_training/split.mat'
        fine_tuning_split_idx = 'model_training/fine_tuning_split.mat'
        stratification(data_directory)

        # json files
        training_root = 'model_training/'

        configs = ['train.json']
        # configs = ['train_12leads.json']
        challenge_dataset = ChallengeDataLoaderCV(data_directory, split_idx, window_size=5000, resample_Fs=500)
        train_dataset, val_dataset = challenge_dataset.train_dataset, challenge_dataset.val_dataset
        # domain_train_dataset, domain_val_dataset = domain_dataset.train_dataset, domain_dataset.val_dataset
        for config_json_path in configs:
            train_model(training_root + config_json_path, split_idx, data_directory, model_directory, train_dataset,
                        val_dataset)

            # domain_classification_model(training_root + config_json_path, fine_tuning_split_idx, data_directory, model_directory, domain_train_dataset,
            #                             domain_val_dataset)
    except:
        print('Done')


def train_model(config_json, split_idx, data_directory, model_directory, train_dataset, val_dataset):
    # Get training configs
    with open(config_json, 'r', encoding='utf8')as fp:
        config = json.load(fp)
    lead_number = config['data_loader']['args']['lead_number']
    assert config['arch']['args']['channel_num'] == lead_number
    # Data_loader
    train_dataset.lead_number = lead_number
    val_dataset.lead_number = lead_number
    print("batch_size: ", config['data_loader']['args']['batch_size'])
    train_loader = ChallengeDataLoader(train_dataset, val_dataset,
                                       batch_size=config['data_loader']['args']['batch_size'])
    if lead_number == 8:
        lead_number = 12
    # Paths to save log, checkpoint, tensorboard logs and results
    base_dir = 'model_training/training_results'
    result_dir, log_dir, checkpoint_dir, tb_dir = make_dirs(base_dir)

    # Build model architecture
    # global model
    for file, types in files_models.items():
        for type in types:
            if config["arch"]["type"] == type:
                model = init_obj(config, 'arch', eval("module_arch_" + file))
    model.to(device)
    # Logger for train
    logger = get_logger(log_dir + '/info_lead_' + str(lead_number) + '.log', name='train')
    logger.info(config["arch"]["type"])
    # Tensorboard
    # train_writer = SummaryWriter(tb_dir + '/train_lead_' + str(lead_number))
    # val_writer = SummaryWriter(tb_dir + '/valid_' + str(lead_number))

    valid_loader = train_loader.valid_data_loader

    header_files = my_find_challenge_files(data_directory)
    classes = set()
    for header_file in header_files:
        header = load_header(header_file)
        classes |= set(get_labels(header))
    if all(is_integer(x) for x in classes):
        classes = sorted(classes, key=lambda x: int(x))  # Sort classes numerically if numbers.
    else:
        classes = sorted(classes)  # Sort classes alphanumerically if not numbers.
    num_classes = len(classes)
    train_loader.all_classes = classes

    # ### for test
    # config_json = 'model_training/train_2leads.json'
    # with open(config_json, 'r', encoding='utf8')as fp:
    #     config = json.load(fp)
    # checkpoint_path = model_directory + '/lead_12_model_best.pth'
    # model = load_my_model(config, checkpoint_path)

    # Get function handles of loss and metrics
    # criterion = getattr(modules, config['loss']['type'])
    criterion = AsymmetricLossOptimized()
    # criterion = torch.nn.BCEWithLogitsLoss(reduction='none')

    # Get function handles of metrics
    train_challenge_metric = ChallengeMetric()
    val_challenge_metric = ChallengeMetric()

    # Build optimizer, learning rate scheduler
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())

    optimizer = init_obj(config, 'optimizer', torch.optim, trainable_params)

    lr_scheduler = init_obj(config, 'lr_scheduler', custom_lr_scheduler, optimizer)

    # Begin training process
    trainer = config['trainer']
    epochs = trainer['epochs']
    # epochs = 1

    # Full train and valid logic
    mnt_metric_name, mnt_mode, mnt_best, early_stop = get_mnt_mode(trainer)
    not_improved_count = 0

    for epoch in range(epochs):
        best = False
        train_loss, train_metric = train(model, optimizer, train_loader, criterion, train_challenge_metric, epoch,
                                         device=device)
        val_loss, val_metric = valid(model, valid_loader, criterion, val_challenge_metric, device=device)

        lr_scheduler.step()

        logger.info(
            'Epoch:[{}/{}]\t {:10s}: {:.5f}\t {:10s}: {:.5f}'.format(epoch, epochs, 'loss', train_loss, 'metric',
                                                                     train_metric))
        logger.info(
            '             \t {:10s}: {:.5f}\t {:10s}: {:.5f}'.format('val_loss', val_loss, 'val_metric', val_metric))
        logger.info('             \t learning_rate: {}'.format(optimizer.param_groups[0]['lr']))

        # check whether model performance improved or not, according to specified metric(mnt_metric)
        if mnt_mode != 'off':
            mnt_metric = val_loss if mnt_metric_name == 'val_loss' else val_metric
            improved = (mnt_mode == 'min' and mnt_metric <= mnt_best) or \
                       (mnt_mode == 'max' and mnt_metric >= mnt_best)
            if improved:
                mnt_best = mnt_metric
                not_improved_count = 0
                best = True
            else:
                not_improved_count += 1

            if not_improved_count > early_stop:
                logger.info("Validation performance didn\'t improve for {} epochs. Training stops.".format(early_stop))
                break
        file_name = 'lead_' + str(lead_number) + '_pretrain_model_best.pth'
        # save_checkpoint(model, epoch, mnt_best, checkpoint_dir, file_name, save_best=False)
        if best == True:
            save_checkpoint(model, epoch, mnt_best, model_directory, file_name, train_loader.all_classes, leads_num=lead_number,
                            config_json=config_json, save_best=True)
            logger.info("Saving current best: {}".format(file_name))

        # Tensorboard log
        # train_writer.add_scalar('loss', train_loss, epoch)
        # train_writer.add_scalar('metric', train_metric, epoch)
        # train_writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], epoch)
        #
        # val_writer.add_scalar('loss', val_loss, epoch)
        # val_writer.add_scalar('metric', val_metric, epoch)
    del model, train_loader, logger, valid_loader


################################################################################
#
# Running trained model function
#
################################################################################

# Run your trained model. This function is *required*. You should edit this function to add your code, but do *not* change the arguments of this function.

def get_pred(outputs, alpha=0.5):
    for i in range(outputs.shape[0]):
        for j in range(outputs.shape[1]):
            if outputs[i, j] >= alpha:
                outputs[i, j] = 1
            else:
                outputs[i, j] = 0
        # if outputs[i, -5] == 1:
        #     outputs[i, 14] = 1
        if outputs[i, :].sum() == 0:
            # print("postprocess this sample as NSR")
            outputs[i, 14] = 1
        if outputs[i, -1] == 1 and outputs[i, -2] != 1:
            print("postprocess: relabeled this sample as TAb")
            outputs[i, -2] = 1
    return outputs

def run_my_model(model_list, header, recording, config_path):
    model_500_1, model_500_2, model_250_1, model_250_2 = model_list[0], model_list[1], model_list[2], model_list[3]
    recording[np.isnan(recording)] = 0
    recording = np.array(recording, dtype=float)
    lead_num = len(recording)
    with open(config_path, 'r', encoding='utf8')as fp:
        config = json.load(fp)

    # divide ADC_gain and resample
    resample_Fs = 500
    window_size = 4992
    header = header.split('\n')
    recording1 = copy.deepcopy(recording)
    recording1 = resample(recording1, header, resample_Fs)

    # to filter and detrend samples
    # recording1 = filter_and_detrend(recording1)

    n_segment = 1
    # slide and cut
    recording1 = slide_and_cut(recording1, n_segment, window_size, resample_Fs, test_time_aug=True)

    data = np.zeros((len(recording1), 8, 4992)) # i, ii, v1, v2, v3, v4, v5, v6

    recording1[np.isnan(recording1)] = 0
    if lead_num == 12:
        leads_index = [0, 1, 6, 7, 8, 9, 10, 11]
        data[:, :, :] = recording1[:, leads_index, :]
    elif lead_num == 6 or lead_num == 2:
        leads_index = [0,1]
        data[:, leads_index, :] = recording1[:, leads_index, :]
    elif lead_num == 4: # i, ii, iii, v2 -> i,ii,_,v2
        leads_index = [0,1,3]
        data[:, leads_index, :] = recording1[:, leads_index, :]
    elif lead_num == 3: # i, ii, v2 -> i, ii, _, v2
        data[:, [0,1,3], :] = recording1[:, :, :]

    data = torch.tensor(data)
    data = data.to(device, dtype=torch.float)
    output_1 = model_500_1(data)
    prediction_1 = torch.sigmoid(output_1)
    prediction_1 = prediction_1.detach().cpu().numpy()
    prediction_1 = np.expand_dims(np.max(prediction_1, axis=0), axis=0)

    output_2 = model_500_2(data)
    prediction_2 = torch.sigmoid(output_2)
    prediction_2 = prediction_2.detach().cpu().numpy()
    prediction_2 = np.expand_dims(np.max(prediction_2, axis=0), axis=0)

    # divide ADC_gain and resample
    resample_Fs = 250
    window_size = 4992
    # header = header.split('\n')
    recording1 = copy.deepcopy(recording)
    recording1 = resample(recording1, header, resample_Fs)

    # to filter and detrend samples
    # recording1 = filter_and_detrend(recording1)

    n_segment = 1
    # slide and cut
    recording1 = slide_and_cut(recording1, n_segment, window_size, resample_Fs, test_time_aug=True)

    data = np.zeros((len(recording1), 8, 4992))  # i, ii, v1, v2, v3, v4, v5, v6

    recording1[np.isnan(recording1)] = 0
    if lead_num == 12:
        leads_index = [0, 1, 6, 7, 8, 9, 10, 11]
        data[:, :, :] = recording1[:, leads_index, :]
    elif lead_num == 6 or lead_num == 2:
        leads_index = [0, 1]
        data[:, leads_index, :] = recording1[:, leads_index, :]
    elif lead_num == 4:  # i, ii, iii, v2 -> i,ii,_,v2
        leads_index = [0, 1, 3]
        data[:, leads_index, :] = recording1[:, leads_index, :]
    elif lead_num == 3:  # i, ii, v2 -> i, ii, _, v2
        data[:, [0, 1, 3], :] = recording1[:, :, :]

    data = torch.tensor(data)
    data = data.to(device, dtype=torch.float)
    output_3 = model_250_1(data)
    prediction_3 = torch.sigmoid(output_3)
    prediction_3 = prediction_3.detach().cpu().numpy()
    prediction_3 = np.expand_dims(np.max(prediction_3, axis=0), axis=0)

    output_4 = model_250_2(data)
    prediction_4 = torch.sigmoid(output_4)
    prediction_4 = prediction_4.detach().cpu().numpy()
    prediction_4 = np.expand_dims(np.max(prediction_4, axis=0), axis=0)

    prediction = np.concatenate([prediction_1, prediction_2, prediction_3, prediction_4], axis=0)
    prediction_out = np.mean(prediction, axis=0)
    y_pred = get_pred(prediction, alpha=0.5)
    pred_sum = y_pred[0] + y_pred[1] + y_pred[2] + y_pred[3]
    y_pred = np.where(pred_sum >= 2, 1, 0)

    classes = "164889003,164890007,6374002,426627000,733534002,713427006,270492004,713426002,39732003,445118002,164947007,251146004,111975006,698252002,426783006,63593006,10370003,365413008,427172004,164917005,47665007,427393009,426177001,427084000,164934002,59931005"
    ### equivalent SNOMED CT codes merged, noted as the larger one
    classes = classes.split(',')
    all_classes = "164889003,164890007,6374002,426627000,733534002,164909002,713427006,59118001,270492004,713426002,39732003,445118002,164947007,251146004,111975006,698252002,426783006,284470004,63593006,10370003,365413008,427172004,17338001,164917005,47665007,427393009,426177001,427084000,164934002,59931005"
    all_classes = all_classes.split(',')


    label = np.zeros((26,), dtype=int)
    indexes = np.where(y_pred > 0.5)
    label[indexes] += 1

    label_output = np.zeros((len(all_classes),), dtype=int)
    prediction_output = np.zeros((len(all_classes),))

    equivalent_classes = {
        "733534002": "164909002",
        "713427006": "59118001",
        "63593006": "284470004",
        "427172004": "17338001"
    }
    for i in range(len(classes)):
        dx = classes[i]
        ind = all_classes.index(dx)
        label_output[ind] = label[i]
        prediction_output[ind] = prediction_out[i]
        if dx == "733534002" or dx == "713427006" or dx == "63593006" or dx == "427172004":
            dx2 = equivalent_classes[dx]
            ind = all_classes.index(dx2)
            label_output[ind] = label[i]
            prediction_output[ind] = prediction_out[i]
        # if dx == "733534002" or dx == "713427006":
        #     ind = all_classes.index("6374002")
        #     if label[i] == 1:
        #         label_output[ind] = label[i]
        #         prediction_output[ind] = prediction[i]
    # for dx2 in ["6374002"]:
    #     label_output[all_classes.index(dx2)] = 0
    #     prediction_output[all_classes.index(dx2)] = 0
    # label_output[all_classes.index("426783006")] = (label_output[all_classes.index("426783006")] > threshold) | (
    #         (label_output > threshold).sum() == 0)
    # try:
    #     label_output[all_classes.index("164934002")] = (label_output[all_classes.index("164934002")] > threshold) | (
    #             label_output[all_classes.index("59931005")] > threshold)
    # except:
    #     print('Exception')
    return all_classes, label_output, prediction_output


################################################################################
#
# File I/O functions
#
################################################################################

# Save a trained model. This function is not required. You can change or remove it.
def save_model(model_directory, leads, classes, imputer, classifier):
    d = {'leads': leads, 'classes': classes, 'imputer': imputer, 'classifier': classifier}
    filename = os.path.join(model_directory, get_model_filename(leads))
    joblib.dump(d, filename, protocol=0)


# Load a trained model. This function is *required*. You should edit this function to add your code, but do *not* change the arguments of this function.
def load_model(model_directory, leads):
    leads_num_to_configs = {
        "2": "model_training/train_2leads.json",
        "3": "model_training/train_3leads.json",
        "4": "model_training/train_4leads.json",
        "6": "model_training/train_6leads.json",
        "12": "model_training/train_12leads.json"
    }
    leads_num = len(leads)

    config_path = 'model_training/train.json'
    print("current leads_num: ", leads)
    config_json = config_path
    with open(config_json, 'r', encoding='utf8')as fp:
        config = json.load(fp)
    global current_config_json
    global current_leads
    current_leads = leads_num
    current_config_json = config_json
    model_500_1 = load_my_model(config, model_directory + '/model_best_500hz_1.pth')
    model_500_2 = load_my_model(config, model_directory + '/model_best_500hz_2.pth')
    model_250_1 = load_my_model(config, model_directory + '/model_best_250hz_1.pth')
    model_250_2 = load_my_model(config, model_directory + '/model_best_250hz_2.pth')

    return [model_500_1, model_500_2, model_250_1, model_250_2]


def load_my_model(config, checkpoint_path=None):
    for file, types in files_models.items():
        for type in types:
            if config["arch"]["type"] == type:
                model = init_obj(config, 'arch', eval("module_arch_" + file))

    model.to(device)

    if checkpoint_path is not None:
        print(checkpoint_path)
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['state_dict'])
        global my_classes
        # my_classes = checkpoint["classes"]

    model.eval()
    return model


# Define the filename(s) for the trained models. This function is not required. You can change or remove it.
def get_model_filename(leads):
    sorted_leads = sort_leads(leads)
    return 'model_' + '-'.join(sorted_leads) + '.sav'


################################################################################
#
# Feature extraction function
#
################################################################################

# Extract features from the header and recording. This function is not required. You can change or remove it.
def get_features(header, recording, leads):
    # Extract age.
    age = get_age(header)
    if age is None:
        age = float('nan')

    # Extract sex. Encode as 0 for female, 1 for male, and NaN for other.
    sex = get_sex(header)
    if sex in ('Female', 'female', 'F', 'f'):
        sex = 0
    elif sex in ('Male', 'male', 'M', 'm'):
        sex = 1
    else:
        sex = float('nan')

    # Reorder/reselect leads in recordings.
    recording = choose_leads(recording, header, leads)

    # Pre-process recordings.
    adc_gains = get_adc_gains(header, leads)
    baselines = get_baselines(header, leads)
    num_leads = len(leads)
    for i in range(num_leads):
        recording[i, :] = (recording[i, :] - baselines[i]) / adc_gains[i]

    # Compute the root mean square of each ECG lead signal.
    rms = np.zeros(num_leads)
    for i in range(num_leads):
        x = recording[i, :]
        rms[i] = np.sqrt(np.sum(x ** 2) / np.size(x))

    return age, sex, rms


def train(model, optimizer, train_loader, criterion, metric, epoch, device=None):
    sigmoid = nn.Sigmoid()
    model.train()
    cc = 0
    Loss = 0
    total = 0
    batchs = 0
    for batch_idx, (data, target, class_weights) in enumerate(train_loader):
        batch_start = time.time()
        data, target, class_weights = data.to(device, dtype=torch.float), target.to(device, dtype=torch.float), class_weights.to(device,
                                                                                                                                 dtype=torch.float)
        # target_coarse = target_coarse.to(device)
        optimizer.zero_grad()
        output = model(data)

        loss = criterion(output, target) * class_weights
        loss = torch.mean(loss)
        loss.backward()
        optimizer.step()

        prediction = to_np(sigmoid(output), device)
        prediction = metric.get_pred(prediction, alpha=0.5)
        target = to_np(target, device)
        c = metric.challenge_metric(prediction, target)
        cc += c
        Loss += float(loss)
        total += 1
        batchs += 1

        ### for debug
        # if total > 50:
        #     break

        if batch_idx % log_step == 0:
            batch_end = time.time()
            # logger.debug('Epoch: {} {} Loss: {:.6f}, 1 batch cost time {:.2f}'.format(epoch, batch_idx, loss.item(),
            #                                                                           batch_end - batch_start))
            print('Train Epoch: {} {} Loss: {:.6f}, 1 batch cost time {:.2f}'.format(epoch,
                                                                                     batch_idx,
                                                                                     loss.item(),
                                                                                     batch_end - batch_start))

    return Loss / total, cc / batchs


def valid(model, valid_loader, criterion, metric, device=None):
    sigmoid = nn.Sigmoid()
    model.eval()
    cc = 0
    Loss = 0
    total = 0
    batchs = 0
    with torch.no_grad():
        for batch_idx, (data, target, class_weights) in enumerate(valid_loader):
            data, target, class_weights = data.to(device, dtype=torch.float), target.to(device, dtype=torch.float), class_weights.to(device,
                                                                                                                                     dtype=torch.float)
            # target_coarse = target_coarse.to(device)
            output = model(data)

            loss = criterion(output, target) * class_weights
            loss = torch.mean(loss)
            # loss = (loss_coarse + loss) / 2

            prediction = to_np(sigmoid(output), device)
            prediction = metric.get_pred(prediction, alpha=0.5)
            target = to_np(target, device)
            c = metric.challenge_metric(prediction, target)
            cc += c
            Loss += loss
            total += 1
            batchs += 1

    return Loss / total, cc / batchs


def train_domain(model, optimizer, train_loader, criterion, metric, epoch, device=None):
    sigmoid = nn.Sigmoid()
    model.train()
    cc = 0
    Loss = 0
    total = 0
    batchs = 0
    for batch_idx, (data, target, class_weights) in enumerate(train_loader):
        batch_start = time.time()
        data, target, class_weights = data.to(device, dtype=torch.float), target.to(device, dtype=torch.float), class_weights.to(device,
                                                                                                                                 dtype=torch.float)
        # target_coarse = target_coarse.to(device)
        optimizer.zero_grad()
        output = model(data)

        loss = criterion(output, target) * class_weights
        loss = torch.mean(loss)
        loss.backward()
        optimizer.step()

        prediction = to_np(sigmoid(output), device)
        prediction = metric.get_pred(prediction, alpha=0.5)
        target = to_np(target, device)
        c = metric.accuracy(prediction, target)
        cc += c
        Loss += float(loss)
        total += 1
        batchs += 1

        ### for debug
        # if total > 50:
        #     break

        if batch_idx % log_step == 0:
            batch_end = time.time()
            # logger.debug('Epoch: {} {} Loss: {:.6f}, 1 batch cost time {:.2f}'.format(epoch, batch_idx, loss.item(),
            #                                                                           batch_end - batch_start))
            print('Train Epoch: {} {} Loss: {:.6f}, 1 batch cost time {:.2f}'.format(epoch,
                                                                                     batch_idx,
                                                                                     loss.item(),
                                                                                     batch_end - batch_start))

    return Loss / total, cc / batchs


def valid_domain(model, valid_loader, criterion, metric, device=None):
    sigmoid = nn.Sigmoid()
    model.eval()
    cc = 0
    Loss = 0
    total = 0
    batchs = 0
    with torch.no_grad():
        for batch_idx, (data, target, class_weights) in enumerate(valid_loader):
            data, target, class_weights = data.to(device, dtype=torch.float), target.to(device, dtype=torch.float), class_weights.to(device,
                                                                                                                                     dtype=torch.float)
            # target_coarse = target_coarse.to(device)
            output = model(data)

            loss = criterion(output, target) * class_weights
            loss = torch.mean(loss)
            # loss = (loss_coarse + loss) / 2

            prediction = to_np(sigmoid(output), device)
            prediction = metric.get_pred(prediction, alpha=0.5)
            target = to_np(target, device)
            c = metric.accuracy(prediction, target)
            cc += c
            Loss += loss
            total += 1
            batchs += 1

    return Loss / total, cc / batchs


def run_model(model, header, recording):
    # classes = model['classes']
    # leads = model['leads']
    config_json = current_config_json
    return run_my_model(model, header, recording, config_json)
