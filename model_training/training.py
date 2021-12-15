import torch
from torch.utils.data import DataLoader
from torch.utils import data
from torch.utils.data.dataloader import default_collate
import numpy as np
import time
from model_training.utils import loadmat, CustomTensorDataset, load_weights, load_labels, resample, slide_and_cut, load_challenge_data
from model_training.util import my_find_challenge_files
import os
from utils.denoising import filter_and_detrend
from torchvision import datasets, transforms
import model_training.transformers as module_transformers

class CustomDataset(data.Dataset):
    """
    PyTorch Dataset generator class
    Ref: https://stanford.edu/~shervine/blog/pytorch-how-to-generate-data-parallel
    """

    def __init__(self, label_files, labels_onehot, label_dir, leads_index, name_list_full=[], transform=None, sample_rate=500, to_get_feature=False, rep_data=[]):
        """Initialization"""
        self.file_names_list = label_files
        self.label_dir = label_dir
        self.labels_onehot = labels_onehot
        self.class_weights = self.get_class_weights()
        self.leads_index = leads_index
        self.transform = transform
        self.to_get_feature = to_get_feature
        self.sample_rate = sample_rate
        # self.normalization = TNormalize()
        self.feature = rep_data

    def __len__(self):
        """Return total number of data samples"""
        return len(self.file_names_list)

    def __getitem__(self, idx):
        """Generate data sample"""
        sample_file_name = self.file_names_list[idx]
        # header_file_name = self.file_names_list[idx][:-3] + "hea"

        label = self.labels_onehot[idx]
        recording, header, name = load_challenge_data(sample_file_name, self.label_dir)

        # get class_weight by name
        class_weight, data_source = self.get_class_weight_and_source_by_name(name)

        # divide ADC_gain and resample
        recording = resample(recording, header, resample_Fs=self.sample_rate)
        for lead in recording:
            assert np.isnan(lead).any() == False
        #     if lead.sum() == 0:
        #         print(idx)
        # to extract features
        # recording = self.normalization(recording)

        # recording = filter_and_detrend(recording)
        recording = recording[self.leads_index, :]

        if self.transform is not None:
            recording = self.transform(recording)
        # print(recording)
        recording = torch.tensor(recording)
        label = torch.tensor(label)
        class_weight = torch.tensor(class_weight)
        data_source = torch.tensor(data_source)

        if self.to_get_feature:
            feature = self.get_features(recording, idx, header)
            feature = torch.tensor(feature)
            # print(feature.size())
            return recording, label, class_weight, data_source, feature

        return recording, label, class_weight

    # def get_class_weights(self):
    #     classes = "164889003,164890007,6374002,426627000,733534002,713427006,270492004,713426002,39732003,445118002,164947007,251146004,111975006,698252002,426783006,63593006,10370003,365413008,427172004,164917005,47665007,427393009,426177001,427084000,164934002,59931005"
    #     ### equivalent SNOMED CT codes merged, noted as the larger one
    #     classes = classes.split(',')
    #     CPSC_classes = ['270492004', '164889003', '733534002', '63593006', '426783006',
    #                     '713427006']  # "59118001" = "713427006"
    #     CPSC_class_weight = np.zeros((26,))
    #     for cla in CPSC_classes:
    #         CPSC_class_weight[classes.index(cla)] = 1
    #     # CPSC_extra
    #     CPSC_extra_excluded_classes = ['6374002', '39732003', '445118002', '251146004', '365413008',
    #                                    '164947007', '365413008', '164947007', '698252002', '426783006',
    #                                    '10370003', '111975006', '164917005', '47665007', '427393009',
    #                                    '426177001', '164934002', '59931005']
    #     CPSC_extra_class_weight = np.ones((26,))
    #     for cla in CPSC_extra_excluded_classes:
    #         CPSC_extra_class_weight[classes.index(cla)] = 0
    #     # PTB-XL
    #     PTB_XL_excluded_classes = ['6374002', '426627000', '365413008', '427172004']  # , '17338001'
    #     PTB_XL_class_weight = np.ones((26,))
    #     for cla in PTB_XL_excluded_classes:
    #         PTB_XL_class_weight[classes.index(cla)] = 0
    #     # PTB_XL_class_weight[classes.index('426783006')] = 0.1
    #     # G12ECG
    #     G12ECG_excluded_classes = ['10370003', '365413008', '164947007']
    #     G12ECG_class_weight = np.ones((26,))
    #     for cla in G12ECG_excluded_classes:
    #         G12ECG_class_weight[classes.index(cla)] = 0
    #     # Chapman Shaoxing
    #     Chapman_excluded_classes = ['6374002', '426627000', '713426002', '445118002', '10370003', '365413008',
    #                                 '427172004', '427393009', '63593006']
    #     Chapman_class_weight = np.ones((26,))
    #     for cla in Chapman_excluded_classes:
    #         Chapman_class_weight[classes.index(cla)] = 0
    #     # Ningbo
    #     Ningbo_excluded_classes = ['164889003', '164890007', '426627000']
    #     Ningbo_class_weight = np.ones((26,))
    #     for cla in Ningbo_excluded_classes:
    #         Ningbo_class_weight[classes.index(cla)] = 0
    #     return [CPSC_extra_class_weight, CPSC_extra_class_weight, PTB_XL_class_weight, G12ECG_class_weight, Chapman_class_weight, Ningbo_class_weight]
    def get_class_weights(self):
        classes = "164889003,164890007,6374002,426627000,733534002,713427006,270492004,713426002,39732003,445118002,164947007,251146004,111975006,698252002,426783006,63593006,10370003,365413008,427172004,164917005,47665007,427393009,426177001,427084000,164934002,59931005"
        ### equivalent SNOMED CT codes merged, noted as the larger one
        classes = classes.split(',')
        CPSC_classes = ['270492004', '164889003', '733534002', '63593006', '426783006',
                        '713427006']  # "59118001" = "713427006"
        CPSC_class_weight = np.zeros((26,))
        for cla in CPSC_classes:
            CPSC_class_weight[classes.index(cla)] = 1
        # CPSC_extra
        CPSC_extra_excluded_classes = ['39732003', '445118002', '251146004', '365413008',
                                       '164947007', '365413008', '164947007', '698252002', '426783006',
                                       '10370003', '111975006', '164917005', '47665007', '427393009',
                                       '426177001', '164934002', '59931005']
        CPSC_extra_class_weight = np.ones((26,))
        for cla in CPSC_extra_excluded_classes:
            CPSC_extra_class_weight[classes.index(cla)] = 0
        # PTB-XL
        PTB_XL_excluded_classes = ['426627000', '365413008', '427172004']  # , '17338001'
        PTB_XL_class_weight = np.ones((26,))
        for cla in PTB_XL_excluded_classes:
            PTB_XL_class_weight[classes.index(cla)] = 0
        # PTB_XL_class_weight[classes.index('426783006')] = 0.1
        # G12ECG
        G12ECG_excluded_classes = ['426627000', '10370003', '365413008', '164947007']
        G12ECG_class_weight = np.ones((26,))
        for cla in G12ECG_excluded_classes:
            G12ECG_class_weight[classes.index(cla)] = 0
        # Chapman Shaoxing
        Chapman_excluded_classes = ['426627000', '713426002', '445118002', '10370003', '365413008',
                                    '164947007', '427393009', '63593006']
        Chapman_class_weight = np.ones((26,))
        for cla in Chapman_excluded_classes:
            Chapman_class_weight[classes.index(cla)] = 0
        # Ningbo
        Ningbo_excluded_classes = ['164889003', '164890007', '426627000']
        Ningbo_class_weight = np.ones((26,))
        for cla in Ningbo_excluded_classes:
            Ningbo_class_weight[classes.index(cla)] = 0
        return [CPSC_extra_class_weight, CPSC_extra_class_weight, PTB_XL_class_weight, G12ECG_class_weight, Chapman_class_weight, Ningbo_class_weight]
    def get_class_weight_and_source_by_name(self, name):
        if name[0] == 'A':  # CPSC
            class_weight = self.class_weights[0]
            data_source_class = 0
        elif name[0] == 'Q':  # CPSC-extra
            class_weight = self.class_weights[1]
            data_source_class = 2
        elif name[0] == 'H':  # PTB-XL
            class_weight = self.class_weights[2]
            data_source_class = 0
        elif name[0] == 'E':  # G12ECG
            class_weight = self.class_weights[3]
            data_source_class = 1
        elif name[0] == 'J' and int(name[2:]) <= 10646:  # Chapman
            class_weight = self.class_weights[4]
            data_source_class = 2
        elif name[0] == 'J' and int(name[2:]) > 10646:  # Ningbo
            class_weight = self.class_weights[5]
            data_source_class = 2
        elif name[0] == 'S' or name[0] == 'I':  # Ningbo
            class_weight = np.zeros((26,))
            data_source_class = 2
        return class_weight, data_source_class
    def get_features(self, sample, idx, header):
        ### to get feature
        # features = self.feature[idx]
        # features = topN_fea(sample, 12, 500, 50)

        ### get age and sex
        meta_info = np.zeros((5,))
        age_index = 13 # 1 + 12
        leads_num = len(self.leads_index)
        if leads_num != 8:
            age_index = 1 + leads_num
        sex_index = age_index + 1
        age_string = header[age_index].split('#Age: ')[-1].split('\n')[0]
        try:
            age = int(age_string)
            meta_info[1] = age / 120
        except:
            meta_info[0] = 1
        # if age_string != "Unknown":
        #     meta_info[0] = 1
        #     age = int(age_string)
        #     meta_info[1] = age / 120
        sex_string = header[sex_index].split('#Sex: ')[-1].split('\n')[0]
        if sex_string != "Unknown":
            meta_info[2] = 1
            if sex_string == "Female":
                meta_info[3] = 1
            elif sex_string == "Male":
                meta_info[4] = 1
            else:
                print('unusual gender: {}'.format(sex_string))
        return meta_info

# Challenge Dataloaders and Challenge metircs

class BaseDataLoader(DataLoader):
    """
    Base class for all data loaders
    """

    def __init__(self, train_dataset, val_dataset, test_dataset, batch_size, shuffle, num_workers,
                 collate_fn=default_collate, pin_memory=True):
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.batch_size = batch_size
        self.batch_idx = 0
        self.shuffle = shuffle

        self.init_kwargs = {
            'dataset': self.train_dataset,
            'batch_size': batch_size,
            'shuffle': self.shuffle,
            'collate_fn': collate_fn,
            'num_workers': num_workers,
            'pin_memory': pin_memory,
            'drop_last': True
        }
        super().__init__(**self.init_kwargs)

        self.n_samples = len(self.train_dataset)

        self.valid_data_loader_init_kwargs = {
            'dataset': self.val_dataset,
            'batch_size': batch_size,
            'shuffle': self.shuffle,
            'collate_fn': collate_fn,
            'num_workers': num_workers,
            'pin_memory': pin_memory,
            'drop_last': True
        }

        self.valid_data_loader = DataLoader(**self.valid_data_loader_init_kwargs)

        self.valid_data_loader.n_samples = len(self.val_dataset)

        if self.test_dataset:
            self.test_data_loader_init_kwargs = {
                'dataset': self.test_dataset,
                'batch_size': batch_size,
                'shuffle': False,
                'collate_fn': collate_fn,
                'num_workers': num_workers,
                'pin_memory': pin_memory,
                'drop_last': True
            }

            self.test_data_loader = DataLoader(**self.test_data_loader_init_kwargs)

            self.test_data_loader.n_samples = len(self.test_dataset)

class ChallengeDataLoaderCV(BaseDataLoader):
    """
    challenge2020 data loading
    """
    def __init__(self, label_dir, split_index, batch_size=256, shuffle=True, num_workers=0, resample_Fs=500, window_size=5000, n_segment=1,
                 normalization=False, training_size=None, train_aug=None, val_aug=None, p=0.5, lead_number=12, save_data=False, load_saved_data=False, to_contrast=False, to_filter_noise=False, network_factor=32, to_include_E=True, dataset_name = 'CustomDataset'):
        self.label_dir = label_dir
        self.dir2save_data = '/data/ecg/challenge2021/data/'
        dir2save_data = '/data/ecg/challenge2021/data/'
        start = time.time()
        train_aug = [
                {
                    "type": "SlideAndCut",
                    "args": {
                        "window_size": 4992,
                        "sampling_rate": 500
                    }
                },
                {
                    "type": "RandomMaskLeads",
                    "args": {}
                }
            ]
        val_aug = [
                {
                    "type": "SlideAndCut",
                    "args": {
                        "window_size": 4992,
                        "sampling_rate": 500
                    }
                },
                {
                    "type": "RandomMaskLeads",
                    "args": {}
                }
            ]
        # Define the weights, the SNOMED CT code for the normal class, and equivalent SNOMED CT codes.
        weights_file = 'weights.csv'

        # Load the scored classes and the weights for the Challenge metric.
        print('Loading weights...')
        _, weights, indices = load_weights(weights_file)
        classes = "164889003,164890007,6374002,426627000,733534002,713427006,270492004,713426002,39732003,445118002,164947007,251146004,111975006,698252002,426783006,63593006,10370003,365413008,427172004,164917005,47665007,427393009,426177001,427084000,164934002,59931005"
        ### equivalent SNOMED CT codes merged, noted as the larger one
        classes = classes.split(',')
        self.weights = weights

        # Load the label and output files.
        print('Loading label and output files...')
        label_files = my_find_challenge_files(label_dir)

        # ### load file names
        # # idx2del = np.load('process/idx2del.npy')
        # label_files_tmp = []
        # count = 0
        # for f in label_files:
        #     count += 1
        #     # if count in idx2del:
        #     #     continue
        #     fname = f.split('/')[-1].split('.')[0]
        #     if to_include_E == False:
        #         if fname[0] == 'E':
        #             continue
        #         if fname[0] == 'S' or fname[0] == 'I':
        #             continue
        #     # make sure samples with 0/nan are deleted
        #     # recording, header, name = load_challenge_data(f, label_dir)
        #     # for lead in recording:
        #     #     assert np.sum(lead) != 0
        #     label_files_tmp.append(f)
        # label_files = label_files_tmp

        labels_onehot = load_labels(label_files, classes)

        split_idx = loadmat(split_index)
        train_index, val_index = split_idx['train_index'], split_idx['val_index']
        train_index = train_index.reshape((train_index.shape[1],))
        if training_size is not None:  # for test
            train_index = train_index[0:training_size]
        val_index = val_index.reshape((val_index.shape[1],))

        num_files = len(label_files)
        label_files_train, label_files_val = [], []
        label_train, label_val = [], []
        for i in train_index:
            label_files_train.append(label_files[i])
            label_train.append(labels_onehot[i])
        for i in val_index:
            label_files_val.append(label_files[i])
            label_val.append(labels_onehot[i])


        ### 12 leads order: I II III aVL aVR aVF V1 V2 V3 V4 V5 V6
        # if lead_number == 2:  # two leads: I II
        #     leads_index = [0, 1]
        # elif lead_number == 3:  # three leads: I II V2
        #     leads_index = [0, 1, 7]
        # elif lead_number == 4:  # four leads: I II III V2
        #     leads_index = [0, 1, 2, 7]
        # elif lead_number == 6:  # six leads: I II III aVL aVR aVF
        #     leads_index = [0, 1, 2, 3, 4, 5]
        # elif lead_number == 8:  # eight leads
        #     leads_index = [0, 1, 6, 7, 8, 9, 10, 11]
        # else:  # twelve leads
        #     leads_index = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
        leads_index = [0, 1, 6, 7, 8, 9, 10, 11]
        train_transforms = None
        if train_aug is not None and len(train_aug) > 0:
            train_transforms = []
            for aug in train_aug:
                key = aug['type']
                module_args = dict(aug['args'])
                current_transform = getattr(module_transformers, key)(**module_args)
                train_transforms.append(current_transform)
            train_transforms = transforms.Compose(train_transforms)
        val_transforms = None
        if val_aug is not None and len(val_aug) > 0:
            val_transforms = []
            for aug in val_aug:
                key = aug['type']
                module_args = dict(aug['args'])
                current_transform = getattr(module_transformers, key)(**module_args)
                val_transforms.append(current_transform)
            val_transforms = transforms.Compose(val_transforms)

        Dataset = eval(dataset_name)
        self.train_dataset = Dataset(label_files_train, label_train, label_dir, leads_index, sample_rate=resample_Fs,
                                           transform=train_transforms)
        self.val_dataset = Dataset(label_files_val, label_val, label_dir, leads_index, sample_rate=resample_Fs, transform=val_transforms)

        # self.train_dataset = CustomDataset4SSL(label_files_train, label_train, label_dir, leads_index,
        #                                    transform=train_transforms)
        # self.val_dataset = CustomDataset4SSL(label_files_val, label_val, label_dir, leads_index, transform=val_transforms)

        end = time.time()
        print('time to get and process data: {}'.format(end - start))
        super().__init__(self.train_dataset, self.val_dataset, None, batch_size, shuffle, num_workers)

        # self.valid_data_loader.file_names = file_names
        # self.valid_data_loader.idx = val_index


class ChallengeDataset():
    """
    challenge2020 data loading
    """

    def __init__(self, label_dir, split_index, batch_size=128, shuffle=True, num_workers=0, resample_Fs=500, window_size=5000, n_segment=1,
                 normalization=False, training_size=None, augmentations=None, p=0.5, lead_number=12, save_data=False, load_saved_data=False):
        self.label_dir = label_dir
        self.dir2save_data = '/data/ecg/challenge2021/data/'
        dir2save_data = '/data/ecg/challenge2021/data/'
        start = time.time()

        # Define the weights, the SNOMED CT code for the normal class, and equivalent SNOMED CT codes.
        weights_file = 'weights.csv'

        # Load the scored classes and the weights for the Challenge metric.
        print('Loading weights...')
        _, weights, indices = load_weights(weights_file)
        classes = "164889003,164890007,6374002,426627000,733534002,713427006,270492004,713426002,39732003,445118002,164947007,251146004,111975006,698252002,426783006,63593006,10370003,365413008,427172004,164917005,47665007,427393009,426177001,427084000,164934002,59931005"
        ### equivalent SNOMED CT codes merged, noted as the larger one
        classes = classes.split(',')
        self.weights = weights

        # Load the label and output files.
        print('Loading label and output files...')
        label_files = my_find_challenge_files(label_dir)
        # label_files_tmp = []
        # for f in label_files:
        #     fname = f.split('/')[-1].split('.')[0]
        #     if fname[0] == 'A' or fname[0] == 'E':
        #         continue
        #     label_files_tmp.append(f)
        # label_files = label_files_tmp

        labels_onehot = load_labels(label_files, classes)

        split_idx = loadmat(split_index)
        train_index, val_index = split_idx['train_index'], split_idx['val_index']
        train_index = train_index.reshape((train_index.shape[1],))
        if training_size is not None:  # for test
            train_index = train_index[0:training_size]
        val_index = val_index.reshape((val_index.shape[1],))
        # test_index = test_index.reshape((test_index.shape[1],))

        num_files = len(label_files)
        train_number = 0
        val_number = 0
        for i in range(num_files):
            if i in train_index:
                train_number += 1
            elif i in val_index:
                val_number += 1
        print("train number: {}, val number: {}".format(train_number, val_number))

        train_recordings = np.zeros((train_number, 12, window_size), dtype=float)
        train_class_weights = np.zeros((train_number, 26,), dtype=float)
        train_labels_onehot = np.zeros((train_number, 26,), dtype=float)

        val_recordings = np.zeros((val_number, 12, window_size), dtype=float)
        val_class_weights = np.zeros((val_number, 26,), dtype=float)
        val_labels_onehot = np.zeros((val_number, 26,), dtype=float)

        # file_names = list()

        ### class weights for datasets
        # equivalent diagnose [['713427006', '59118001'], ['63593006', '284470004'], ['427172004', '17338001'], ['733534002', '164909002']]
        # CPSC
        CPSC_classes = ['270492004', '164889003', '733534002', '63593006', '426783006', '713427006']  # "59118001" = "713427006"
        CPSC_class_weight = np.zeros((26,))
        for cla in CPSC_classes:
            CPSC_class_weight[classes.index(cla)] = 1
        # CPSC_extra
        CPSC_extra_excluded_classes = ['6374002', '39732003', '445118002', '251146004', '365413008',
                                       '164947007', '365413008', '164947007', '698252002', '426783006',
                                       '10370003', '111975006', '164917005', '47665007', '427393009',
                                       '426177001', '164934002', '59931005']
        CPSC_extra_class_weight = np.ones((26,))
        for cla in CPSC_extra_excluded_classes:
            CPSC_extra_class_weight[classes.index(cla)] = 0
        # PTB-XL
        PTB_XL_excluded_classes = ['6374002', '426627000', '365413008', '427172004']  # , '17338001'
        PTB_XL_class_weight = np.ones((26,))
        for cla in PTB_XL_excluded_classes:
            PTB_XL_class_weight[classes.index(cla)] = 0
        # G12ECG
        G12ECG_excluded_classes = ['10370003', '365413008', '164947007']
        G12ECG_class_weight = np.ones((26,))
        for cla in G12ECG_excluded_classes:
            G12ECG_class_weight[classes.index(cla)] = 0
        # Chapman Shaoxing
        Chapman_excluded_classes = ['6374002', '426627000', '713426002', '445118002', '10370003', '365413008', '427172004', '427393009', '427084000',
                                    '63593006']
        Chapman_class_weight = np.ones((26,))
        for cla in Chapman_excluded_classes:
            Chapman_class_weight[classes.index(cla)] = 0
        # Ningbo
        Ningbo_excluded_classes = ['164889003', '164890007', '426627000']
        Ningbo_class_weight = np.ones((26,))
        for cla in Ningbo_excluded_classes:
            Ningbo_class_weight[classes.index(cla)] = 0

        train_num = 0
        val_num = 0
        for i in range(num_files):
            print('{}/{}'.format(i + 1, num_files))
            recording, header, name = load_challenge_data(label_files[i], label_dir)

            if name[0] == 'S' or name[0] == 'I':  # filter PTB or St.P dataset
                continue
            elif name[0] == 'A':  # CPSC
                class_weight = CPSC_class_weight
            elif name[0] == 'Q':  # CPSC-extra
                class_weight = CPSC_extra_class_weight
            elif name[0] == 'H':  # PTB-XL
                class_weight = PTB_XL_class_weight
            elif name[0] == 'E':  # G12ECG
                class_weight = G12ECG_class_weight
            elif name[0] == 'J' and int(name[2:]) <= 10646:  # Chapman
                class_weight = Chapman_class_weight
            elif name[0] == 'J' and int(name[2:]) > 10646:  # Ningbo
                class_weight = Ningbo_class_weight
            else:
                print('warning! not from one of the datasets:  ', name)
                continue

            recording[np.isnan(recording)] = 0

            # divide ADC_gain and resample
            recording = resample(recording, header, resample_Fs)

            # to filter and detrend samples
            recording = filter_and_detrend(recording)

            # slide and cut
            recording = slide_and_cut(recording, n_segment, window_size, resample_Fs)
            # file_names.append(name)
            if i in train_index:
                for j in range(recording.shape[0]):  # segment number = 1 -> j=0
                    train_recordings[train_num] = recording[j]
                    train_labels_onehot[train_num] = labels_onehot[i]
                    train_class_weights[train_num] = class_weight
                train_num += 1
            elif i in val_index:
                for j in range(recording.shape[0]):
                    val_recordings[val_num] = recording[j]
                    val_labels_onehot[val_num] = labels_onehot[i]
                    val_class_weights[val_num] = class_weight
                val_num += 1
            else:
                pass

        # # Normalization
        # if normalization:
        #     train_recordings = self.normalization(train_recordings)
        #     val_recordings = self.normalization(val_recordings)

        train_recordings = torch.from_numpy(train_recordings)
        train_class_weights = torch.from_numpy(train_class_weights)
        train_labels_onehot = torch.from_numpy(train_labels_onehot)

        val_recordings = torch.from_numpy(val_recordings)
        val_class_weights = torch.from_numpy(val_class_weights)
        val_labels_onehot = torch.from_numpy(val_labels_onehot)

        self.train_dataset = CustomTensorDataset(train_recordings, train_labels_onehot, train_class_weights)
        self.val_dataset = CustomTensorDataset(val_recordings, val_labels_onehot, val_class_weights)

        end = time.time()
        print('time to get and process data: {}'.format(end - start))


class ChallengeDataLoader(BaseDataLoader):
    """
    challenge2020 data loading
    """

    def __init__(self, train_dataset, val_dataset, batch_size=256, shuffle=True, num_workers=0):
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

        super().__init__(self.train_dataset, self.val_dataset, None, batch_size, shuffle, num_workers)

        # self.valid_data_loader.file_names = file_names
        # self.valid_data_loader.idx = val_index
