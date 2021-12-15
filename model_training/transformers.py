#coding=utf-8
import numpy as np
import torch
from scipy.interpolate import CubicSpline      # for warping
from transforms3d.axangles import axangle2mat  # for rotation
import pywt
from scipy import signal
import pandas as pd
import torch
import torchvision.transforms
import random
import math
from scipy.interpolate import interp1d

## This example using cubic splice is not the best approach to generate random curves.
## You can use other aprroaches, e.g., Gaussian process regression, Bezier curve, etc.
def GenerateRandomCurves(X, sigma=0.2, knot=4):
    # X (C, L)
    # out (C, L) np.ndarry
    from scipy.interpolate import CubicSpline

    xx = (np.ones((X.shape[0], 1)) * (np.arange(0, X.shape[1], (X.shape[1] - 1) / (knot + 1)))).transpose()
    yy = np.random.normal(loc=1.0, scale=sigma, size=(knot + 2, X.shape[0]))
    x_range = np.arange(X.shape[1])

    cs = []
    for i in range(X.shape[0]):
        cs.append(CubicSpline(xx[:, i], yy[:, i]))

    return np.array([cs_i(x_range) for cs_i in cs])

def DistortTimesteps(X, sigma=0.2):
    # X: (C, L)
    # out: (C, L) np.ndarry
    tt = GenerateRandomCurves(X, sigma).transpose() # Regard these samples aroun 1 as time intervals
    tt_cum = np.cumsum(tt, axis=0)        # Add intervals to make a cumulative graph
    # Make the last value to have X.shape[0]
    t_scale = [(X.shape[1] - 1) / tt_cum[-1, i] for i in range(X.shape[0])]
    for i in range(X.shape[0]):
        tt_cum[:,i] = tt_cum[:,i]*t_scale[i]
    return tt_cum.transpose()

def RandSampleTimesteps(X, nSample=1000):
    # X: (C, L)
    # out: (C, L) np.ndarry
    tt = np.zeros((nSample,X.shape[0]), dtype=int)
    for i in range(X.shape[0]):
        tt[1:-1,i] = np.sort(np.random.randint(1,X.shape[1]-1,nSample-2))
    tt[-1,:] = X.shape[1]-1
    return tt.transpose()

def WTfilt_1d(sig):
    # https://blog.csdn.net/weixin_39929602/article/details/111038295
    coeffs = pywt.wavedec(data=sig, wavelet='db5', level=9)
    cA9, cD9, cD8, cD7, cD6, cD5, cD4, cD3, cD2, cD1 = coeffs
    threshold = (np.median(np.abs(cD1)) / 0.6745) * (np.sqrt(2 * np.log(len(cD1))))
    # 将高频信号cD1、cD2置零
    cD1.fill(0)
    cD2.fill(0)
    # 将其他中低频信号按软阈值公式滤波
    for i in range(1, len(coeffs) - 2):
        coeffs[i] = pywt.threshold(coeffs[i], threshold)
    rdata = pywt.waverec(coeffs=coeffs, wavelet='db5')
    return rdata

class Jitter(object):
    """
    Args:
        sigma
    """

    def __init__(self, sigma=0.05):
        self.sigma = sigma

    def __call__(self, tensors):
        """
        Args:
            tensor (Tensor): Tensor of size (C, L) to be scaled.
        Returns:
            Tensor: Scaled Tensor.
        """

        myNoise = torch.normal(mean=torch.zeros(tensors.shape), std=self.sigma)

        # print("This is Jitter")
        # print(type(tensors + myNoise))

        return (tensors + myNoise).float()

    def __repr__(self):
        return self.__class__.__name__ + '(sigma={0})'.format(self.sigma)


class Scaling(object):
    """
    Args:
        sigma
    """

    def __init__(self, sigma=0.1):
        self.sigma = sigma

    def __call__(self, tensors):
        """
        Args:
            tensor (Tensor): Tensor of size (C, L) to be scaled.
        Returns:
            Tensor: Scaled Tensor.
        """

        scalingFactor = torch.normal(mean=torch.ones((tensors.shape[0], 1)), std=self.sigma)
        myNoise = torch.matmul(scalingFactor, torch.ones((1, tensors.shape[1])))

        # print("This is Scaling")
        # print(type(tensors * myNoise))

        return (tensors * myNoise).float()

    def __repr__(self):
        return self.__class__.__name__ + '(sigma={0})'.format(self.sigma)


class MagWarp(object):
    """
    Args:
        sigma
    """

    def __init__(self, sigma=0.2):
        self.sigma = sigma

    def __call__(self, tensors):
        """
        Args:
            tensor (Tensor): Tensor of size (C, L) to be scaled.
        Returns:
            Tensor: Scaled Tensor.
        """
        # print("This is MagWarp")
        # print(type(tensors * torch.from_numpy(GenerateRandomCurves(tensors, self.sigma))))

        return tensors * torch.from_numpy(GenerateRandomCurves(tensors, self.sigma))

    def __repr__(self):
        return self.__class__.__name__ + '(sigma={0})'.format(self.sigma)


class TimeWarp(object):
    """
    Args:
        sigma
    """

    def __init__(self, sigma=0.2):
        self.sigma = sigma

    def __call__(self, tensors):
        """
        Args:
            tensor (Tensor): Tensor of size (C, L) to be scaled.
        Returns:
            Tensor: Scaled Tensor.
        """

        tt_new = DistortTimesteps(tensors, self.sigma)
        X_new = np.zeros(tensors.shape)
        x_range = np.arange(tensors.shape[1])
        for i in range(tensors.shape[0]):
            X_new[i, :] = np.interp(x_range, tt_new[i, :], tensors[i, :])

        # print("This is TimeWarp")
        # print(type(torch.from_numpy(X_new)))

        return torch.from_numpy(X_new).float()

    def __repr__(self):
        return self.__class__.__name__ + '(sigma={0})'.format(self.sigma)


class Rotation(object):
    """
    Args:
    """

    def __init__(self):
        pass

    def __call__(self, tensors):
        """
        Args:
            tensor (Tensor): Tensor of size (C, L) to be scaled.
        Returns:
            Tensor: Scaled Tensor.
        """

        axis = torch.Tensor(tensors.shape[0]).uniform_(-1, 1)
        angle = torch.Tensor().uniform_(-np.pi, np.pi)

        # print("This is Rotation")
        # print(type(torch.matmul(axangle2mat(axis, angle), tensors)))

        return torch.matmul(axangle2mat(axis, angle), tensors).float()

    def __repr__(self):
        return self.__class__.__name__


class Permutation(object):
    """
    Args:
        nPerm:
        minSegLength:
    """

    def __init__(self, nPerm=4, minSegLength=10):
        self.nPerm = nPerm
        self.minSegLength = minSegLength

    def __call__(self, tensors):
        """
        Args:
            tensor (Tensor): Tensor of size (C, L) to be scaled.
        Returns:
            Tensor: Scaled Tensor.
        """

        X_new = torch.zeros(tensors.shape, dtype=torch.int64)
        idx = torch.randperm(self.nPerm)
        bWhile = True
        while bWhile == True:
            segs = torch.zeros(self.nPerm + 1, dtype=torch.int64)
            segs[1:-1] = torch.sort(torch.randint(self.minSegLength, tensors.shape[1] - self.minSegLength, (self.nPerm - 1,))).values
            segs[-1] = tensors.shape[1]
            if torch.min(segs[1:] - segs[0:-1]) > self.minSegLength:
                bWhile = False
        pp = 0
        for ii in range(self.nPerm):
            x_temp = tensors[:, segs[idx[ii]]:segs[idx[ii] + 1]]
            X_new[:, pp:pp + x_temp.shape[1]] = x_temp
            pp += x_temp.shape[1]

        # print("This is Permutation")
        # print(type(X_new))

        return (X_new).float()

    def __repr__(self):
        return self.__class__.__name__


class RandSampling(object):
    """
    Args:
        nSample:
    """

    def __init__(self, nSample=1000):
        self.nSample = nSample

    def __call__(self, tensors):
        """
        Args:
            tensor (Tensor): Tensor of size (C, L) to be scaled.
        Returns:
            Tensor: Scaled Tensor.
        """

        tt = RandSampleTimesteps(tensors, self.nSample)
        X_new = np.zeros(tensors.shape)
        for i in range(tensors.shape[0]):
            X_new[i, :] = np.interp(np.arange(tensors.shape[1]), tt[i, :], tensors[i, tt[i, :]])

        # print("This is RandSampling")
        # print(type(torch.from_numpy(X_new)))

        return (torch.from_numpy(X_new).float())

    def __repr__(self):
        return self.__class__.__name__


class filter_and_detrend(object):
    """
    Args:
    """
    def __init__(self):
        pass

    def __call__(self, data):
        """
        Args:
            data: 12 lead ECG data . For example,the shape of data is (12,5000)
        Returns:
            Tensor: 12 lead ECG data after filtered and detrended
        """

        filtered_data = pd.DataFrame()
        for k in range(12):
            try:
                filtered_data[k] = signal.detrend(WTfilt_1d(data[k]))
            except ValueError:
                ##有些数据全是0，记录下来，无法进行detrend处理
                filtered_data[k] = WTfilt_1d(data[k])

        return (filtered_data.values).T

    def __repr__(self):
        return self.__class__.__name__

import random

###########################################################
# UTILITIES
###########################################################


def interpolate(data, marker):
    timesteps, channels = data.shape
    data = data.flatten(order="F")
    data[data == marker] = np.interp(np.where(data == marker)[0], np.where(
        data != marker)[0], data[data != marker])
    data = data.reshape(timesteps, channels, order="F")
    return data

def Tinterpolate(data, marker):
    timesteps, channels = data.shape
    data = data.transpose(0, 1).flatten()
    ndata = data.numpy()
    interpolation = torch.from_numpy(np.interp(np.where(ndata == marker)[0], np.where(ndata != marker)[0], ndata[ndata != marker]))
    data[data == marker] = interpolation.type(data.type())
    data = data.reshape(channels, timesteps).T
    return data

def squeeze(arr, center, radius, step):
    squeezed = arr[center-step*radius:center+step*radius+1:step, :].copy()
    arr[center-step*radius:center+step*radius+1, :] = np.inf
    arr[center-radius:center+radius+1, :] = squeezed
    return arr

def Tsqueeze(arr, center, radius, step):
    squeezed = arr[center-step*radius:center+step*radius+1:step, :].clone()
    arr[center-step*radius:center+step*radius+1, :]=float("inf")
    arr[center-radius:center+radius+1, :] = squeezed
    return arr

def refill(arr, center, radius, step):
    left_fill_values = arr[center-radius*step -
                           radius:center-radius*step, :].copy()
    right_fill_values = arr[center+radius*step +
                            1:center+radius*step+radius+1, :].copy()
    arr[center-radius*step-radius:center-radius*step, :] = arr[center +
                                                               radius*step+1:center+radius*step+radius+1, :] = np.inf
    arr[center-radius*step-radius:center-radius:step, :] = left_fill_values
    arr[center+radius+step:center+radius*step +
        radius+step:step, :] = right_fill_values
    return arr

def Trefill(arr, center, radius, step):
    left_fill_values = arr[center-radius*step-radius:center-radius*step, :].clone()
    right_fill_values = arr[center+radius*step+1:center+radius*step+radius+1, :].clone()
    arr[center-radius*step-radius:center-radius*step, :] = arr[center+radius*step+1:center+radius*step+radius+1, :] = float("inf")
    arr[center-radius*step-radius:center-radius:step, :] = left_fill_values
    arr[center+radius+step:center+radius*step+radius+step:step, :] = right_fill_values
    return arr
# Cell
class RandomCrop(object):
    """Crop randomly the image in a sample.
    """

    def __init__(self, output_size,annotation=False):
        self.output_size = output_size
        self.annotation = annotation

    def __call__(self, sample):
        data = sample.T
        timesteps= len(data)
        if timesteps < self.output_size:
            output = np.zeros((self.output_size, data.shape[1]))
            output[:timesteps, :] = data
            return output.T
            # print(timesteps)
        assert(timesteps>=self.output_size)
        if(timesteps==self.output_size):
            start=0
        else:
            start = random.randint(0, timesteps - self.output_size-1) #np.random.randint(0, timesteps - self.output_size)

        data = data[start: start + self.output_size]

        return data.T

class RandomCrop_tmp(object):
    """Crop randomly the image in a sample.
    """

    def __init__(self, output_size,annotation=False):
        self.output_size = output_size
        self.annotation = annotation

    def __call__(self, sample):
        data = sample.T
        timesteps= len(data)
        if timesteps < self.output_size:
            output = np.zeros((self.output_size, data.shape[1]))
            output[:timesteps, :] = data
            return output.T
            # print(timesteps)
        assert(timesteps>=self.output_size)
        if(timesteps==self.output_size):
            start=0
        else:
            start = 0
        data = data[start: start + self.output_size]

        return data.T

class RandomCropBaT(object):
    """Crop randomly the image in a sample.
    """

    def __init__(self, output_size,annotation=False):
        self.output_size = output_size
        self.annotation = annotation

    def __call__(self, sample):
        data = sample.T
        timesteps= len(data)
        if timesteps < self.output_size:
            output = np.zeros((self.output_size, data.shape[1]))
            output[:timesteps, :] = data
            return output.T
            # print(timesteps)
        assert(timesteps>=self.output_size)
        if(timesteps==self.output_size):
            start=0
        else:
            start = random.randint(0, timesteps - self.output_size-1) #np.random.randint(0, timesteps - self.output_size)

        data = data[start: start + self.output_size]

        return data.T

class RandomCropOri(object):
    """Crop randomly the image in a sample.
    """

    def __init__(self, output_size,annotation=False):
        self.output_size = output_size
        self.annotation = annotation

    def __call__(self, sample):
        data = sample
        timesteps= len(data)
        assert(timesteps >= self.output_size)
        if(timesteps == self.output_size):
            start=0
        else:
            start = random.randint(0, timesteps - self.output_size-1) #np.random.randint(0, timesteps - self.output_size)

        data = data[start: start + self.output_size]

        return data

class SlideAndCut(object):
    """Crop randomly the image in a sample.
    """

    def __init__(self, n_segment=1, window_size=4992, sampling_rate=500, test_time_aug=False):
        self.n_segment = n_segment
        self.window_size = window_size
        self.sampling_rate = sampling_rate
        self.test_time_aug = test_time_aug

    def __call__(self, sample):
        data = sample

        length = data.shape[1]
        # print("length:", length)
        if length < self.window_size:
            segments = []
            ecg_filled = np.zeros((data.shape[0], self.window_size))
            ecg_filled[:, 0:length] = data[:, 0:length]
            segments.append(ecg_filled)
            segments = np.array(segments)
        elif self.test_time_aug == False:
            # print("not using test-time-aug")
            offset = (length - self.window_size * self.n_segment) / (self.n_segment + 1)
            if offset >= 0:
                start = 0 + offset
            else:
                offset = (length - self.window_size * self.n_segment) / (self.n_segment - 1)
                start = 0
            segments = []
            recording_count = 0
            for j in range(self.n_segment):
                recording_count += 1
                # print(recording_count)
                ind = int(start + j * (self.window_size + offset))
                segment = data[:, ind:ind + self.window_size]
                segments.append(segment)
            segments = np.array(segments)
        elif self.test_time_aug == True:
            # print("using test-time-aug")
            ind = 0
            rest_length = length
            segments = []
            recording_count = 0
            while rest_length - ind >= self.window_size:
                recording_count += 1
                # print(recording_count)
                segment = data[:, ind:ind + self.window_size]
                segments.append(segment)
                ind += int(self.window_size / 2)
            segments = np.array(segments)
        data = segments[0]
        return data

class Transformation:
    def __init__(self, *args, **kwargs):
        self.params = kwargs

    def get_params(self):
        return self.params

# class TimeOut(Transformation):
#     """ replace random crop by zeros
#     """
#
#     def __init__(self, crop_ratio_range=[0.0, 0.5]):
#         super(TimeOut, self).__init__(crop_ratio_range=crop_ratio_range)
#         self.crop_ratio_range = crop_ratio_range
#
#     def __call__(self, sample):
#         data, label = sample
#         data = data.copy()
#         timesteps, channels = data.shape
#         crop_ratio = random.uniform(*self.crop_ratio_range)
#         crop_timesteps = int(crop_ratio*timesteps)
#         start_idx = random.randint(0, timesteps - crop_timesteps-1)
#         data[start_idx:start_idx+crop_timesteps, :] = 0
#         return data, label

class SpatialTransform(Transformation):
    """ Extract crop at random position and resize it to full size
    """

    def __init__(self, rescale_ratio_range=[1.0, 1.5], rx_range=[-30, 30], ry_range=[-30, 30], rz_range=[-30, 30], p=1, mode="idt", not2reverse=False):
        ### mode: idt, kors, or qlsv
        ### rx: rotation along x axis
        ### p: ratio to apply spatial transformation
        super(SpatialTransform, self).__init__(
            rescale_ratio_range=rescale_ratio_range, rx_range=rx_range, ry_range=ry_range, rz_range=rz_range)
        self.rescale_ratio_range = rescale_ratio_range
        self.rx_range = [i / 180 * math.pi for i in rx_range]
        self.ry_range = [i / 180 * math.pi for i in ry_range]
        self.rz_range = [i / 180 * math.pi for i in rz_range]
        self.p = p
        # idt_channels = ['v1', 'v2', 'v3', 'v4', 'v5', 'v6', 'i', 'ii']
        # idt = np.matrix([[-0.17, -0.07, 0.12, 0.23, 0.24, 0.19, 0.16, -0.01],  # x
        #                  [0.06, -0.02, -0.11, -0.02, 0.04, 0.05, -0.23, 0.89],  # y
        #                  [-0.23, -0.31, -0.25, -0.06, 0.05, 0.11, 0.02, 0.10]])  # z
        self.idt_channels = ['i', 'ii', 'v1', 'v2', 'v3', 'v4', 'v5', 'v6']
        self.idt = np.matrix([[0.156, -0.01, -0.172, -0.074, 0.122, 0.231, 0.239, 0.194],  # x
                         [-0.227, 0.887, 0.057, -0.019, -0.106, -0.022, 0.041, 0.048],  # y
                         [0.022, 0.102, -0.229, -0.31, -0.246, -0.063, 0.055, 0.108]])  # z
        if mode == 'kors':
            self.idt = np.matrix([[0.38, -0.07, -0.13, 0.05, 0.01, 0.14, 0.06, 0.54],  # x
                                  [-0.07, 0.93, 0.06, -0.02, -0.05, 0.06, -0.17, 0.13],  # y
                                  [0.11, -0.23, -0.43, -0.06, -0.14, -0.20, -0.11, 0.31]])  # z
        elif mode == 'qlsv':
            self.idt = np.matrix([[0.199, -0.018, -0.147, -0.058, 0.037, 0.139, 0.232, 0.226],  # x
                                  [-0.164, 0.503, 0.023, -0.085, -0.003, 0.033, 0.060, 0.104],  # y
                                  [0.085, -0.130, -0.184, -0.163, -0.193, -0.119, -0.023, 0.043]])  # z

        self.idt_inverse = self.idt.I
        self.not2reverse = not2reverse

    def get_rz(self, theta):
        rz = np.matrix([[math.cos(theta), math.sin(theta), 0],
                        [-math.sin(theta), math.cos(theta), 0],
                        [0, 0, 1]])
        return rz
    def get_ry(self, beta):
        ry = np.matrix([[math.cos(beta), 0, math.sin(beta)],
                        [0, 1, 0],
                        [-math.sin(beta), 0, math.cos(beta)]])
        return ry
    def get_rx(self, alpha):
        rx = np.matrix([[1, 0, 0],
                        [0, math.cos(alpha), math.sin(alpha)],
                        [0, -math.sin(alpha), math.cos(alpha)]])
        return rx
    def get_rs(self, sx, sy, sz):
        rs = np.matrix([[sx, 0, 0],
                        [0, sy, 0],
                        [0, 0, sz]])
        return rs
    def __call__(self, sample):
        if random.random() > self.p:
            return sample
        # print(sample.shape)
        data = sample.T
        theta = random.uniform(self.rz_range[0], self.rz_range[1])
        beta = random.uniform(self.ry_range[0], self.ry_range[1])
        alpha = random.uniform(self.rx_range[0], self.rx_range[1])
        rz = self.get_rz(theta)
        ry = self.get_ry(beta)
        rx = self.get_rx(alpha)
        sx = random.uniform(self.rescale_ratio_range[0], self.rescale_ratio_range[1])
        sy = random.uniform(self.rescale_ratio_range[0], self.rescale_ratio_range[1])
        sz = random.uniform(self.rescale_ratio_range[0], self.rescale_ratio_range[1])
        rescale_matrix = self.get_rs(sx, sy, sz)
        # print(data.shape)
        # print(self.idt.T.shape)
        output = data @ self.idt.T @ rz @ ry @ rx @ rescale_matrix
        if self.not2reverse == False:
            output = output @ self.idt_inverse.T
        return np.array(output.T)

    def __str__(self):
        return "SpatialTransform"

class RandomMaskLeads(Transformation):
    """ Extract crop at random position and resize it to full size
    """

    def __init__(self):
        super(RandomMaskLeads, self).__init__()

    def __call__(self, sample):
        # lead_num_candidate = [8, 3, 2]
        lead_num_candidate = [8, 6, 4, 3, 2]
        # lead_random = random.randint(0, 2)
        lead_random = random.randint(0, 4)
        lead_num = lead_num_candidate[lead_random]  # rewrite lead num
        if lead_num == 8:
            return sample
        elif lead_num == 6 or lead_num == 2:
            sample[2:, :] = 0
        elif lead_num == 3 or lead_num == 4:
            sample[4:, :] = 0
            sample[2:3, :] = 0
        return sample

    def __str__(self):
        return "RandomMaskLeads"

class RandomResizedCrop(Transformation):
    """ Extract crop at random position and resize it to full size
    """

    def __init__(self, crop_ratio_range=[0.5, 1.0], output_size=5000):
        super(RandomResizedCrop, self).__init__(
            crop_ratio_range=crop_ratio_range, output_size=output_size)
        self.crop_ratio_range = crop_ratio_range
        self.output_size = output_size

    def __call__(self, sample):
        # print(sample.shape)
        data = sample.T
        timesteps, channels = data.shape
        timesteps = len(data)
        if timesteps < self.output_size:
            output = np.zeros((self.output_size, data.shape[1]))
            output[:timesteps, :] = data
            return output.T
        output = np.full((self.output_size, channels), np.inf)
        output_timesteps, channels = output.shape
        crop_ratio = random.uniform(*self.crop_ratio_range)
        data = RandomCrop(
            int(crop_ratio*timesteps))(sample)  # apply random crop
        data = data.T
        cropped_timesteps = data.shape[0]
        if output_timesteps >= cropped_timesteps:
            indices = np.sort(np.random.choice(
                np.arange(output_timesteps-2)+1, size=cropped_timesteps-2, replace=False))
            indices = np.concatenate(
                [np.array([0]), indices, np.array([output_timesteps-1])])
            # fill output array randomly (but in right order) with values from random crop
            output[indices, :] = data

            # use interpolation to resize random crop
            output = interpolate(output, np.inf)
        else:
            indices = np.sort(np.random.choice(
                np.arange(cropped_timesteps), size=output_timesteps, replace=False))
            output = data[indices]
        return output.T

    def __str__(self):
        return "RandomResizedCrop"


class GaussianNoise(Transformation):
    """Add gaussian noise to sample.
    """

    def __init__(self, scale=0.1, p=0.8):
        super(GaussianNoise, self).__init__(scale=scale)
        self.scale = scale
        self.p = p

    def __call__(self, sample):
        if random.random() > self.p:
            return sample
        if self.scale == 0:
            return sample
        else:
            data = sample.T
            # np.random.normal(scale=self.scale,size=data.shape).astype(np.float32)
            data = data + np.reshape(np.array([random.gauss(0, self.scale)
                                               for _ in range(np.prod(data.shape))]), data.shape)
            return data.T

    def __str__(self):
        return "GaussianNoise"


class TGaussianNoise(Transformation):
    """Add gaussian noise to sample.
    """

    def __init__(self, scale=0.01):
        super(TGaussianNoise, self).__init__(scale=scale)
        self.scale = scale

    def __call__(self, sample):
        if self.scale == 0:
            return sample
        else:
            data = sample
            data = data + self.scale * torch.randn(data.shape)
            return data

    def __str__(self):
        return "GaussianNoise"


class TRandomResizedCrop(Transformation):
    """ Extract crop at random position and resize it to full size
    """

    def __init__(self, crop_ratio_range=[0.5, 1.0], output_size=5000):
        super(TRandomResizedCrop, self).__init__(
            crop_ratio_range=crop_ratio_range, output_size=output_size)
        self.crop_ratio_range = crop_ratio_range

    def __call__(self, sample):
        # print('rrc')
        sample = torch.transpose(sample, 0, 1)
        output = torch.full(sample.shape, float("inf")).type(sample.type())
        timesteps, channels = output.shape
        crop_ratio = random.uniform(*self.crop_ratio_range)
        data = TRandomCrop(int(crop_ratio * timesteps))(sample)  # apply random crop
        cropped_timesteps = data.shape[0]
        indices = torch.sort((torch.randperm(timesteps - 2) + 1)[:cropped_timesteps - 2])[0]
        indices = torch.cat([torch.tensor([0]), indices, torch.tensor([timesteps - 1])])
        output[indices, :] = data  # fill output array randomly (but in right order) with values from random crop

        # use interpolation to resize random crop
        output = Tinterpolate(output, float("inf"))

        output = torch.transpose(output, 0, 1)
        return output

    def __str__(self):
        return "RandomResizedCrop"


class TRandomCrop(object):
    """Crop randomly the image in a sample.
    """

    def __init__(self, output_size, annotation=False):
        self.output_size = output_size
        self.annotation = annotation

    def __call__(self, sample):
        data = sample

        timesteps, _ = data.shape
        assert (timesteps >= self.output_size)
        if (timesteps == self.output_size):
            start = 0
        else:
            start = random.randint(0,
                                   timesteps - self.output_size - 1)  # np.random.randint(0, timesteps - self.output_size)

        data = data[start: start + self.output_size, :]

        return data

    def __str__(self):
        return "RandomCrop"


class OldDynamicTimeWarp(Transformation):
    """Stretch and squeeze signal randomly along time axis"""

    def __init__(self):
        pass

    def __call__(self, sample):
        data, label = sample
        data = data.copy()
        timesteps, channels = data.shape
        warp_indices = np.sort(np.random.choice(timesteps, size=timesteps))
        data = data[warp_indices, :]
        return data, label

    def __str__(self):
        return "OldDynamicTimeWarp"


class DynamicTimeWarp(Transformation):
    """Stretch and squeeze signal randomly along time axis"""

    def __init__(self, warps=3, radius=10, step=2):
        super(DynamicTimeWarp, self).__init__(
            warps=warps, radius=radius, step=step)
        self.warps = warps
        self.radius = radius
        self.step = step
        self.min_center = self.radius * (self.step + 1)

    def __call__(self, sample):
        data, label = sample
        data = data.copy()
        timesteps, channels = data.shape
        for _ in range(self.warps):
            center = np.random.randint(
                self.min_center, timesteps - self.min_center - self.step)
            data = squeeze(data, center, self.radius, self.step)
            data = refill(data, center, self.radius, self.step)
            data = interpolate(data, np.inf)
        return data, label

    def __str__(self):
        return "DynamicTimeWarp"


class TDynamicTimeWarp(Transformation):
    """Stretch and squeeze signal randomly along time axis"""

    def __init__(self, warps=3, radius=10, step=2):
        super(TDynamicTimeWarp, self).__init__(
            warps=warps, radius=radius, step=step)
        self.warps = warps
        self.radius = radius
        self.step = step
        self.min_center = self.radius * (self.step + 1)

    def __call__(self, sample):
        data = sample.T
        timesteps, channels = data.shape
        for _ in range(self.warps):
            center = random.randint(self.min_center, timesteps - self.min_center - self.step - 1)
            data = Tsqueeze(data, center, self.radius, self.step)
            data = Trefill(data, center, self.radius, self.step)
            data = Tinterpolate(data, float("inf"))
        return data.T

    def __str__(self):
        return "DynamicTimeWarp"


class TimeWarp(Transformation):
    """apply random monotoneous transformation (random walk) to the time axis"""

    def __init__(self, epsilon=10, interpolation_kind="linear", annotation=False):
        super(TimeWarp, self).__init__(epsilon=epsilon,
                                       interpolation_kind=interpolation_kind, annotation=annotation)
        self.scale = 1.
        self.loc = 0.
        self.epsilon = epsilon
        self.annotation = annotation
        self.interpolation_kind = interpolation_kind

    def __call__(self, sample):
        data, label = sample
        data = data.copy()
        timesteps, channels = data.shape

        pmf = np.random.normal(loc=self.loc, scale=self.scale, size=timesteps)
        pmf = np.cumsum(pmf)  # random walk
        pmf = pmf - np.min(pmf) + self.epsilon  # make it positive

        cdf = np.cumsum(pmf)  # by definition monotonically increasing
        tnew = (cdf - cdf[0]) / (cdf[-1] - cdf[0]) * \
               (len(cdf) - 1)  # correct normalization
        told = np.arange(timesteps)

        for c in range(channels):
            f = interp1d(tnew, data[:, c], kind=self.interpolation_kind)
            data[:, c] = f(told)
        if (self.annotation):
            for c in range(label.shape[0]):
                f = interp1d(tnew, label[:, c], kind=self.interpolation_kind)
                label[:, c] = f(told)

        return data, label

    def __str__(self):
        return "TimeWarp"


class ChannelResize(Transformation):
    """Scale amplitude of sample (per channel) by random factor in given magnitude range"""

    def __init__(self, magnitude_range=[0.33, 3], p=0.8):
        magnitude_range = (magnitude_range[0], magnitude_range[1])
        super(ChannelResize, self).__init__(magnitude_range=magnitude_range)
        self.log_magnitude_range = np.log(magnitude_range)
        self.p = p

    def __call__(self, sample):
        if random.random() > self.p:
            return sample
        data = sample.T
        timesteps, channels = data.shape
        resize_factors = np.exp(np.random.uniform(
            *self.log_magnitude_range, size=channels))
        resize_factors_same_shape = np.tile(
            resize_factors, timesteps).reshape(data.shape)
        data = np.multiply(resize_factors_same_shape, data)
        return data.T

    def __str__(self):
        return "ChannelResize"


class TChannelResize(Transformation):
    """Scale amplitude of sample (per channel) by random factor in given magnitude range"""

    def __init__(self, magnitude_range=[0.33, 3]):
        magnitude_range = (magnitude_range[0], magnitude_range[1])
        super(TChannelResize, self).__init__(magnitude_range=magnitude_range)
        self.log_magnitude_range = torch.log(torch.tensor(magnitude_range).float())

    def __call__(self, sample):
        data = sample.T
        timesteps, channels = data.shape
        resize_factors = torch.exp(torch.empty(channels).uniform_(*self.log_magnitude_range))
        resize_factors_same_shape = resize_factors.repeat(timesteps).reshape(data.shape)
        data = resize_factors_same_shape * data
        return data.T

    def __str__(self):
        return "ChannelResize"


class Negation(Transformation):
    """Flip signal horizontally"""

    def __init__(self):
        super(Negation, self).__init__()
        pass

    def __call__(self, sample):
        data, label = sample
        return -1 * data, label

    def __str__(self):
        return "Negation"


class TNegation(Transformation):
    """Flip signal horizontally"""

    def __init__(self):
        super(TNegation, self).__init__()

    def __call__(self, sample):
        data, label = sample
        return -1 * data, label

    def __str__(self):
        return "Negation"


class DownSample(Transformation):
    """Downsample signal"""

    def __init__(self, downsample_ratio=0.2):
        super(DownSample, self).__init__(downsample_ratio=downsample_ratio)
        self.downsample_ratio = 0.5

    def __call__(self, sample):
        data, label = sample
        data = data.copy()
        timesteps, channels = data.shape
        inpt_indices = np.random.choice(np.arange(
            timesteps - 2) + 1, size=int(self.downsample_ratio * timesteps), replace=False)
        data[inpt_indices, :] = np.inf
        data = interpolate(data, np.inf)
        return data, label

    def __str__(self):
        return "DownSample"


class TDownSample(Transformation):
    """Downsample signal"""

    def __init__(self, downsample_ratio=0.8):
        super(TDownSample, self).__init__(downsample_ratio=downsample_ratio)
        self.downsample_ratio = downsample_ratio

    def __call__(self, sample):
        data, label = sample
        timesteps, channels = data.shape
        inpt_indices = (torch.randperm(timesteps - 2) + 1)[:int(1 - self.downsample_ratio * timesteps)]
        output = data.clone()
        output[inpt_indices, :] = float("inf")
        output = Tinterpolate(output, float("inf"))
        return output, label

    def __str__(self):
        return "DownSample"


class TimeOut(Transformation):
    """ replace random crop by zeros
    """

    def __init__(self, crop_ratio_range=[0.0, 0.5], p=0.8):
        super(TimeOut, self).__init__(crop_ratio_range=crop_ratio_range)
        self.crop_ratio_range = crop_ratio_range
        self.p = p

    def __call__(self, sample):
        if random.random() > self.p:
            return sample
        data = sample.T
        data = data.copy()
        timesteps, channels = data.shape
        crop_ratio = random.uniform(*self.crop_ratio_range)
        crop_timesteps = int(crop_ratio * timesteps)
        start_idx = random.randint(0, timesteps - crop_timesteps - 1)
        data[start_idx:start_idx + crop_timesteps, :] = 0
        return data.T


class TTimeOut(Transformation):
    """ replace random crop by zeros
    """

    def __init__(self, crop_ratio_range=[0.0, 0.5]):
        super(TTimeOut, self).__init__(crop_ratio_range=crop_ratio_range)
        self.crop_ratio_range = crop_ratio_range

    def __call__(self, sample):
        sample = torch.transpose(sample, 0, 1)
        data = sample
        data = data.clone()
        timesteps, channels = data.shape
        crop_ratio = random.uniform(*self.crop_ratio_range)
        crop_timesteps = int(crop_ratio * timesteps)
        start_idx = random.randint(0, timesteps - crop_timesteps - 1)
        data[start_idx:start_idx + crop_timesteps, :] = 0
        data = torch.transpose(data, 0, 1)
        return data

    def __str__(self):
        return "TimeOut"


class TGaussianBlur1d(Transformation):
    def __init__(self):
        super(TGaussianBlur1d, self).__init__()
        self.conv = torch.nn.modules.conv.Conv1d(1, 1, 5, 1, 2, bias=False)
        self.conv.weight.data = torch.nn.Parameter(torch.tensor([[[0.1, 0.2, 0.4, 0.2, 0.1]]]))
        self.conv.weight.requires_grad = False

    def __call__(self, sample):
        sample = torch.transpose(sample, 0, 1)
        data = sample
        transposed = data.T
        transposed = torch.unsqueeze(transposed, 1)
        blurred = self.conv(transposed)
        output = blurred.reshape(data.T.shape).T
        output = torch.transpose(output, 0, 1)
        return output

    def __str__(self):
        return "GaussianBlur"


class ToTensor(Transformation):
    """Convert ndarrays in sample to Tensors."""

    def __init__(self, transpose_data=True, transpose_label=False):
        super(ToTensor, self).__init__(
            transpose_data=transpose_data, transpose_label=transpose_label)
        # swap channel and time axis for direct application of pytorch's convs
        self.transpose_data = transpose_data
        self.transpose_label = transpose_label

    def __call__(self, sample):

        def _to_tensor(data, transpose=False):
            if (isinstance(data, np.ndarray)):
                if (transpose):  # seq,[x,y,]ch
                    return torch.from_numpy(np.moveaxis(data, -1, 0))
                else:
                    return torch.from_numpy(data)
            else:  # default_collate will take care of it
                return data

        data = sample

        if not isinstance(data, tuple):
            data = _to_tensor(data, self.transpose_data)
        else:
            data = tuple(_to_tensor(x, self.transpose_data) for x in data)

        return data  # returning as a tuple (potentially of lists)

    def __str__(self):
        return "ToTensor"


class TNormalize(Transformation):
    """Normalize using given stats.
    """

    def __init__(self, stats_mean=None, stats_std=None, input=True, channels=[]):
        super(TNormalize, self).__init__(
            stats_mean=stats_mean, stats_std=stats_std, input=input, channels=channels)
        if stats_mean is not None:
            self.stats_mean = torch.tensor(stats_mean)
        else:
            self.stats_mean = torch.tensor(
                [5.33480519e-06,  1.05746304e-04,  1.00707498e-04, -8.36899566e-06, -9.41383376e-05,  5.62222407e-05, -5.31167319e-05,  6.35228545e-05, 7.90993635e-05,  9.46307917e-05,  6.74639889e-05, -8.13159921e-05])
        if stats_std is not None:
            self.stats_std = torch.tensor(stats_std)
        else:
            self.stats_std = torch.tensor(
                [0.04878782, 0.05301385, 0.04988954, 0.04432854, 0.0415432,  0.04630683, 0.07092083, 0.10342499, 0.10579466, 0.10725811, 0.10877038, 0.11254028])
        # self.stats_mean = torch.tensor(
        #     [-0.00184586, -0.00130277, 0.00017031, -0.00091313, -0.00148835, -0.00174687, -0.00077071, -0.00207407,
        #      0.00054329, 0.00155546, -0.00114379, -0.00035649])
        # self.stats_std = torch.tensor(
        #     [0.16401004, 0.1647168, 0.23374124, 0.33767231, 0.33362807, 0.30583013, 0.2731171, 0.27554379, 0.17128962,
        #      0.14030828, 0.14606956, 0.14656108])
        # self.stats_mean = self.stats_mean if stats_mean is None else stats_mean
        # self.stats_std = self.stats_std if stats_std is None else stats_std
        self.input = input
        if (len(channels) > 0):
            for i in range(len(stats_mean)):
                if (not (i in channels)):
                    self.stats_mean[:, i] = 0
                    self.stats_std[:, i] = 1

    def __call__(self, sample):
        datax = sample.T
        data = datax
        # assuming channel last
        if (self.stats_mean is not None):
            data = data - self.stats_mean
        if (self.stats_std is not None):
            data = data / self.stats_std
        return data.T


class Transpose(Transformation):

    def __init__(self):
        super(Transpose, self).__init__()

    def __call__(self, sample):
        data = sample.T
        return data

    def __str__(self):
        return "Transpose"


###########################################################
# ECG Noise Transformations
###########################################################

def signal_power(s):
    return np.mean(s * s)


def snr(s1, s2):
    return 10 * np.log10(signal_power(s1) / signal_power(s2))


def baseline_wonder(ss_length=250, fs=500, C=1, K=50, df=0.01):
    """
        Args:
            ss_length: sample size length in steps, default 250
            st_length: sample time legnth in secondes, default 10
            C:         scaling factor of baseline wonder, default 1
            K:         number of sinusoidal functions, default 50
            df:        f_s/ss_length with f_s beeing the sampling frequency, default 0.01
    """
    t = np.tile(np.arange(0, ss_length / fs, 1. / fs), K).reshape(K, ss_length)
    k = np.tile(np.arange(K), ss_length).reshape(K, ss_length, order="F")
    phase_k = np.random.uniform(0, 2 * np.pi, size=K)
    phase_k = np.tile(phase_k, ss_length).reshape(K, ss_length, order="F")
    a_k = np.tile(np.random.uniform(0, 1, size=K),
                  ss_length).reshape(K, ss_length, order="F")
    # a_k /= a_k[:, 0].sum() # normalize a_k's for convex combination?
    pre_cos = 2 * np.pi * k * df * t + phase_k
    cos = np.cos(pre_cos)
    weighted_cos = a_k * cos
    res = weighted_cos.sum(axis=0)
    return C * res


def noise_baseline_wander(fs=500, N=1000, C=1.0, fc=0.5, fdelta=0.01, channels=1, independent_channels=False):
    '''baseline wander as in https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5361052/
    fs: sampling frequency (Hz)
    N: lenght of the signal (timesteps)
    C: relative scaling factor (default scale : 1)
    fc: cutoff frequency for the baseline wander (Hz)
    fdelta: lowest resolvable frequency (defaults to fs/N if None is passed)
    channels: number of output channels
    independent_channels: different channels with genuinely different outputs (but all components in phase) instead of just a global channel-wise rescaling
    '''
    if (fdelta is None):  # 0.1
        fdelta = fs / N

    t = np.arange(0, N / fs, 1. / fs)
    K = int(np.round(fc / fdelta))

    signal = np.zeros((N, channels))
    for k in range(1, K + 1):
        phik = random.uniform(0, 2 * math.pi)
        ak = random.uniform(0, 1)
        for c in range(channels):
            if (independent_channels and c > 0):  # different amplitude but same phase
                ak = random.uniform(0, 1) * (2 * random.randint(0, 1) - 1)
            signal[:, c] += C * ak * np.cos(2 * math.pi * k * fdelta * t + phik)

    if (not (independent_channels) and channels > 1):  # just rescale channels by global factor
        channel_gains = np.array(
            [(2 * random.randint(0, 1) - 1) * random.gauss(1, 1) for _ in range(channels)])
        signal = signal * channel_gains[None]
    return signal


def Tnoise_baseline_wander(fs=500, N=1000, C=1.0, fc=0.5, fdelta=0.01, channels=1, independent_channels=False):
    '''baseline wander as in https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5361052/
    fs: sampling frequency (Hz)
    N: lenght of the signal (timesteps)
    C: relative scaling factor (default scale : 1)
    fc: cutoff frequency for the baseline wander (Hz)
    fdelta: lowest resolvable frequency (defaults to fs/N if None is passed)
    channels: number of output channels
    independent_channels: different channels with genuinely different outputs (but all components in phase) instead of just a global channel-wise rescaling
    '''
    if (fdelta is None):  # 0.1
        fdelta = fs / N

    K = int((fc / fdelta) + 0.5)
    t = torch.arange(0, N / fs, 1. / fs).repeat(K).reshape(K, N)
    k = torch.arange(K).repeat(N).reshape(N, K).T
    phase_k = torch.empty(K).uniform_(0, 2 * math.pi).repeat(N).reshape(N, K).T
    a_k = torch.empty(K).uniform_(0, 1).repeat(N).reshape(N, K).T
    pre_cos = 2 * math.pi * k * fdelta * t + phase_k
    cos = torch.cos(pre_cos)
    weighted_cos = a_k * cos
    res = weighted_cos.sum(dim=0)
    return C * res


#     if(not(independent_channels) and channels>1):#just rescale channels by global factor
#         channel_gains = np.array([(2*random.randint(0,1)-1)*random.gauss(1,1) for _ in range(channels)])
#         signal = signal*channel_gains[None]
#     return signal

def noise_electromyographic(N=1000, C=1, channels=1):
    '''electromyographic (hf) noise inspired by https://ieeexplore.ieee.org/document/43620
    N: lenght of the signal (timesteps)
    C: relative scaling factor (default scale: 1)
    channels: number of output channels
    '''
    # C *=0.3 #adjust default scale

    signal = []
    for c in range(channels):
        signal.append(np.array([random.gauss(0.0, C) for i in range(N)]))

    return np.stack(signal, axis=1)


def Tnoise_electromyographic(N=1000, C=1, channels=1):
    '''electromyographic (hf) noise inspired by https://ieeexplore.ieee.org/document/43620
    N: lenght of the signal (timesteps)
    C: relative scaling factor (default scale: 1)
    channels: number of output channels
    '''
    # C *=0.3 #adjust default scale

    signal = torch.empty((N, channels)).normal_(0.0, C)

    return signal


def noise_powerline(fs=500, N=1000, C=1, fn=50., K=3, channels=1):
    '''powerline noise inspired by https://ieeexplore.ieee.org/document/43620
    fs: sampling frequency (Hz)
    N: lenght of the signal (timesteps)
    C: relative scaling factor (default scale: 1)
    fn: base frequency of powerline noise (Hz)
    K: number of higher harmonics to be considered
    channels: number of output channels (just rescaled by a global channel-dependent factor)
    '''
    # C *= 0.333 #adjust default scale
    t = np.arange(0, N / fs, 1. / fs)

    signal = np.zeros(N)
    phi1 = random.uniform(0, 2 * math.pi)
    for k in range(1, K + 1):
        ak = random.uniform(0, 1)
        signal += C * ak * np.cos(2 * math.pi * k * fn * t + phi1)
    signal = C * signal[:, None]
    if (channels > 1):
        channel_gains = np.array([random.uniform(-1, 1)
                                  for _ in range(channels)])
        signal = signal * channel_gains[None]
    return signal


def Tnoise_powerline(fs=500, N=1000, C=1, fn=50., K=3, channels=1):
    '''powerline noise inspired by https://ieeexplore.ieee.org/document/43620
    fs: sampling frequency (Hz)
    N: lenght of the signal (timesteps)
    C: relative scaling factor (default scale: 1)
    fn: base frequency of powerline noise (Hz)
    K: number of higher harmonics to be considered
    channels: number of output channels (just rescaled by a global channel-dependent factor)
    '''
    # C *= 0.333 #adjust default scale
    t = torch.arange(0, N / fs, 1. / fs)

    signal = torch.zeros(N)
    phi1 = random.uniform(0, 2 * math.pi)
    for k in range(1, K + 1):
        ak = random.uniform(0, 1)
        signal += C * ak * torch.cos(2 * math.pi * k * fn * t + phi1)
    signal = C * signal[:, None]
    if (channels > 1):
        channel_gains = torch.empty(channels).uniform_(-1, 1)
        signal = signal * channel_gains[None]
    return signal


def noise_baseline_shift(fs=500, N=1000, C=1.0, mean_segment_length=3, max_segments_per_second=0.3, channels=1):
    '''baseline shifts inspired by https://ieeexplore.ieee.org/document/43620
    fs: sampling frequency (Hz)
    N: lenght of the signal (timesteps)
    C: relative scaling factor (default scale: 1)
    mean_segment_length: mean length of a shifted baseline segment (seconds)
    max_segments_per_second: maximum number of baseline shifts per second (to be multiplied with the length of the signal in seconds)
    '''
    # C *=0.5 #adjust default scale
    signal = np.zeros(N)

    maxsegs = int(np.ceil(max_segments_per_second * N / fs))

    for i in range(random.randint(0, maxsegs)):
        mid = random.randint(0, N - 1)
        seglen = random.gauss(mean_segment_length, 0.2 * mean_segment_length)
        left = max(0, int(mid - 0.5 * fs * seglen))
        right = min(N - 1, int(mid + 0.5 * fs * seglen))
        ak = random.uniform(-1, 1)
        signal[left:right + 1] = ak
    signal = C * signal[:, None]

    if (channels > 1):
        channel_gains = np.array(
            [(2 * random.randint(0, 1) - 1) * random.gauss(1, 1) for _ in range(channels)])
        signal = signal * channel_gains[None]
    return signal


def Tnoise_baseline_shift(fs=500, N=1000, C=1.0, mean_segment_length=3, max_segments_per_second=0.3, channels=1):
    '''baseline shifts inspired by https://ieeexplore.ieee.org/document/43620
    fs: sampling frequency (Hz)
    N: lenght of the signal (timesteps)
    C: relative scaling factor (default scale: 1)
    mean_segment_length: mean length of a shifted baseline segment (seconds)
    max_segments_per_second: maximum number of baseline shifts per second (to be multiplied with the length of the signal in seconds)
    '''
    # C *=0.5 #adjust default scale
    signal = torch.zeros(N)

    maxsegs = int((max_segments_per_second * N / fs) + 0.5)

    for i in range(random.randint(0, maxsegs)):
        mid = random.randint(0, N - 1)
        seglen = random.gauss(mean_segment_length, 0.2 * mean_segment_length)
        left = max(0, int(mid - 0.5 * fs * seglen))
        right = min(N - 1, int(mid + 0.5 * fs * seglen))
        ak = random.uniform(-1, 1)
        signal[left:right + 1] = ak
    signal = C * signal[:, None]

    if (channels > 1):
        channel_gains = 2 * torch.randint(2, (channels,)) - 1 * torch.empty(channels).normal_(1, 1)
        signal = signal * channel_gains[None]
    return signal


def baseline_wonder(N=250, fs=500, C=1, fc=0.5, df=0.01):
    """
        Args:
            ss_length: sample size length in steps, default 250
            st_length: sample time legnth in secondes, default 10
            C:         scaling factor of baseline wonder, default 1
            K:         number of sinusoidal functions, default 50
            df:        f_s/ss_length with f_s beeing the sampling frequency, default 0.01
    """
    K = int(np.round(fc / df))
    t = np.tile(np.arange(0, N / fs, 1. / fs), K).reshape(K, N)
    k = np.tile(np.arange(K), N).reshape(K, N, order="F")
    phase_k = np.random.uniform(0, 2 * np.pi, size=K)
    phase_k = np.tile(phase_k, N).reshape(K, N, order="F")
    a_k = np.tile(np.random.uniform(0, 1, size=K), N).reshape(K, N, order="F")

    pre_cos = 2 * np.pi * k * df * t + phase_k
    cos = np.cos(pre_cos)
    weighted_cos = a_k * cos
    res = weighted_cos.sum(axis=0)
    return C * res


class BaselineWander(Transformation):
    """Adds baseline wander to the sample.
    """

    def __init__(self, fs=500, Cmax=0.3, fc=0.5, fdelta=0.01, independent_channels=False):
        super(BaselineWander, self).__init__(fs=fs, Cmax=Cmax, fc=fc, fdelta=fdelta,
                                             independent_channels=independent_channels)

    def __call__(self, sample):
        data, label = sample
        timesteps, channels = data.shape
        C = random.uniform(0, self.params["Cmax"])
        data = data + noise_baseline_wander(fs=self.params["fs"], N=len(data), C=0.05, fc=self.params["fc"],
                                            fdelta=self.params["fdelta"], channels=channels,
                                            independent_channels=self.params["independent_channels"])
        return data, label

    def __str__(self):
        return "BaselineWander"


class TBaselineWander(Transformation):
    """Adds baseline wander to the sample.
    """

    def __init__(self, fs=500, Cmax=0.1, fc=0.5, fdelta=0.01, independent_channels=False):
        super(TBaselineWander, self).__init__(fs=fs, Cmax=Cmax, fc=fc, fdelta=fdelta,
                                              independent_channels=independent_channels)

    def __call__(self, sample):
        data = sample.T
        timesteps, channels = data.shape
        C = random.uniform(0, self.params["Cmax"])
        noise = Tnoise_baseline_wander(fs=self.params["fs"], N=len(data), C=C, fc=self.params["fc"],
                                       fdelta=self.params["fdelta"], channels=channels,
                                       independent_channels=self.params["independent_channels"])
        data += noise.repeat(channels).reshape(channels, timesteps).T
        return data.T

    def __str__(self):
        return "BaselineWander"


class PowerlineNoise(Transformation):
    """Adds powerline noise to the sample.
    """

    def __init__(self, fs=500, Cmax=2, K=3):
        super(PowerlineNoise, self).__init__(fs=fs, Cmax=Cmax, K=K)

    def __call__(self, sample):
        data, label = sample
        C = random.uniform(0, self.params["Cmax"])
        data = data + noise_powerline(fs=self.params["fs"], N=len(
            data), C=C, K=self.params["K"], channels=len(data[0]))
        return data, label

    def __str__(self):
        return "PowerlineNoise"


class TPowerlineNoise(Transformation):
    """Adds powerline noise to the sample.
    """

    def __init__(self, fs=500, Cmax=1.0, K=3):
        super(TPowerlineNoise, self).__init__(fs=fs, Cmax=Cmax, K=K)

    def __call__(self, sample):
        data = sample.T
        C = random.uniform(0, self.params["Cmax"])
        data = data + noise_powerline(fs=self.params["fs"], N=len(data), C=C, K=self.params["K"], channels=len(data[0]))
        return data.T

    def __str__(self):
        return "PowerlineNoise"


class EMNoise(Transformation):
    """Adds electromyographic hf noise to the sample.
    """

    def __init__(self, Cmax=0.5, K=3):
        super(EMNoise, self).__init__(Cmax=Cmax, K=K)

    def __call__(self, sample):
        data, label = sample
        C = random.uniform(0, self.params["Cmax"])
        data = data + \
               noise_electromyographic(N=len(data), C=C, channels=len(data[0]))
        return data, label

    def __str__(self):
        return "EMNoise"


class TEMNoise(Transformation):
    """Adds electromyographic hf noise to the sample.
    """

    def __init__(self, Cmax=0.1, K=3):
        super(TEMNoise, self).__init__(Cmax=Cmax, K=K)

    def __call__(self, sample):
        sample = sample.T
        data = sample
        C = random.uniform(0, self.params["Cmax"])
        data = data + Tnoise_electromyographic(N=len(data), C=C, channels=len(data[0]))
        return data.T

    def __str__(self):
        return "EMNoise"


class BaselineShift(Transformation):
    """Adds abrupt baseline shifts to the sample.
    """

    def __init__(self, fs=500, Cmax=3, mean_segment_length=3, max_segments_per_second=0.3):
        super(BaselineShift, self).__init__(fs=fs, Cmax=Cmax,
                                            mean_segment_length=mean_segment_length,
                                            max_segments_per_second=max_segments_per_second)

    def __call__(self, sample):
        data, label = sample
        C = random.uniform(0, self.params["Cmax"])
        data = data + noise_baseline_shift(fs=self.params["fs"], N=len(data), C=C,
                                           mean_segment_length=self.params["mean_segment_length"],
                                           max_segments_per_second=self.params["max_segments_per_second"],
                                           channels=len(data[0]))
        return data, label

    def __str__(self):
        return "BaselineShift"


class TBaselineShift(Transformation):
    """Adds abrupt baseline shifts to the sample.
    """

    def __init__(self, fs=500, Cmax=1.0, mean_segment_length=3, max_segments_per_second=0.3):
        super(TBaselineShift, self).__init__(fs=fs, Cmax=Cmax, mean_segment_length=mean_segment_length,
                                             max_segments_per_second=max_segments_per_second)

    def __call__(self, sample):
        data = sample.T
        C = random.uniform(0, self.params["Cmax"])
        data = data + Tnoise_baseline_shift(fs=self.params["fs"], N=len(data), C=C,
                                            mean_segment_length=self.params["mean_segment_length"],
                                            max_segments_per_second=self.params["max_segments_per_second"],
                                            channels=len(data[0]))
        return data.T

    def __str__(self):
        return "BaselineShift"

### From BUTTeam


class HardClip(object):
    """Returns scaled and clipped data between range <-clipping_threshold:clipping_threshold>"""

    def __init__(self, threshold):
        self.threshold = threshold

    def __call__(self, sample, **kwargs):
        sample_mean = np.mean(sample, axis=1)
        sample = sample - sample_mean.reshape(-1, 1)
        sample[sample > self.threshold] = self.threshold
        sample[sample < -self.threshold] = -self.threshold
        sample = sample / self.threshold

        return sample


class ZScore:
    """Returns Z-score normalized data"""

    def __init__(self, mean=0, std=1000):
        self.mean = mean
        self.std = std

    def __call__(self, sample, **kwargs):
        sample = sample - np.array(self.mean).reshape(-1, 1)
        sample = sample / self.std

        return sample


class RandomShift:
    """
        Class randomly shifts signal within temporal dimension
    """

    def __init__(self, p=0):
        self.probability = p

    def __call__(self, sample, **kwargs):
        self.sample_length = sample.shape[1]
        self.sample_channels = sample.shape[0]

        if random.random() < self.probability:
            shift = torch.randint(self.sample_length, (1, 1)).view(-1).numpy()

            sample = np.roll(sample, shift, axis=1)

        return sample


class RandomStretch:
    """
    Class randomly stretches temporal dimension of signal
    """

    def __init__(self, p=0, max_stretch=0.1):
        self.probability = p
        self.max_stretch = max_stretch

    def __call__(self, sample, **kwargs):
        self.sample_length = sample.shape[1]
        self.sample_channels = sample.shape[0]

        if random.random() < self.probability:
            relative_change = 1 + torch.rand(1).numpy()[0] * 2 * self.max_stretch - self.max_stretch
            if relative_change < 1:
                relative_change = 1 / (1 - relative_change + 1)

            new_len = int(relative_change * self.sample_length)

            stretched_sample = np.zeros((self.sample_channels, new_len))
            for channel_idx in range(self.sample_channels):
                stretched_sample[channel_idx, :] = np.interp(np.linspace(0, self.sample_length - 1, new_len),
                                                             np.linspace(0, self.sample_length - 1, self.sample_length),
                                                             sample[channel_idx, :])

            sample = stretched_sample
        return sample


class RandomAmplifier:
    """
    Class randomly amplifies signal
    """

    def __init__(self, p=0, max_multiplier=0.2):
        self.probability = p
        self.max_multiplier = max_multiplier

    def __call__(self, sample, **kwargs):
        self.sample_length = sample.shape[1]
        self.sample_channels = sample.shape[0]

        if random.random() < self.probability:
            for channel_idx in range(sample.shape[0]):
                multiplier = 1 + random.random() * 2 * self.max_multiplier - self.max_multiplier

                ##mutliply by 2 is same as equvalent to multiply by 0.5 not 0!
                if multiplier < 1:
                    multiplier = 1 / (1 - multiplier + 1)

                sample[channel_idx, :] = sample[channel_idx, :] * multiplier

        return sample


class RandomLeadSwitch(object):
    """Simulates reversal of ecg leads"""
    """Should be only I and aVL"""

    def __init__(self, p=0.05):
        self.probability = p
        self.reversal_type = ["LA_LR", "LA_LL", "RA_LL", "PRECORDIAL"]
        self.weights = [3, 1, 1, 2]
        self.precordial_pairs = [("V1", "V2"), ("V2", "V3"), ("V3", "V4"), ("V4", "V5"), ("V5", "V6")]
        self.lead_map = dict(zip(
            ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"],
            range(0, 12))
        )

    def __call__(self, sample, **kwargs):
        """
        :param sample (numpy array): multidimensional array
        :return: sample (numpy array)
        """
        self.sample = sample

        if random.random() < self.probability:
            selected_type = random.choices(self.reversal_type, weights=self.weights, k=1)[0]
            if selected_type == "LA_LR":
                self.invert_channel("I")
                self.switch_channel(["II", "III"])
                self.switch_channel(["aVL", "aVR"])
                return self.sample

            if selected_type == "LA_LL":
                self.invert_channel("III")
                self.switch_channel(["I", "II"])
                self.switch_channel(["aVL", "aVF"])
                return self.sample

            if selected_type == "RA_LL":
                self.invert_channel("I")
                self.invert_channel("II")
                self.invert_channel("III")
                self.switch_channel(["I", "III"])
                self.switch_channel(["aVR", "aVF"])
                return self.sample

            if selected_type == "PRECORDIAL":
                self.switch_channel(random.choices(self.precordial_pairs, k=1)[0])
                return self.sample
        else:
            return self.sample

    def invert_channel(self, channel_name):
        self.sample[self.lead_map[channel_name], :] *= -1

    def switch_channel(self, channel_names):
        self.sample[[self.lead_map[channel_names[0]], self.lead_map[channel_names[1]]], :] = \
            self.sample[[self.lead_map[channel_names[1]], self.lead_map[channel_names[0]]], :]


class Resample:
    def __init__(self, output_sampling=500):
        self.output_sampling = int(output_sampling)

    def __call__(self, sample, input_sampling, gain):

        sample = sample.astype(np.float32)
        for k in range(sample.shape[0]):
            sample[k, :] = sample[k, :] * gain[k]

        # Rescale data
        self.sample = sample
        self.input_sampling = int(input_sampling)

        factor = self.output_sampling / self.input_sampling

        len_old = self.sample.shape[1]
        num_of_leads = self.sample.shape[0]

        new_length = int(factor * len_old)
        resampled_sample = np.zeros((num_of_leads, new_length))

        for channel_idx in range(num_of_leads):
            tmp = self.sample[channel_idx, :]

            ### antialias
            if factor < 1:
                q = 1 / factor

                half_len = 10 * q
                n = 2 * half_len
                b, a = firwin(int(n) + 1, 1. / q, window='hamming'), 1.
                tmp = filtfilt(b, a, tmp)

            l1 = np.linspace(0, len_old - 1, new_length)
            l2 = np.linspace(0, len_old - 1, len_old)
            tmp = np.interp(l1, l2, tmp)
            resampled_sample[channel_idx, :] = tmp

        return resampled_sample

class BaseLineFilter:
    def __init__(self, window_size=1000):
        self.window_size = window_size

    def __call__(self, sample, **kwargs):
        for channel_idx in range(sample.shape[0]):
            running_mean = BaseLineFilter._running_mean(sample[channel_idx], self.window_size)
            sample[channel_idx] = sample[channel_idx] - running_mean
        return sample

    @staticmethod
    def _running_mean(sample, window_size):
        window = signal.windows.blackman(window_size)
        window = window / np.sum(window)
        return signal.fftconvolve(sample, window, mode="same")
