import numpy as np
from PIL import Image

import math
import os
import random
import torch
import json
import torch.utils.data
import numpy as np
import librosa
from librosa.util import normalize
from scipy.io.wavfile import read
from librosa.filters import mel as librosa_mel_fn

import torch.nn.functional as F
import torch.nn as nn
from torch.nn import Conv1d, ConvTranspose1d, AvgPool1d, Conv2d
from torch.nn.utils import weight_norm, remove_weight_norm, spectral_norm

MAX_WAV_VALUE = 32768.0


def load_wav(full_path):
    sampling_rate, data = read(full_path)
    return data, sampling_rate


def dynamic_range_compression(x, C=1, clip_val=1e-5):
    return np.log(np.clip(x, a_min=clip_val, a_max=None) * C)


def dynamic_range_decompression(x, C=1):
    return np.exp(x) / C


def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
    return torch.log(torch.clamp(x, min=clip_val) * C)


def dynamic_range_decompression_torch(x, C=1):
    return torch.exp(x) / C


def spectral_normalize_torch(magnitudes):
    output = dynamic_range_compression_torch(magnitudes)
    return output


def spectral_de_normalize_torch(magnitudes):
    output = dynamic_range_decompression_torch(magnitudes)
    return output


mel_basis = {}
hann_window = {}


def mel_spectrogram(y, n_fft, num_mels, sampling_rate, hop_size, win_size, fmin, fmax, center=False):
    if torch.min(y) < -1.:
        print('min value is ', torch.min(y))
    if torch.max(y) > 1.:
        print('max value is ', torch.max(y))

    global mel_basis, hann_window
    if fmax not in mel_basis:
        mel = librosa_mel_fn(sr=sampling_rate, n_fft=n_fft, n_mels=num_mels, fmin=fmin, fmax=fmax)
        mel_basis[str(fmax)+'_'+str(y.device)] = torch.from_numpy(mel).float().to(y.device)
        hann_window[str(y.device)] = torch.hann_window(win_size).to(y.device)

    y = torch.nn.functional.pad(y.unsqueeze(1), (int((n_fft-hop_size)/2), int((n_fft-hop_size)/2)), mode='reflect')
    y = y.squeeze(1)

    # complex tensor as default, then use view_as_real for future pytorch compatibility
    spec = torch.stft(y, n_fft, hop_length=hop_size, win_length=win_size, window=hann_window[str(y.device)],
                      center=center, pad_mode='reflect', normalized=False, onesided=True, return_complex=True)
    spec = torch.view_as_real(spec)
    spec = torch.sqrt(spec.pow(2).sum(-1)+(1e-9))

    spec = torch.matmul(mel_basis[str(fmax)+'_'+str(y.device)], spec)
    spec = spectral_normalize_torch(spec)

    return spec


def spectrogram(y, n_fft, num_mels, sampling_rate, hop_size, win_size, fmin, fmax, center=False):
    if torch.min(y) < -1.:
        print('min value is ', torch.min(y))
    if torch.max(y) > 1.:
        print('max value is ', torch.max(y))

    global hann_window
    hann_window[str(y.device)] = torch.hann_window(win_size).to(y.device)

    y = torch.nn.functional.pad(y.unsqueeze(1), (int((n_fft-hop_size)/2), int((n_fft-hop_size)/2)), mode='reflect')
    y = y.squeeze(1)

    # complex tensor as default, then use view_as_real for future pytorch compatibility
    spec = torch.stft(y, n_fft, hop_length=hop_size, win_length=win_size, window=hann_window[str(y.device)],
                      center=center, pad_mode='reflect', normalized=False, onesided=True, return_complex=True)
    spec = torch.view_as_real(spec)
    spec = torch.sqrt(spec.pow(2).sum(-1)+(1e-9))

    return spec


def normalize_spectrogram(
    spectrogram: torch.Tensor,
    max_value: float = 200, 
    min_value: float = 1e-5, 
    power: float = 1., 
    inverse: bool = False
) -> torch.Tensor:
    # Rescale to 0-1
    max_value = np.log(max_value) # 5.298317366548036
    min_value = np.log(min_value) # -11.512925464970229

    assert spectrogram.max() <= max_value and spectrogram.min() >= min_value

    data = (spectrogram - min_value) / (max_value - min_value)

    # Invert
    if inverse:
        data = 1 - data

    # Apply the power curve
    data = torch.pow(data, power)  
    
    # 1D -> 3D
    data = data.unsqueeze(1)
    # data = data.repeat(1, 3, 1, 1)
    # (b f) (h w) c -> b f (h w) c -> b t (h w) c -> b t (h' w') c 

    # Flip Y axis: image origin at the top-left corner, spectrogram origin at the bottom-left corner
    data = torch.flip(data, [1])

    return data

def denormalize_spectrogram(
    data: torch.Tensor,
    max_value: float = 200, 
    min_value: float = 1e-5, 
    power: float = 1, 
    inverse: bool = False,
) -> torch.Tensor:
    
    max_value = np.log(max_value)
    min_value = np.log(min_value)

    # Flip Y axis: image origin at the top-left corner, spectrogram origin at the bottom-left corner
    data = torch.flip(data, [1])

    assert len(data.shape) == 3, "Expected 3 dimensions, got {}".format(len(data.shape))
    
    if data.shape[0] == 1:
        data = data.repeat(3, 1, 1)
        
    assert data.shape[0] == 3, "Expected 3 channels, got {}".format(data.shape[0])
    data = data[0]

    # Reverse the power curve
    data = torch.pow(data, 1 / power)

    # Invert
    if inverse:
        data = 1 - data

    # Rescale to max value
    spectrogram = data * (max_value - min_value) + min_value

    return spectrogram


def get_mel_spectrogram_from_audio(audio):
    # for auffusion 
    spec = mel_spectrogram(audio, n_fft=2048, num_mels=256, sampling_rate=16000, hop_size=160, win_size=1024, fmin=0, fmax=8000, center=False)

    # for audioldm
    # spec = mel_spectrogram(audio, n_fft=1024, num_mels=64, sampling_rate=16000, hop_size=160, win_size=1024, fmin=0, fmax=8000, center=False)
    spec = normalize_spectrogram(spec)
    return spec