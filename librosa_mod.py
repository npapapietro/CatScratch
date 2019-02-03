import numpy as np
import scipy
import scipy.fftpack as fft
import scipy.signal

#
#  See https://librosa.github.io/librosa/generated/librosa.feature.mfcc.html
#  This file aims to reproduce the librosa.feature.mfcc function
#  but because that library is not supported in the raspberrypi.
#  I just pulled the related functions that are needed for this 
#  and threw them together so minimal packages are needed.
#


def pad_center(data, size, **kwargs):

    kwargs.setdefault('mode', 'constant')

    n = data.shape[-1]

    lpad = int((size - n) // 2)

    lengths = [(0, 0)] * data.ndim
    lengths[-1] = (lpad, int(size - n - lpad))

    return np.pad(data, lengths, **kwargs)

def frame(y, frame_length=2048, hop_length=512):
    n_frames = 1 + int((len(y) - frame_length) / hop_length)
    y_frames = np.lib.stride_tricks.as_strided(y, shape=(frame_length, n_frames),
                          strides=(y.itemsize, hop_length * y.itemsize))
    return y_frames
        
def stft(y, n_fft=2048, dtype=np.complex64):
    
    fft_window = scipy.signal.get_window('hann', n_fft)

    # Pad the window out to n_fft size
    fft_window = pad_center(fft_window, n_fft)

    # Reshape so that the window can be broadcast
    fft_window = fft_window.reshape((-1, 1))

    y = np.pad(y, int(n_fft // 2), mode='reflect')

    # Window the time series.
    y_frames = frame(y, frame_length=n_fft)

    # Pre-allocate the STFT matrix
    stft_matrix = np.empty((int(1 + n_fft // 2), y_frames.shape[1]),
                           dtype=dtype,
                           order='F')
    MAX_MEM_BLOCK = 2**8 * 2**10
    # how many columns can we fit within MAX_MEM_BLOCK?
    n_columns = int(MAX_MEM_BLOCK / (stft_matrix.shape[0] *
                                          stft_matrix.itemsize))

    for bl_s in range(0, stft_matrix.shape[1], n_columns):
        bl_t = min(bl_s + n_columns, stft_matrix.shape[1])

        stft_matrix[:, bl_s:bl_t] = fft.fft(fft_window *
                                            y_frames[:, bl_s:bl_t],
                                            axis=0)[:stft_matrix.shape[0]]

    return stft_matrix                          
                                                                               
def power_to_db(S):
    S = np.asarray(S)

    if np.issubdtype(S.dtype, np.complexfloating):
        magnitude = np.abs(S)
    else:
        magnitude = S
    ref=1.0
    amin=1e-10
    ref_value = np.abs(ref)

    log_spec = 10.0 * np.log10(np.maximum(amin, magnitude))
    log_spec -= 10.0 * np.log10(np.maximum(amin, ref_value))

    return np.maximum(log_spec, log_spec.max() - 80.0)

def hz_to_mel(frequencies):

    frequencies = np.asanyarray(frequencies)

    # Fill in the linear part
    f_min = 0.0
    f_sp = 200.0 / 3

    mels = (frequencies - f_min) / f_sp

    # Fill in the log-scale part

    min_log_hz = 1000.0                         # beginning of log region (Hz)
    min_log_mel = (min_log_hz - f_min) / f_sp   # same (Mels)
    logstep = np.log(6.4) / 27.0                # step size for log region

    if frequencies.ndim:
        # If we have array data, vectorize
        log_t = (frequencies >= min_log_hz)
        mels[log_t] = min_log_mel + np.log(frequencies[log_t]/min_log_hz) / logstep
    elif frequencies >= min_log_hz:
        # If we have scalar data, heck directly
        mels = min_log_mel + np.log(frequencies / min_log_hz) / logstep

    return mels

def mel_to_hz(mels):
    mels = np.asanyarray(mels)

    # Fill in the linear scale
    f_min = 0.0
    f_sp = 200.0 / 3
    freqs = f_min + f_sp * mels

    # And now the nonlinear scale
    min_log_hz = 1000.0                         # beginning of log region (Hz)
    min_log_mel = (min_log_hz - f_min) / f_sp   # same (Mels)
    logstep = np.log(6.4) / 27.0                # step size for log region

    if mels.ndim:
        # If we have vector data, vectorize
        log_t = (mels >= min_log_mel)
        freqs[log_t] = min_log_hz * np.exp(logstep * (mels[log_t] - min_log_mel))
    elif mels >= min_log_mel:
        # If we have scalar data, check directly
        freqs = min_log_hz * np.exp(logstep * (mels - min_log_mel))

    return freqs


def mel_frequencies(n_mels=128, fmin=0.0, fmax=11025.0):
    # 'Center freqs' of mel bands - uniformly spaced between limits
    min_mel = hz_to_mel(fmin)
    max_mel = hz_to_mel(fmax)

    mels = np.linspace(min_mel, max_mel, n_mels)

    return mel_to_hz(mels)

def mel(sr, n_fft):
    n_mels=128
    fmax = float(sr) / 2

    # Initialize the weights
    weights = np.zeros((n_mels, int(1 + n_fft // 2)))

    # Center freqs of each FFT bin
    fftfreqs = np.linspace(0, float(sr) / 2, int(1 + n_fft//2), endpoint=True)

    # 'Center freqs' of mel bands - uniformly spaced between limits
    mel_f = mel_frequencies(n_mels + 2, fmin=0, fmax=fmax)

    fdiff = np.diff(mel_f)
    ramps = np.subtract.outer(mel_f, fftfreqs)

    for i in range(n_mels):
        # lower and upper slopes for all bins
        lower = -ramps[i] / fdiff[i]
        upper = ramps[i+2] / fdiff[i+1]

        # .. then intersect them with each other and zero
        weights[i] = np.maximum(0, np.minimum(lower, upper))

    
    enorm = 2.0 / (mel_f[2:n_mels+2] - mel_f[:n_mels])
    weights *= enorm[:, np.newaxis]

    return weights   

def melspectrogram(y, sr):
    n_fft=2048
    d = stft(y, n_fft=n_fft)
    S = np.abs(d)**2.0

    # Build a Mel filter
    mel_basis = mel(sr, n_fft)

    return np.dot(mel_basis, S)                                 
def mfcc(y, sr, n_mfcc=40):
    S = power_to_db(melspectrogram(y=y, sr=sr))

    return fft.dct(S, axis=0, type=2, norm='ortho')[:n_mfcc]
