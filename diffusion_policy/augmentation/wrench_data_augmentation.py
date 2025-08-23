import numpy as np
from scipy.fft import fft, ifft
import matplotlib.pyplot as plt
import h5py
from IPython import embed




# --- Load Wrench Data ---
# file_path = "/home/pannaga/Documents/ARM_Lab/robot_tool_2025S/ep_1.h5"
file_path = "/home/pannaga/Documents/ARM_Lab/robot_tool_2025S/ds_processed.h5"
with h5py.File(file_path, 'r') as f:
    # wrench_data = f['left_arm']['wrench']['data'][:]
    wrench_data = f['data']['wrench_data'][:]  # shape: (T, 6)

# --- FFT-Based Augmentation ---
def fft_augment(wrench, noise_scale=0.05, mode='all', k=5, sr=200):
    """
    Augments wrench data in frequency domain.

    Args:
        wrench: (T, 6) time-series wrench data
        noise_scale: scale of the jitter noise
        mode: 'all' → jitter all frequencies, 'topk' → jitter top-k dominant freqs
        k: number of top-k frequencies to jitter if mode == 'topk'
        sr: sampling rate (Hz)
    
    Returns:
        wrench_aug: augmented wrench data (same shape)
    """
    wrench_aug = wrench.copy()
    N = wrench.shape[0]

    for i in range(6):
        f = fft(wrench[:, i])

        if mode == 'all':
            noise = np.random.randn(*f.shape) * noise_scale
            f_aug = f * (1 + noise)

        elif mode == 'topk':
            mag = np.abs(f)
            top_k_indices = np.argsort(mag[1:])[-k:] + 1  # skip DC component at index 0
            f_aug = f.copy()
            for idx in top_k_indices:
                jitter = 1 + np.random.normal(0, noise_scale)
                f_aug[idx] *= jitter
                f_aug[-idx] = np.conj(f_aug[idx])  # ensure symmetry for real signal

        else:
            raise ValueError("Invalid mode. Choose from ['all', 'topk']")

        wrench_aug[:, i] = np.real(ifft(f_aug))

    return wrench_aug

# --- Visualization: Overlay Time Domain of Original vs Augmented ---
def overlay_fft_comparison_with_freq(original, augmented, sr=200, channel=1):
    N = len(original)
    duration = N / sr
    t = np.linspace(0, duration, N)
    
    # FFT of original
    f_orig = fft(original)
    freq = np.fft.fftfreq(N, d=1/sr)
    idx = np.where(freq >= 0)
    freq = freq[idx]
    f_orig = f_orig[idx]

    plt.figure(figsize=(12, 6))

    # Frequency Domain
    plt.subplot(121)
    plt.stem(freq, np.abs(f_orig), linefmt='b-', markerfmt=' ', basefmt=' ')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude |X(f)|')
    plt.title('Original FFT (Channel {})'.format(channel))
    plt.xlim(0, 10)

    # Time Domain
    plt.subplot(122)
    plt.plot(t, original, 'r', label='Original')
    plt.plot(t, augmented, 'g--', label='Augmented (IFFT)')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.title('Time Domain Overlay')
    plt.legend()

    plt.tight_layout()
    plt.show()

def overlay_fft_comparison(original, augmented, sr=200, channel=1):
    N = len(original)
    duration = N / sr
    t = np.linspace(0, duration, N)

    # FFT of original
    f_orig = fft(original)
    freq = np.fft.fftfreq(N, d=1/sr)
    idx = np.where(freq >= 0)
    freq = freq[idx]
    f_orig = f_orig[idx]

    plt.figure(figsize=(12, 6))
    plt.plot(t, original, 'r', label='Original')
    plt.plot(t, augmented, 'g--', label='Augmented (IFFT)')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.title('Time Domain Overlay')
    plt.legend()

    plt.tight_layout()
    plt.grid()
    plt.show()

import matplotlib.pyplot as plt
import seaborn as sns

# Apply a clean style
sns.set(style="whitegrid")

sns.set(style="whitegrid")

def plot_all_channels_overlay(original, augmented, sr=200):
    """
    Plots all 6 channels (0 to 5) of wrench data: original vs augmented.
    """
    N = len(original)
    duration = N / sr
    t = np.linspace(0, duration, N)
    
    channel_labels = ['Fx', 'Fy', 'Fz', 'Tx', 'Ty', 'Tz']  # optional: can switch to ['0', '1', ..., '5']

    plt.figure(figsize=(16, 12))
    
    for i in range(6):
        plt.subplot(3, 2, i + 1)
        plt.plot(t, original[:, i], label='Original', color='royalblue', linewidth=1.5)
        plt.plot(t, augmented[:, i], label='Augmented', color='seagreen', linestyle='--', linewidth=1.5)
        plt.title(f'Channel {i} ({channel_labels[i]})')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.grid(True)
        if i == 0:
            plt.legend()
    
    plt.tight_layout()
    plt.show()



# --- Run the Pipeline ---
sr = 1000
wrench_aug = fft_augment(wrench_data, noise_scale= 0.01, mode='topk', sr=sr)
# overlay_fft_comparison_with_freq(wrench_data[:, 1], wrench_aug[:, 1], sr=200, channel=1)
# overlay_fft_comparison(wrench_data , wrench_aug , sr=sr, channel=1)

# plot_all_channels_overlay(wrench_data, wrench_aug, sr=sr)
embed()