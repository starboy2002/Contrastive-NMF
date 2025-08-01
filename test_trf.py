import numpy as np
import librosa
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import NMF

from trf import build_design_matrix, train_trf, predict_trf


def get_H_target(audio_path, n_fft=1024, hop_length=512, K=16):
    y, sr = librosa.load(audio_path, sr=None)
    S = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
    V = np.abs(S)  # [F, T]

    model = NMF(n_components=K, init='random', solver='mu', beta_loss='kullback-leibler', max_iter=200)
    W = model.fit_transform(V)
    H = model.components_  # [K, T]
    return H, V.shape[1], sr, hop_length


def match_time_length(eeg, H, sr_audio, hop_length_audio, eeg_sr):
    """
    将 EEG 和 H 对齐：返回裁剪后的 EEG 和 H
    """
    T_h = H.shape[1]
    audio_duration = T_h * hop_length_audio / sr_audio
    T_eeg_target = int(audio_duration * eeg_sr)

    eeg_cut = eeg[:, :T_eeg_target]
    H_cut = H[:, :min(T_eeg_target, T_h)]

    return eeg_cut, H_cut


if __name__ == "__main__":
    # 假设路径
    eeg_train = np.load("data/eeg_train.npy")      # shape: [C, T]
    eeg_test = np.load("data/eeg_test.npy")        # shape: [C, T_test]
    audio_path = "data/audio_solo.wav"

    # Step 1: Get H_target from solo audio
    H_target, T_audio, sr, hop = get_H_target(audio_path)
    print("H_target shape:", H_target.shape)

    # Step 2: Match EEG and H_target time
    eeg_sr = 256
    eeg_train, H_target = match_time_length(eeg_train, H_target, sr_audio=sr, hop_length_audio=hop, eeg_sr=eeg_sr)

    # Step 3: 构建设计矩阵
    lags = list(range(0, 65))  # 0~250ms for 256Hz
    R_train = build_design_matrix(eeg_train, lags)

    # Step 4: 训练 TRF
    G, scaler = train_trf(R_train, H_target, alpha=0.1)

    # Step 5: 对测试 EEG 做预测
    R_test = build_design_matrix(eeg_test, lags)
    H_pred = predict_trf(R_test, G, scaler)

    print("Predicted h_tilde shape:", H_pred.shape)

    # Step 6: 可视化对比
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    plt.imshow(H_target, aspect='auto', origin='lower')
    plt.title("True H_target (from solo NMF)")
    plt.subplot(2, 1, 2)
    plt.imshow(H_pred[:, :H_target.shape[1]], aspect='auto', origin='lower')
    plt.title("Predicted h_tilde (from EEG via TRF)")
    plt.tight_layout()
    plt.show()
