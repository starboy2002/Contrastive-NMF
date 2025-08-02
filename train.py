import os
import numpy as np
import librosa
from glob import glob
from sklearn.decomposition import NMF
from trf import build_design_matrix
from train_contrastive_nmf import train_contrastive_nmf_with_trf
from torchmetrics.audio import ScaleInvariantSignalDistortionRatio
import torch

# ========== 路径配置 ==========
data_root = r"E:\Projects\AI_Music\Neuro_Music\processed_data_MADEEEG"
eeg_dir = os.path.join(data_root, "response_npy")
solo_dir = os.path.join(data_root, "isolated_wav")
mix_dir = os.path.join(data_root, "stimulus_wav")

# ========== 遍历处理每个样本 ==========
eeg_files = sorted(glob(os.path.join(eeg_dir, "*_response.npy")))

for eeg_path in eeg_files:
    basename = os.path.basename(eeg_path).replace("_response.npy", "")
    solo_path = os.path.join(solo_dir, f"{basename}_soli.wav")
    mix_path = os.path.join(mix_dir, f"{basename}_stimulus.wav")

    if not os.path.exists(solo_path) or not os.path.exists(mix_path):
        print(f"[WARNING] Missing audio for {basename}, skipping.")
        continue

    print("\n[INFO] Processing:", basename)
    print("[INFO] EEG:", eeg_path)
    print("[INFO] SOLO:", solo_path)
    print("[INFO] MIXED:", mix_path)

    # ========== 加载数据 ==========
    eeg = np.load(eeg_path)  # shape: [C, T]
    y_solo, sr = librosa.load(solo_path, sr=None)
    y_mix, _ = librosa.load(mix_path, sr=sr)

    # ========== Step 1: 提取 H_target ==========
    S_solo = librosa.stft(y_solo, n_fft=1024, hop_length=512)
    V_solo = np.abs(S_solo)
    nmf = NMF(n_components=16, init='random', solver='mu', beta_loss='kullback-leibler', max_iter=200)
    W_solo = nmf.fit_transform(V_solo)
    H_target = nmf.components_  # shape: [K, T]

    # ========== Step 2: 构造 EEG 延迟特征 ==========
    lags = list(range(0, 65))
    R = build_design_matrix(eeg, lags)
    T = min(R.shape[1], H_target.shape[1])
    R = R[:, :T]
    H_target = H_target[:, :T]

    # ========== Step 3: 混音音频 STFT ==========
    S_mix = librosa.stft(y_mix, n_fft=1024, hop_length=512)
    V_mix = np.abs(S_mix)
    phase_mix = np.angle(S_mix)
    F, T = V_mix.shape
    K = 32

    # ========== Step 4: 初始化 W, H ==========
    # w_init = np.abs(np.random.rand(F, K))
    # h_init = np.abs(np.random.rand(K, T))

    nmf_init = NMF(n_components=K, init='random', solver='mu', beta_loss='kullback-leibler', max_iter=200)
    w_init = nmf_init.fit_transform(V_mix)
    h_init = nmf_init.components_


    # ========== Step 5: 外层 TRF + NMF 双循环训练 ==========
    w_final, h_final, h_tilde = train_contrastive_nmf_with_trf(
        eeg_train=eeg,
        audio_mix_mag=V_mix,
        h_target=H_target,
        w_init=w_init,
        h_init=h_init,
        lags=lags,
        delta=10000,
        mu=10,
        beta=10,
        alpha=0.1,
        outer_iter=10,
        inner_iter=400,
        nr_src=2
    )

    # ========== Step 6: 重建音频 ==========
    Lambda = np.dot(w_final, h_final)
    mask = (np.dot(w_final[:, :K//2], h_final[:K//2, :])) / (Lambda + 1e-9)
    S_hat = mask * S_mix
    y_hat = librosa.istft(S_hat)

    # ========== Step 7: SISDR 评估 ==========
    min_len = min(len(y_hat), len(y_solo))
    y_hat = y_hat[:min_len]
    y_solo = y_solo[:min_len]
    sisdr = ScaleInvariantSignalDistortionRatio()
    score = sisdr(torch.tensor(y_hat), torch.tensor(y_solo))
    print(f"[RESULT] SI-SDR for {basename}: {score.item():.2f} dB")

    # ========== Step 8: 保存重建音频 ==========
    import soundfile as sf
    output_dir = "reconstructed_audio"
    os.makedirs(output_dir, exist_ok=True)

    out_path = os.path.join(output_dir, f"{basename}_estimate.wav")
    sf.write(out_path, y_hat, samplerate=sr, subtype='PCM_16')
    print(f"[INFO] Saved estimated audio to {out_path}")
