import numpy as np
from trf import build_design_matrix, train_trf, predict_trf
from contrastive_NMF import contrastive_NMF


def train_contrastive_nmf_with_trf(
    eeg_train,      # np.array (C, T)	EEG 延迟展开前的训练数据
    audio_mix_mag,  # np.array (F, T)	混音音频的 magnitude STFT
    h_target,       # np.array (K, T)	来自目标 source 的 solo NMF 激活
    w_init,
    h_init,
    lags=list(range(0, 65)),
    delta=1000,
    mu=1,
    beta=1,
    alpha=0.1,
    outer_iter=20,
    inner_iter=300,
    nr_src=2,
):
    """
    完整复现伪代码的双层循环：外层更新 TRF，内层运行 Contrastive-NMF

    参数：
    - eeg_train: np.array (C, T), EEG 训练数据
    - audio_mix_mag: np.array (F, T), 混音音频的幅度谱
    - h_target: np.array (K, T), 目标 source 的 NMF 激活（用于训练 TRF）
    - w_init: np.array (F, K), W 初始化
    - h_init: np.array (K, T), H 初始化
    - lags: list of int, EEG 延迟
    - delta, mu, beta: Contrastive-NMF 参数
    - alpha: TRF 正则化
    - outer_iter: 外层循环次数（TRF + NMF）
    - inner_iter: 每轮 Contrastive-NMF 的迭代次数
    - nr_src: 源数量

    返回：最终的 W, H, h_tilde
    """
    print("[INFO] Start dual-loop training...")

    R = build_design_matrix(eeg_train, lags)

    # ✅ 裁剪
    T_align = min(R.shape[1], h_target.shape[1])
    R = R[:, :T_align]
    h_target = h_target[:, :T_align]

    g, scaler = train_trf(R, h_target, alpha=alpha)

    w = w_init.copy()
    h = h_init.copy()

    for epoch in range(outer_iter):
        print(f"[INFO] Outer loop {epoch+1}/{outer_iter}...")

        # Step 1: 预测 Sa（h_tilde）
        R = build_design_matrix(eeg_train, lags)
        h_tilde = predict_trf(R, g, scaler)

        min_T = min(h_tilde.shape[1], h.shape[1])
        h_tilde = h_tilde[:, :min_T]
        h = h[:, :min_T]

        # Step 2: Contrastive NMF（固定 g）
        w, h, cost = contrastive_NMF(
            v=audio_mix_mag,
            w_init=w,
            h_init=h,
            h_tilde=h_tilde,
            delta=delta,
            mu=mu,
            beta=beta,
            n_iter=inner_iter,
            nr_src=nr_src,
        )

        # Step 3: 更新 TRF（update g）
        T = min(R.shape[1], h.shape[1])
        R = R[:, :T]
        h = h[:, :T]
        g, scaler = train_trf(R, h, alpha=alpha)

    print("[INFO] Training complete.")
    return w, h, h_tilde
