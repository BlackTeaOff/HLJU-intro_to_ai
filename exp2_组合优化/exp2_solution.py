#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""实验二 组合优化：Otsu + PSO —— 参考实现（教师版）

必做题 3 题完整答案 + 进阶题（多阈值 Otsu + PSO，用三峰图）。
学生版见 exp2_scaffold.py。
依赖：numpy + matplotlib
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt


def setup_chinese_font():
    import matplotlib.font_manager as fm
    candidates = [
        "Microsoft YaHei", "SimHei", "SimSun",
        "PingFang SC", "Hiragino Sans GB", "Arial Unicode MS", "STHeiti", "Songti SC",
        "Noto Sans CJK SC", "Source Han Sans SC",
    ]
    available = {f.name for f in fm.fontManager.ttflist}
    chosen = None
    for name in candidates:
        if name in available:
            chosen = name
            break
    if chosen is None:
        for f in fm.fontManager.ttflist:
            if any(k in f.name for k in ("YaHei", "Hei", "Song", "CJK", "PingFang")):
                chosen = f.name
                break
    if chosen:
        matplotlib.rcParams["font.sans-serif"] = [chosen]
    matplotlib.rcParams["axes.unicode_minus"] = False


# ---------- 测试图 ----------

def make_test_image(size=256, seed=42):
    """合成「双峰」灰度图：暗背景(~60) + 两个亮圆盘(~200) + 高斯噪声。

    直方图呈双峰，Otsu 类间方差最大化能找到分割阈值。
    """
    rng = np.random.default_rng(seed)
    img = np.full((size, size), 60.0)
    yy, xx = np.mgrid[0:size, 0:size]
    disk1 = (xx - 90) ** 2 + (yy - 110) ** 2 < 45 ** 2
    disk2 = (xx - 170) ** 2 + (yy - 150) ** 2 < 35 ** 2
    img[disk1 | disk2] = 200.0
    img += rng.normal(0, 8, (size, size))
    return np.clip(img, 0, 255).astype(np.uint8)


def make_test_image_3class(size=256, seed=7):
    """合成「三峰」灰度图（进阶题用）：暗背景(~40) + 中灰方块(~130) + 亮圆盘(~220)。"""
    rng = np.random.default_rng(seed)
    img = np.full((size, size), 40.0)
    yy, xx = np.mgrid[0:size, 0:size]
    square = (xx > 140) & (xx < 210) & (yy > 60) & (yy < 200)
    disk = (xx - 80) ** 2 + (yy - 90) ** 2 < 55 ** 2
    img[square] = 130.0
    img[disk] = 220.0
    img += rng.normal(0, 7, (size, size))
    return np.clip(img, 0, 255).astype(np.uint8)


# ---------- 必做题 ----------

def otsu_score(img, t):
    """题1 核心：阈值 t 的「类间方差」（Otsu 目标函数，越大分割越好）。

    返回 ω0·ω1·(μ0 − μ1)²，ω0/ω1 为前景/背景像素占比，μ0/μ1 为两类均值。
    """
    t = int(round(t))
    t = min(max(t, 1), 255)
    hist = np.bincount(img.ravel(), minlength=256).astype(np.float64)
    total = img.size
    n0 = hist[:t].sum()
    n1 = hist[t:].sum()
    if n0 == 0 or n1 == 0:
        return 0.0
    w0, w1 = n0 / total, n1 / total
    mu0 = (hist[:t] * np.arange(t)).sum() / n0
    mu1 = (hist[t:] * np.arange(t, 256)).sum() / n1
    return w0 * w1 * (mu0 - mu1) ** 2


def exhaustive_otsu(img):
    """题1：穷举 1~255 所有阈值，返回 (最优阈值, 最优类间方差)。"""
    scores = [otsu_score(img, t) for t in range(1, 256)]
    t_best = int(np.argmax(scores)) + 1
    return t_best, scores[t_best - 1]


def pso_otsu(img, n_particles=30, n_iter=50, w=0.7, c1=1.5, c2=1.5, seed=0):
    """题2：PSO 寻优。粒子位置 = 阈值，适应度 = 类间方差。

    更新式：v ← w·v + c1·r1·(pbest − x) + c2·r2·(gbest − x)，x ← x + v
    返回 (最优阈值, 最优适应度, 每代 gbest 适应度历史)。
    """
    rng = np.random.default_rng(seed)
    x = rng.uniform(0, 60, n_particles)         # 位置 = 候选阈值（初始集中在低端，便于观察收敛过程）
    v = rng.uniform(-10, 10, n_particles)       # 速度
    pbest_x = x.copy()
    pbest_f = np.array([otsu_score(img, xi) for xi in x])
    gbest_idx = int(np.argmax(pbest_f))
    gbest_x, gbest_f = pbest_x[gbest_idx], pbest_f[gbest_idx]
    history = [gbest_f]

    for _ in range(n_iter):
        r1, r2 = rng.random(n_particles), rng.random(n_particles)
        v = w * v + c1 * r1 * (pbest_x - x) + c2 * r2 * (gbest_x - x)
        x = np.clip(x + v, 0, 255)
        f = np.array([otsu_score(img, xi) for xi in x])
        improved = f > pbest_f
        pbest_x[improved] = x[improved]
        pbest_f[improved] = f[improved]
        if pbest_f.max() > gbest_f:
            gbest_idx = int(np.argmax(pbest_f))
            gbest_x, gbest_f = pbest_x[gbest_idx], pbest_f[gbest_idx]
        history.append(gbest_f)
    return gbest_x, gbest_f, history


def visualize_segmentation(img, t):
    """题1 可视化：原图 / 直方图+阈值 / 二值分割结果。"""
    setup_chinese_font()
    binary = (img >= t).astype(np.uint8) * 255
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    axes[0].imshow(img, cmap="gray"); axes[0].set_title("原图（双峰）"); axes[0].axis("off")
    axes[1].hist(img.ravel(), bins=256, range=(0, 255), color="#888888")
    axes[1].axvline(t, color="r", linestyle="--", linewidth=2, label=f"Otsu 阈值={t}")
    axes[1].set_title("灰度直方图"); axes[1].legend()
    axes[2].imshow(binary, cmap="gray"); axes[2].set_title(f"分割结果（阈值 {t}）"); axes[2].axis("off")
    plt.tight_layout()
    plt.savefig("otsu_segmentation.png", dpi=150, bbox_inches="tight")
    print("已保存分割可视化到 otsu_segmentation.png")


def visualize_convergence(history, exhaustive_score):
    """题3 可视化：PSO 收敛曲线 vs 穷举最优值水平线。"""
    setup_chinese_font()
    plt.figure(figsize=(8, 5))
    plt.plot(history, "o-", markersize=3, label="PSO 每代 gbest 适应度")
    plt.axhline(exhaustive_score, color="r", linestyle="--",
                label=f"穷举最优值 {exhaustive_score:.2f}")
    plt.xlabel("迭代代数"); plt.ylabel("类间方差（适应度）")
    plt.title("PSO 收敛曲线 vs 穷举最优")
    plt.legend(); plt.tight_layout()
    plt.savefig("pso_convergence.png", dpi=150, bbox_inches="tight")
    print("已保存收敛曲线到 pso_convergence.png")


# ---------- 进阶题（选做加分） ----------

def otsu_score_multi(img, ts):
    """题4：多阈值 Otsu 类间方差。ts = 升序阈值列表。

    类间方差 = Σ_c ω_c · (μ_c − μ_T)²，ω_c/μ_c 为第 c 类占比/均值，μ_T 为全局均值。
    """
    hist = np.bincount(img.ravel(), minlength=256).astype(np.float64)
    total = img.size
    mu_T = (hist * np.arange(256)).sum() / total
    bounds = sorted(set([0] + [min(max(int(round(t)), 1), 255) for t in ts] + [256]))
    sigma_b = 0.0
    for i in range(len(bounds) - 1):
        lo, hi = bounds[i], bounds[i + 1]
        if lo >= hi:
            continue
        w = hist[lo:hi].sum() / total
        if w == 0:
            continue
        mu = (hist[lo:hi] * np.arange(lo, hi)).sum() / hist[lo:hi].sum()
        sigma_b += w * (mu - mu_T) ** 2
    return sigma_b


def pso_otsu_multi(img, k=2, n_particles=30, n_iter=100, w=0.7, c1=1.5, c2=1.5, seed=0):
    """题4：多阈值 PSO。粒子为 k 维阈值向量（升序）。返回 (阈值向量, 适应度, 历史)。"""
    rng = np.random.default_rng(seed)
    x = rng.uniform(0, 60, (n_particles, k))   # 初始集中在低端，便于观察收敛
    x.sort(axis=1)
    v = rng.uniform(-10, 10, (n_particles, k))
    pbest_x = x.copy()
    pbest_f = np.array([otsu_score_multi(img, xi) for xi in x])
    gbest_idx = int(np.argmax(pbest_f))
    gbest_x, gbest_f = pbest_x[gbest_idx].copy(), pbest_f[gbest_idx]
    history = [gbest_f]
    for _ in range(n_iter):
        r1 = rng.random((n_particles, k)); r2 = rng.random((n_particles, k))
        v = w * v + c1 * r1 * (pbest_x - x) + c2 * r2 * (gbest_x - x)
        x = np.clip(x + v, 0, 255)
        x.sort(axis=1)
        f = np.array([otsu_score_multi(img, xi) for xi in x])
        improved = f > pbest_f
        pbest_x[improved] = x[improved]
        pbest_f[improved] = f[improved]
        if pbest_f.max() > gbest_f:
            gbest_idx = int(np.argmax(pbest_f))
            gbest_x, gbest_f = pbest_x[gbest_idx].copy(), pbest_f[gbest_idx]
        history.append(gbest_f)
    return gbest_x, gbest_f, history


def split_planes(img, thresholds):
    """题4：按阈值把灰度图分解成多个「二值面」——每个面 = 一个灰度区间，区间内像素为 1，其余为 0。

    thresholds: 升序阈值列表 [t1, t2, ...]，把 [0, 255] 分成 len(thresholds)+1 个区间。
    返回 planes（list，每个元素是 0/1 uint8 掩码，表示一个灰度层）。
    """
    ts = [0] + sorted(int(round(t)) for t in thresholds) + [256]
    planes = []
    for i in range(len(ts) - 1):
        lo, hi = ts[i], ts[i + 1]
        planes.append(((img >= lo) & (img < hi)).astype(np.uint8))
    return planes


def visualize_planes(img, thresholds, out_path="otsu_planes.png"):
    """题4 可视化：原图 + 按阈值分解出的多个二值面（教师已给，可直接调用）。"""
    setup_chinese_font()
    ts = [0] + sorted(int(round(t)) for t in thresholds) + [256]
    planes = split_planes(img, thresholds)
    n = len(planes)
    fig, axes = plt.subplots(1, n + 1, figsize=(3.2 * (n + 1), 3.4))
    axes[0].imshow(img, cmap="gray"); axes[0].set_title("原图"); axes[0].axis("off")
    for i in range(n):
        axes[i + 1].imshow(planes[i], cmap="gray")
        axes[i + 1].set_title(f"面{i+1} 灰度[{ts[i]},{ts[i+1]})")
        axes[i + 1].axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"已保存二值面分解到 {out_path}")
    return planes


def main():
    img = make_test_image()
    print(f"双峰测试图：尺寸 {img.shape}，均值 {img.mean():.1f}，"
          f"直方图峰值 {np.bincount(img.ravel()).argmax()}")

    # 题1 穷举 Otsu
    t_ex, f_ex = exhaustive_otsu(img)
    print(f"\n[题1] 穷举 Otsu 最优阈值 = {t_ex}，类间方差 = {f_ex:.4f}")
    visualize_segmentation(img, t_ex)

    # 题2 PSO
    t_pso, f_pso, history = pso_otsu(img)
    print(f"[题2] PSO 最优阈值 = {t_pso:.2f}，适应度 = {f_pso:.4f}")

    # 题3 对比
    print(f"\n[题3] 穷举阈值 {t_ex} vs PSO 阈值 {t_pso:.2f}")
    print(f"      适应度一致：{abs(f_pso - f_ex) < 1e-6}（这才是 PSO 收敛的正确判据）")
    print(f"      阈值可能不同：Otsu 目标函数在两峰之间是「平台」——谷底无像素，"
          f"平台内任意阈值分割效果相同")
    visualize_convergence(history, f_ex)

    # 题4 进阶：三峰图 + 多阈值
    img3 = make_test_image_3class()
    print("\n[题4·进阶] 三峰图 + 多阈值 PSO（K=2）：")
    t2, f2, _ = pso_otsu_multi(img3, k=2)
    print(f"  PSO 双阈值 = {np.round(t2, 1).tolist()}，类间方差 = {f2:.4f}")
    best, best_ts = -1, None
    for t1 in range(1, 255):
        for t2v in range(t1 + 1, 256):
            s = otsu_score_multi(img3, [t1, t2v])
            if s > best:
                best, best_ts = s, (t1, t2v)
    print(f"  穷举双阈值最优 = {list(best_ts)}，类间方差 = {best:.4f}")
    print(f"  → PSO 逼近穷举最优（适应度 {f2:.4f} vs {best:.4f}），"
          f"且无需遍历 {255*254//2} 种组合")
    # 题4 可视化：把三峰图按双阈值分解成 3 个二值面，逐面打印
    visualize_planes(img3, t2)


if __name__ == "__main__":
    main()
