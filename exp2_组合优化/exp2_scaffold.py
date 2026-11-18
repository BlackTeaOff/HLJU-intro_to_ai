#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""实验二 组合优化：Otsu + PSO —— 学生脚手架

完成下面 3 处 TODO（对应 3 道必做题），即可运行出结果。
依赖：numpy + matplotlib
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt


def setup_chinese_font():
    """设置 matplotlib 中文字体（Windows/macOS/Linux 自动适配，教师已给，无需改动）。"""
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


def make_test_image(size=256, seed=42):
    """合成「双峰」灰度图：暗背景(~60) + 两个亮圆盘(~200) + 高斯噪声（教师已给）。"""
    rng = np.random.default_rng(seed)
    img = np.full((size, size), 60.0)
    yy, xx = np.mgrid[0:size, 0:size]
    disk1 = (xx - 90) ** 2 + (yy - 110) ** 2 < 45 ** 2
    disk2 = (xx - 170) ** 2 + (yy - 150) ** 2 < 35 ** 2
    img[disk1 | disk2] = 200.0
    img += rng.normal(0, 8, (size, size))
    return np.clip(img, 0, 255).astype(np.uint8)


def otsu_score(img, t):
    """TODO 1：阈值 t 的「类间方差」（Otsu 目标函数，越大分割越好）

    公式：σ_b² = ω0·ω1·(μ0 − μ1)²
    - ω0/ω1：前景(灰度 < t) / 背景(灰度 ≥ t) 的像素占比
    - μ0/μ1：前景 / 背景 的平均灰度

    提示：
    - 直方图：hist = np.bincount(img.ravel(), minlength=256)
    - 前景像素数 n0 = hist[:t].sum()，背景 n1 = hist[t:].sum()
    - 平均灰度 μ0 = (hist[:t] * np.arange(t)).sum() / n0
    - 边界：n0 或 n1 为 0 时返回 0.0（避免除零）
    """
    t = int(round(t))
    t = min(max(t, 1), 255)
    hist = np.bincount(img.ravel(), minlength=256).astype(np.float64)
    total = img.size
    # ============ 你的代码 ============
    n0 = hist[:t].sum()
    n1 = hist[t:].sum()

    if n0 == 0 or n1 == 0:
        return 0.0

    ω0 = n0 / total
    ω1 = n1 / total
    μ0 = ((hist[:t] * np.arange(t)).sum()) / n0 
    μ1 = ((hist[t:] * np.arange(t, 256)).sum()) / n1

    result = ω0 * ω1 * ((μ0 - μ1) ** 2)

    return result
    # ====================================


def exhaustive_otsu(img):
    """TODO 2：穷举搜索最优阈值

    遍历 t = 1 ~ 254，用 otsu_score 求每个阈值的类间方差，返回
    (最优阈值, 最优类间方差)。

    提示：scores = [otsu_score(img, t) for t in range(1, 256)]，
          用 np.argmax 找最大值下标。
    """
    # ============ 你的代码 ============
    scores = [otsu_score(img, t) for t in range(1, 256)]

    best_idx = int(np.argmax(scores))
    best_t = best_idx + 1
    best_score = scores[best_idx]

    return best_t, best_score
    # ====================================


def pso_otsu(img, n_particles=30, n_iter=50, w=0.7, c1=1.5, c2=1.5, seed=0):
    """TODO 3：粒子群优化（PSO）寻优

    粒子位置 = 候选阈值，适应度 = otsu_score。
    更新式：v ← w·v + c1·r1·(pbest − x) + c2·r2·(gbest − x)，x ← x + v

    步骤：
    1. 初始化：位置 x = uniform(0, 60)、速度 v = uniform(-10, 10)
    2. 记录每个粒子的个体最优 pbest_x / pbest_f，全局最优 gbest_x / gbest_f
    3. 每轮迭代：
       - 按更新式算新速度 v、新位置 x（用 np.clip 限制在 0~255）
       - 计算新适应度 f = otsu_score
       - 更新 pbest（f 比 pbest_f 大则更新）、gbest
       - 把每轮 gbest_f 记入 history
    返回 (gbest_x, gbest_f, history)

    提示：r1、r2 用 rng.random(n_particles) 生成随机数向量。
    """
    rng = np.random.default_rng(seed)
    x = rng.uniform(0, 60, n_particles)       # 位置 = 候选阈值
    v = rng.uniform(-10, 10, n_particles)     # 速度
    # ============ 你的代码（pbest/gbest 初始化 + 迭代循环）============
    pbest_x = x.copy()
    pbest_score = [0] * n_particles 
    gbest_x = x[0]
    gbest_score = 0
    history = []
    for step in range(n_iter):
        for i in range(n_particles):
            score = otsu_score(img, x[i])

            if score > pbest_score[i]:
                pbest_score[i] = score
                pbest_x[i] = x[i]

        best_particle_idx = np.argmax(pbest_score)
        if pbest_score[best_particle_idx] > gbest_score:
            gbest_score = pbest_score[best_particle_idx]
            gbest_x = pbest_x[best_particle_idx]

        history.append(gbest_score)

        r1 = rng.uniform(0, 1, size=n_particles)
        r2 = rng.uniform(0, 1, size=n_particles)
        
        v = w * v + c1 * r1 * (pbest_x - x) + c2 * r2 * (gbest_x - x)
        x = x + v
        x = np.clip(x, 1, 254)

    best_t = int(round(gbest_x))
    return best_t, gbest_score, history

            
    # ================================================================


def visualize_segmentation(img, t):
    """题1 可视化：原图 / 直方图+阈值 / 二值分割结果（教师已给，可直接运行）。"""
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
    """题3 可视化：PSO 收敛曲线 vs 穷举最优值水平线（教师已给）。"""
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


def main():
    img = make_test_image()
    print(f"双峰测试图：尺寸 {img.shape}，均值 {img.mean():.1f}")

    # 题1 穷举 Otsu
    t_ex, f_ex = exhaustive_otsu(img)
    print(f"\n[题1] 穷举 Otsu 最优阈值 = {t_ex}，类间方差 = {f_ex:.4f}")
    visualize_segmentation(img, t_ex)

    # 题2 PSO
    t_pso, f_pso, history = pso_otsu(img)
    print(f"[题2] PSO 最优阈值 = {t_pso:.2f}，适应度 = {f_pso:.4f}")

    # 题3 对比
    print(f"\n[题3] 穷举阈值 {t_ex} vs PSO 阈值 {t_pso:.2f}")
    print(f"      适应度一致：{abs(f_pso - f_ex) < 1e-6}")
    print("      思考：如果阈值不同但适应度相同，说明什么？（提示：Otsu 目标函数的形状）")
    visualize_convergence(history, f_ex)


# ================================================================
# 进阶题（选做加分）：多阈值 Otsu + PSO + 二值面分解
# 完成下面 2 处「进阶 TODO」后，运行 advanced_main() 看分解出的多个二值面。
# ================================================================

def make_test_image_3class(size=256, seed=7):
    """合成「三峰」灰度图：暗背景(~40) + 中灰方块(~130) + 亮圆盘(~220)（教师已给）。"""
    rng = np.random.default_rng(seed)
    img = np.full((size, size), 40.0)
    yy, xx = np.mgrid[0:size, 0:size]
    square = (xx > 140) & (xx < 210) & (yy > 60) & (yy < 200)
    disk = (xx - 80) ** 2 + (yy - 90) ** 2 < 55 ** 2
    img[square] = 130.0
    img[disk] = 220.0
    img += rng.normal(0, 7, (size, size))
    return np.clip(img, 0, 255).astype(np.uint8)


def otsu_score_multi(img, ts):
    """进阶 TODO 4：多阈值 Otsu 类间方差 = Σ_c ω_c·(μ_c − μ_T)²

    ts：升序阈值列表 [t1, t2, ...]。把直方图按阈值切成多段，每段算
    占比 ω_c 和均值 μ_c，μ_T 是全局均值。
    提示：先求全局均值 mu_T，再用 bounds=[0]+ts+[256] 循环每段累加。
    """
    hist = np.bincount(img.ravel(), minlength=256).astype(np.float64)
    total = img.size
    # ============ 你的代码 ============
    raise NotImplementedError("进阶 TODO 4：请实现多阈值 Otsu 类间方差")
    # ==================================


def pso_otsu_multi(img, k=2, n_particles=30, n_iter=100, w=0.7, c1=1.5, c2=1.5, seed=0):
    """进阶 TODO 5：多阈值 PSO。粒子 = k 维阈值向量（升序）。

    把单阈值 PSO 的「标量位置」改成「k 维向量」，每维独立更新速度/位置，
    更新后对每行 sort(axis=1) 保持阈值升序。返回 (阈值向量, 适应度, 历史)。
    提示：照抄 pso_otsu 的结构，只是 x/v 从 1 维数组变成 (n_particles, k) 矩阵。
    """
    rng = np.random.default_rng(seed)
    x = rng.uniform(0, 60, (n_particles, k))
    x.sort(axis=1)
    v = rng.uniform(-10, 10, (n_particles, k))
    # ============ 你的代码 ============
    raise NotImplementedError("进阶 TODO 5：请实现多阈值 PSO")
    # ==================================


def split_planes(img, thresholds):
    """把灰度图按阈值分解成多个「二值面」（教师已给，直接调用）。

    每个面 = 一个灰度区间，区间内像素为 1，其余为 0。
    thresholds: 升序阈值 [t1,t2,...]，分成 len(thresholds)+1 个区间。
    """
    ts = [0] + sorted(int(round(t)) for t in thresholds) + [256]
    planes = []
    for i in range(len(ts) - 1):
        lo, hi = ts[i], ts[i + 1]
        planes.append(((img >= lo) & (img < hi)).astype(np.uint8))
    return planes


def visualize_planes(img, thresholds, out_path="otsu_planes.png"):
    """直接打出分解出的多个二值面（教师已给，直接调用）。"""
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


def advanced_main():
    """进阶题主入口（完成进阶 TODO 4/5 后运行）。"""
    img3 = make_test_image_3class()
    t2, f2, _ = pso_otsu_multi(img3, k=2)
    print(f"[进阶] 多阈值 PSO 双阈值 = {np.round(t2, 1).tolist()}，类间方差 = {f2:.4f}")
    # 直接分别打出分解出的多个二值面
    visualize_planes(img3, t2)


if __name__ == "__main__":
    main()
    # 进阶题：完成上面两个「进阶 TODO」后，取消下面这行注释运行
    # advanced_main()
