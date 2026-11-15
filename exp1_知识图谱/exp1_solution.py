#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""实验一 知识图谱构建 —— 参考实现（教师版）

必做题 3 题的完整答案，以及进阶题（选做）的参考代码。
学生版见 exp1_scaffold.py。
"""

import networkx as nx
import matplotlib
import matplotlib.pyplot as plt

DATA_FILE = "triples.txt"


def setup_chinese_font():
    """设置 matplotlib 中文字体（Windows/macOS/Linux 自动适配，教师已给，无需改动）。"""
    import matplotlib.font_manager as fm
    candidates = [
        "Microsoft YaHei", "SimHei", "SimSun",            # Windows（中文版必自带微软雅黑）
        "PingFang SC", "Hiragino Sans GB", "Arial Unicode MS", "STHeiti", "Songti SC",  # macOS
        "Noto Sans CJK SC", "Source Han Sans SC",           # Linux / 通用
    ]
    available = {f.name for f in fm.fontManager.ttflist}
    chosen = None
    for name in candidates:
        if name in available:
            chosen = name
            break
    if chosen is None:  # 兑底：模糊匹配含中文关键字的字体
        for f in fm.fontManager.ttflist:
            if any(k in f.name for k in ("YaHei", "Hei", "Song", "CJK", "PingFang")):
                chosen = f.name
                break
    if chosen:
        matplotlib.rcParams["font.sans-serif"] = [chosen]
    matplotlib.rcParams["axes.unicode_minus"] = False


def load_triples(path):
    triples = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            h, r, t = line.split("\t")
            triples.append((h, r, t))
    return triples


# ---------- 必做题 ----------

def build_graph(triples):
    """题1（构建）：实体为节点、关系为有向边，关系存到边属性 'relation'。"""
    G = nx.DiGraph()
    for h, r, t in triples:
        G.add_edge(h, t, relation=r)
    return G


def related_entities(G, entity):
    """题2（一跳相关实体）：区分出边(out)与入边(in)。"""
    out = list(G.successors(entity))
    inn = list(G.predecessors(entity))
    return {"out": sorted(set(out)), "in": sorted(set(inn))}


def find_path(G, e1, e2):
    """题3（路径查询）：返回 (节点序列, 关系序列)，无路径返回 None。"""
    if not nx.has_path(G, e1, e2):
        return None
    nodes = nx.shortest_path(G, e1, e2)
    rels = [G[u][v]["relation"] for u, v in zip(nodes[:-1], nodes[1:])]
    return nodes, rels


# ---------- 进阶题（选做加分） ----------

def entity_importance(G):
    """题4：实体重要性——度中心性（入度+出度，即「连接了多少其他实体」）。

    返回按 (度中心性, 实体) 降序排列的列表，找出「枢纽实体」。
    """
    scores = [(G.in_degree(n) + G.out_degree(n), n) for n in G.nodes()]
    return sorted(scores, reverse=True)


def add_own_triples(G):
    """题5：在给定数据基础上，补充自己设计的 5-8 条三元组（示例）。

    学生应替换为自己的设计，并说明理由（为什么这样连、是否引入了新关系类型）。
    """
    extra = [
        ("李飞飞", "提出", "ImageNet"),
        ("ImageNet", "属于", "深度学习"),
        ("卷积神经网络", "属于", "深度学习"),
        ("AlphaGo", "基于", "深度强化学习"),
        ("深度强化学习", "属于", "机器学习"),
        ("ResNet", "属于", "卷积神经网络"),
    ]
    for h, r, t in extra:
        G.add_edge(h, t, relation=r)
    return G


def visualize(G, out_path="kg_graph.png"):
    def node_color(n):
        if n in {"人工智能", "机器学习", "深度学习", "连接主义", "符号主义"}:
            return "#FFD54F"
        if n in {"Transformer", "GPT", "AlphaGo", "感知机", "神经网络"}:
            return "#4FC3F7"
        if n in {"图灵", "McCarthy", "Hebb", "李世石"}:
            return "#81C784"
        if n in {"2017"}:
            return "#E0E0E0"
        return "#B39DDB"

    setup_chinese_font()
    plt.figure(figsize=(11, 8))
    pos = nx.spring_layout(G, seed=7, k=1.2, iterations=200)
    colors = [node_color(n) for n in G.nodes()]
    nx.draw_networkx_nodes(G, pos, node_color=colors, node_size=1500, alpha=0.92)
    nx.draw_networkx_labels(G, pos, font_size=9)
    nx.draw_networkx_edges(G, pos, arrowstyle="-|>", arrowsize=15,
                           edge_color="#999999", connectionstyle="arc3,rad=0.12")
    edge_labels = {(u, v): G[u][v]["relation"] for u, v in G.edges()}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=7)
    plt.axis("off")
    plt.title("人工智能 知识图谱（三元组可视化）")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"已保存可视化到 {out_path}")


def main():
    triples = load_triples(DATA_FILE)
    print(f"读取 {len(triples)} 条三元组")

    G = build_graph(triples)
    print(f"图：{G.number_of_nodes()} 个实体，{G.number_of_edges()} 条关系边")

    # 题2
    print("\n[题2] 与「Transformer」直接相关：", related_entities(G, "Transformer"))
    print("[题2] 与「人工智能」直接相关：", related_entities(G, "人工智能"))

    # 题3
    print("\n[题3] 路径查询：")
    for e1, e2 in [("GPT", "机器学习"), ("图灵", "人工智能"), ("图灵", "机器学习"),
                   ("AlphaGo", "李世石"), ("大语言模型", "人工智能")]:
        res = find_path(G, e1, e2)
        if res:
            nodes, rels = res
            print(f"  {e1} -> {e2}: {' -> '.join(nodes)}（关系：{'、'.join(rels)}）")
        else:
            print(f"  {e1} -> {e2}: 无有向路径")

    # 题4（进阶）
    print("\n[题4] 实体重要性（度中心性 top5）：")
    for score, n in entity_importance(G)[:5]:
        print(f"  {n:8s} 度数 = {score}")

    # 题1 可视化
    visualize(G)


if __name__ == "__main__":
    main()
