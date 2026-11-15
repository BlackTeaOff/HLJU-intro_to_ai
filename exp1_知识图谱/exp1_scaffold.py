#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""实验一 知识图谱构建 —— 学生脚手架

完成下面 3 处 TODO（对应 3 道必做题），即可运行出结果。
依赖：pip install networkx matplotlib
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
    """读取三元组文件，返回 [(head, relation, tail), ...]（教师已给，无需改动）。"""
    triples = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            h, r, t = line.split("\t")
            triples.append((h, r, t))
    return triples


def build_graph(triples):
    """TODO 1：用 NetworkX 构建「有向图」（约 3 行代码）

    要求：
    - 实体为节点，关系为有向边（方向 head -> tail）
    - 把关系名存到边的属性 'relation' 上，后续查询要用

    提示：G = nx.DiGraph()；G.add_edge(h, t, relation=r)
    """
    G = nx.DiGraph()
    # ============ 你的代码（替换下面两行）============
    for triple in triples:
        G.add_edge(triple[0], triple[2], relation=triple[1])
    # ==================================================
    return G


def related_entities(G, entity):
    """TODO 2：查询与 entity 直接相关（一跳）的所有实体

    要求：返回字典 {"out": [...], "in": [...]}
    - "out"：从 entity 出发的边所指向的实体（去重、排序）
    - "in" ：指向 entity 的边所来源的实体（去重、排序）

    提示：出边用 G.successors(entity)；入边用 G.predecessors(entity)
    """
    # ============ 你的代码 ============
    out = []
    inn = []
    # ... 补全 ...
    out = list(G.successors(entity))
    inn = list(G.predecessors(entity))
    # ================================
    return {"out": sorted(set(out)), "in": sorted(set(inn))}


def find_path(G, e1, e2):
    """TODO 3：判断 e1 到 e2 是否存在「有向路径」

    要求：
    - 若存在：返回 (节点序列, 关系序列)，例如 (["GPT","Transformer","深度学习"], ["基于","属于"])
    - 若不存在：返回 None

    提示：
    - nx.has_path(G, e1, e2) 判断是否有路径
    - nx.shortest_path(G, e1, e2) 得到节点序列
    - 关系序列从边属性取：G[u][v]["relation"]，遍历相邻节点对 (zip)
    """
    if not nx.has_path(G, e1, e2):
        return None
    # ============ 你的代码 ============
    nodes = []
    rels = []
    # ... 补全 ...
    nodes = nx.shortest_path(G, e1, e2)
    rels = [G[u][v]["relation"] for u, v in zip(nodes[:-1], nodes[1:])]
    # ================================
    return nodes, rels


def visualize(G):
    """可视化（教师已给完整示例，可直接运行；鼓励你自定义颜色/布局规则）。"""
    # 简单按节点名判断「类型」来着色（你可以换成自己的规则，作为加分点）
    def node_color(n):
        if n in {"人工智能", "机器学习", "深度学习", "连接主义", "符号主义"}:
            return "#FFD54F"   # 领域/范式 —— 黄
        if n in {"Transformer", "GPT", "AlphaGo", "感知机", "神经网络"}:
            return "#4FC3F7"   # 模型/技术 —— 蓝
        if n in {"图灵", "McCarthy", "Hebb", "李世石"}:
            return "#81C784"   # 人物 —— 绿
        if n in {"2017"}:
            return "#E0E0E0"   # 年份 —— 灰
        return "#B39DDB"       # 概念/其他 —— 紫

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
    plt.savefig("kg_graph.png", dpi=150, bbox_inches="tight")
    print("已保存可视化到 kg_graph.png")


def main():
    triples = load_triples(DATA_FILE)
    print(f"读取 {len(triples)} 条三元组")

    G = build_graph(triples)
    print(f"图：{G.number_of_nodes()} 个实体，{G.number_of_edges()} 条关系边")

    
    # 题2 验证
    print("\n[题2] 与「Transformer」直接相关的实体：",
          related_entities(G, "Transformer"))

    # 题3 验证
    for e1, e2 in [("GPT", "机器学习"), ("图灵", "机器学习"), ("AlphaGo", "李世石")]:
        res = find_path(G, e1, e2)
        if res:
            nodes, rels = res
            print(f"[题3] {e1} -> {e2}: {' -> '.join(nodes)}（关系：{'、'.join(rels)}）")
        else:
            print(f"[题3] {e1} -> {e2}: 无有向路径（体会「有向图的方向性」）")

    # 题1 可视化
    visualize(G)


if __name__ == "__main__":
    main()
