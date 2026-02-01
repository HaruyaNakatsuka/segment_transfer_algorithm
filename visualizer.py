import matplotlib
matplotlib.use("Agg")   # GUI(Tk)を使わない backend に固定

import matplotlib.pyplot as plt
import numpy as np
import os
import shutil
import logging

from model import CollaborativeState

logger = logging.getLogger(__name__)

plt.rcParams['font.family'] = 'MS Gothic'  # Windowsの場合
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.monospace'] = ['MS Gothic']


def plot_routes(
    all_LSP_state: CollaborativeState,
    phase: str,                      # "init" / "segment" / "ORTools" / "gat"
    output_file_index: int,                   # PNGファイル名用：呼び出し順の連番（衝突防止）
    round_number: int | None = None,
    output_dir: str = "figures",
    elapsed_time: float | None = None,
):
    """
    各車両の経路を描画し保存する関数
    """

    # =======================================================
    # 0) state から必要情報を取得
    # =======================================================
    customers = all_LSP_state.all_customers
    routes = all_LSP_state.get_all_routes()
    depot_id_list = all_LSP_state.depot_id_list
    vehicle_num_list = all_LSP_state.vehicle_num_list
    instance_name = all_LSP_state.instance_name

    id_to_coord = {int(c["id"]): (float(c["x"]), float(c["y"])) for c in customers}
    colors = ["tab:blue", "tab:green", "tab:red", "tab:orange", "tab:purple", "tab:brown"]

    # =======================================================
    # 1) 内部ユーティリティ
    # =======================================================
    def pct(old: float | None, new: float | None) -> str:
        """改善を (new-old)/old * 100 で表示（短縮がマイナス）。"""
        if old is None or old <= 0 or new is None:
            return "—"
        return f"{(new - old) / old * 100:.2f}%"

    # =======================================================
    # 2) 出力フォルダ準備（plot_seq == 0 のときだけ掃除）
    # =======================================================
    instance_folder = os.path.join(output_dir, instance_name)
    if output_file_index == 0 and os.path.isdir(instance_folder):
        shutil.rmtree(instance_folder)
    os.makedirs(instance_folder, exist_ok=True)

    # =======================================================
    # 3) 図のセットアップ
    # =======================================================
    fig = plt.figure(figsize=(8, 8))

    if phase == "init":
        plt.title("初期解")
    elif phase == "segment":
        plt.title(f"ラウンド{round_number}：セグメント移管による全体最適化")
    elif phase == "ORTools":
        plt.title(f"ラウンド{round_number}：ORToolsによる個別最適化")
    elif phase == "gat":
        plt.title(f"ラウンド{round_number}：社内GATによる個別最適化")

    # =======================================================
    # 4) 経路描画
    # =======================================================
    vehicle_index = 0
    for lsp_index, num_vehicles in enumerate(vehicle_num_list):
        color = colors[lsp_index % len(colors)]
        depot_id = int(depot_id_list[lsp_index])
        depot_x, depot_y = id_to_coord[depot_id]

        plt.scatter(
            depot_x, depot_y,
            marker="s", c=color, s=120,
            edgecolor="black",
            label=f"LSP {lsp_index+1}"
        )

        for _ in range(num_vehicles):
            route = routes[vehicle_index]
            vehicle_index += 1
            if len(route) <= 2:
                continue
            xs = [id_to_coord[int(i)][0] for i in route]
            ys = [id_to_coord[int(i)][1] for i in route]
            plt.plot(xs, ys, color=color, alpha=0.8)
            plt.scatter(xs, ys, c=color, s=15)

    # =======================================================
    # 5) ボロノイ領域分割線の描画
    # =======================================================
    if len(depot_id_list) > 1:
        depot_coords = np.array([id_to_coord[int(d)] for d in depot_id_list])

        x_min = min(float(c["x"]) for c in customers) - 10
        x_max = max(float(c["x"]) for c in customers) + 10
        y_min = min(float(c["y"]) for c in customers) - 10
        y_max = max(float(c["y"]) for c in customers) + 10

        x_vals = np.linspace(x_min, x_max, 300)
        y_vals = np.linspace(y_min, y_max, 300)
        X, Y = np.meshgrid(x_vals, y_vals)

        distances = np.zeros((len(depot_coords), *X.shape), dtype=float)
        for k, (dx, dy) in enumerate(depot_coords):
            distances[k] = np.sqrt((X - dx) ** 2 + (Y - dy) ** 2)

        if len(depot_id_list) == 2:
            # 2社：等距離線
            plt.contour(
                X, Y, distances[0] - distances[1],
                levels=[0], colors="gray", linestyles="--", linewidths=1
            )
        else:
            # 3社以上：ボロノイ境界線
            region = np.argmin(distances, axis=0)  # (H, W) int

            boundary = np.zeros_like(region, dtype=bool)
            boundary[:-1, :] |= (region[:-1, :] != region[1:, :])   # 縦方向
            boundary[:, :-1] |= (region[:, :-1] != region[:, 1:])   # 横方向

            plt.contour(
                X, Y, boundary.astype(int),
                levels=[0.5], colors="gray", linestyles="--", linewidths=1
            )


    # =======================================================
    # 6) フッター
    # =======================================================
    curr_company = [c.current_total_route_length for c in all_LSP_state.companies]
    curr_total = all_LSP_state.current_total_route_length

    init_company = [c.initial_total_route_length for c in all_LSP_state.companies]
    init_total = all_LSP_state.initial_total_route_length

    # ラウンド比（前回比）は state に previous_* があれば使う（無ければ —）
    prev_company = [getattr(c, "previous_total_route_length", None) for c in all_LSP_state.companies]
    prev_total = getattr(all_LSP_state, "previous_total_route_length", None)

    lines = []
    if phase == "init":
        lines.append("【初期解】")
        for i, c in enumerate(curr_company, 1):
            lines.append(f"  LSP {i}: {c}")
        lines.append(f"  TOTAL: {curr_total}")
    else:
        head = "【改善状況】"

        lines.append(head)
        for i, c in enumerate(curr_company, 1):
            base_prev = prev_company[i - 1] if prev_company else None
            base_init = init_company[i - 1] if init_company else None
            lines.append(
                f"    LSP {i}: {c}   改善(直前比): {pct(base_prev, c)}   改善(初期比): {pct(base_init, c)}"
            )

        lines.append(
            f"    TOTAL: {curr_total}   改善(直前比): {pct(prev_total, curr_total)}   改善(初期比): {pct(init_total, curr_total)}"
        )

    if elapsed_time is not None:
        lines.append(f"\n暫定時間: {elapsed_time:.2f} 秒")

    if lines:
        text = "\n".join(lines)
        plt.subplots_adjust(bottom=0.17)
        fig.text(
            0.02, 0.02, text,
            ha="left", va="bottom",
            fontsize=10,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.9, edgecolor="#999"),
            family="monospace",
        )

    # =======================================================
    # 7) 保存
    # =======================================================
    save_path = os.path.join(instance_folder, f"routes_iter_{output_file_index:04d}.png")
    plt.legend()
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid(False)
    fig.savefig(save_path)
    plt.close(fig)
    logger.info(f"✅図を保存しました: {save_path}")






# === フッター・タイトル非表示用（修論付録用） ===
def _plot_routes(
    all_LSP_state: CollaborativeState,
    phase: str,                      # "init" / "segment" / "ORTools" / "gat"
    output_file_index: int,          # PNGファイル名用：呼び出し順の連番（衝突防止）
    round_number: int | None = None,
    output_dir: str = "figures",
    elapsed_time: float | None = None,
):
    """
    各車両の経路を描画し保存する関数

    仕様変更：
      - タイトル非表示
      - フッター非表示
      - グラフ内グリッド線表示（plt.grid(True)）
    """

    # =======================================================
    # 0) state から必要情報を取得
    # =======================================================
    customers = all_LSP_state.all_customers
    routes = all_LSP_state.get_all_routes()
    depot_id_list = all_LSP_state.depot_id_list
    vehicle_num_list = all_LSP_state.vehicle_num_list
    instance_name = all_LSP_state.instance_name

    id_to_coord = {int(c["id"]): (float(c["x"]), float(c["y"])) for c in customers}
    colors = ["tab:blue", "tab:green", "tab:red", "tab:orange", "tab:purple", "tab:brown"]

    # =======================================================
    # 1) 出力フォルダ準備
    # =======================================================
    instance_folder = os.path.join(output_dir, instance_name)
    if output_file_index == 0 and os.path.isdir(instance_folder):
        shutil.rmtree(instance_folder)
    os.makedirs(instance_folder, exist_ok=True)

    # =======================================================
    # 2) 図のセットアップ
    # =======================================================
    fig = plt.figure(figsize=(8, 8))

    # =======================================================
    # 3) 経路描画
    # =======================================================
    vehicle_index = 0
    for lsp_index, num_vehicles in enumerate(vehicle_num_list):
        color = colors[lsp_index % len(colors)]
        depot_id = int(depot_id_list[lsp_index])
        depot_x, depot_y = id_to_coord[depot_id]

        plt.scatter(
            depot_x, depot_y,
            marker="s", c=color, s=120,
            edgecolor="black",
            label=f"LSP {lsp_index+1}"
        )

        for _ in range(num_vehicles):
            route = routes[vehicle_index]
            vehicle_index += 1
            if len(route) <= 2:
                continue
            xs = [id_to_coord[int(i)][0] for i in route]
            ys = [id_to_coord[int(i)][1] for i in route]
            plt.plot(xs, ys, color=color, alpha=0.8)
            plt.scatter(xs, ys, c=color, s=15)

    # =======================================================
    # 4) ボロノイ領域分割線の描画
    # =======================================================
    if len(depot_id_list) > 1:
        depot_coords = np.array([id_to_coord[int(d)] for d in depot_id_list])

        x_min = min(float(c["x"]) for c in customers) - 10
        x_max = max(float(c["x"]) for c in customers) + 10
        y_min = min(float(c["y"]) for c in customers) - 10
        y_max = max(float(c["y"]) for c in customers) + 10

        x_vals = np.linspace(x_min, x_max, 300)
        y_vals = np.linspace(y_min, y_max, 300)
        X, Y = np.meshgrid(x_vals, y_vals)

        distances = np.zeros((len(depot_coords), *X.shape), dtype=float)
        for k, (dx, dy) in enumerate(depot_coords):
            distances[k] = np.sqrt((X - dx) ** 2 + (Y - dy) ** 2)

        if len(depot_id_list) == 2:
            # 2社：等距離線
            plt.contour(
                X, Y, distances[0] - distances[1],
                levels=[0], colors="gray", linestyles="--", linewidths=1
            )
        else:
            # 3社以上：ボロノイ境界（最近傍が変わるところ）
            region = np.argmin(distances, axis=0)  # (H, W) int

            boundary = np.zeros_like(region, dtype=bool)
            boundary[:-1, :] |= (region[:-1, :] != region[1:, :])   # 縦方向
            boundary[:, :-1] |= (region[:, :-1] != region[:, 1:])   # 横方向

            plt.contour(
                X, Y, boundary.astype(int),
                levels=[0.5], colors="gray", linestyles="--", linewidths=1
            )

    # =======================================================
    # 5) 保存
    # =======================================================
    save_path = os.path.join(instance_folder, f"routes_iter_{output_file_index:04d}.png")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.grid(True)
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    logger.info(f"✅図を保存しました: {save_path}")

