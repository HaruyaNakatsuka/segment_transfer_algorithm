from __future__ import annotations

import os
import sys
import time
import random
import unicodedata
from typing import List, Tuple

from model import CompanyState, CollaborativeState
from parser import parse_lilim200
import proposed_method
from visualizer import plot_routes
from web_exporter import export_vrp_state, generate_index_json
import utils


# ======================== 実行モデル・環境設定（フラグ）一覧 ==================================
# --- ビジュアライザ有効化 ---
ENABLE_PLOT = True  # PNG出力(plot_routes)
ENABLE_EXPORT = True  # JSON出力(export_vrp_state)


# --- optimize_by_partial_route_reassignment 関数の設定 ---
ENABLE_STOCHASTIC_IMP: bool = False  # 移管による改善量を確率変数化する/しない
STOCH_NET_IMP_STDDEV: float = 30.0  # 正規分布の標準偏差
STOCH_RANDOM_SEED: int = 0  # 乱数のシード値
FEASIBILITY_CHECK = False  # feasibility チェック有効/無効

# --- 確率動作ONの時、タスク割り当て変更を最大何回許可するか ---
NUM_REASSIGNMENT_LIMIT = 10

# --- optimize_intra_company_by_exact_2vehicle_gat 関数の設定 ---
OPTIMIZE_BY_GAT = False  # Trueなら社内最適化にGATが使用される
DEBUG_2GAT = False
# ===========================================================================================


# ================================ テストケースの定義部 =======================================
test_cases = [
    (["data/LC1_2_2.txt", "data/LC1_2_6.txt"], [(0, 0), (42, -42)], 0),
]

"""
# --- 規模の小さいVRPに対し厳密解ソルバーを使用して実験する場合の入力 ---
test_cases = [
    (["data/LC1_2_2.txt", "data/LC1_2_6.txt"], [(0, 0), (42, -42)], sys.maxsize),
    (["data/LC1_2_2.txt", "data/LC1_2_7.txt"], [(0, 0), (-32, -32)], sys.maxsize),
    (["data/LC1_2_4.txt", "data/LC1_2_7.txt"], [(0, 0), (-30, 0)], 9),
    (["data/LC1_2_4.txt", "data/LC1_2_8.txt"], [(0, 0), (-30, 0)], 9),
    (["data/LC1_2_10.txt", "data/LC1_2_4.txt"], [(0, 0), (30, 0)], 8),
    (["data/LR1_2_3.txt", "data/LR1_2_8.txt"], [(0, 0), (0, 30)], 7),
    (["data/LR1_2_5.txt", "data/LR1_2_8.txt"], [(0, 0), (0, 30)], 7),
    (["data/LR1_2_8.txt", "data/LR1_2_9.txt"], [(0, 0), (0, -30)], 7),
    (["data/LR1_2_10.txt", "data/LR1_2_3.txt"], [(0, 0), (0, -30)], 7),
    (["data/LR1_2_10.txt", "data/LR1_2_8.txt"], [(0, 0), (0, 30)], 5)
]

# --- ORToolsをのみを使用して実験する場合の入力 ---
test_cases = [
    (["data/LC1_2_2.txt", "data/LC1_2_6.txt"], [(0, 0), (42, -42)], 0),
    (["data/LC1_2_2.txt", "data/LC1_2_7.txt"], [(0, 0), (-32, -32)], 0),
    (["data/LC1_2_4.txt", "data/LC1_2_7.txt"], [(0, 0), (-30, 0)], 0),
    (["data/LC1_2_4.txt", "data/LC1_2_8.txt"], [(0, 0), (-30, 0)], 0),
    (["data/LC1_2_10.txt", "data/LC1_2_4.txt"], [(0, 0), (30, 0)], 0),
    (["data/LR1_2_3.txt", "data/LR1_2_8.txt"], [(0, 0), (0, 30)], 0),
    (["data/LR1_2_5.txt", "data/LR1_2_8.txt"], [(0, 0), (0, 30)], 0),
    (["data/LR1_2_8.txt", "data/LR1_2_9.txt"], [(0, 0), (0, -30)], 0),
    (["data/LR1_2_10.txt", "data/LR1_2_3.txt"], [(0, 0), (0, -30)], 0),
    (["data/LR1_2_10.txt", "data/LR1_2_8.txt"], [(0, 0), (0, 30)], 0)
]

# --- 4社参加のテストケース ---
test_cases = [
    # --- Cluster type (LC) ---
    (["data/LC1_2_1.txt", "data/LC1_2_2.txt", "data/LC1_2_3.txt", "data/LC1_2_4.txt"],
     [(0, 0), (9, 35), (34, 4), (40, 33)], 0),

    (["data/LC1_2_5.txt", "data/LC1_2_6.txt", "data/LC1_2_7.txt", "data/LC1_2_8.txt"],
     [(0, 0), (2, 46), (45, 10), (48, 41)], 0),

    (["data/LC1_2_1.txt", "data/LC1_2_2.txt", "data/LC1_2_5.txt", "data/LC1_2_6.txt"],
     [(0, 0), (6, 38), (35, 7), (39, 39)], 0),

    (["data/LC1_2_3.txt", "data/LC1_2_4.txt", "data/LC1_2_7.txt", "data/LC1_2_8.txt"],
     [(0, 0), (1, 37), (41, 2), (35, 42)], 0),
     
    # --- Random type (LR) ---
    (["data/LR1_2_1.txt", "data/LR1_2_2.txt", "data/LR1_2_3.txt", "data/LR1_2_4.txt"],
     [(0, 0), (7, 36), (35, 9), (44, 38)], 0),

    (["data/LR1_2_5.txt", "data/LR1_2_6.txt", "data/LR1_2_7.txt", "data/LR1_2_8.txt"],
     [(0, 0), (3, 49), (48, 0), (45, 44)], 0),

    (["data/LR1_2_1.txt", "data/LR1_2_2.txt", "data/LR1_2_5.txt", "data/LR1_2_6.txt"],
     [(0, 0), (0, 39), (41, 3), (33, 38)], 0),

    (["data/LR1_2_3.txt", "data/LR1_2_4.txt", "data/LR1_2_7.txt", "data/LR1_2_8.txt"],
     [(0, 0), (4, 38), (45, 6), (37, 46)], 0),
]
"""
# ===========================================================================================


def print_testcase_title(case_index: int, file_paths: List[str], offsets: List[Tuple[float, float]], exact_pd_pair_limit: int) -> str:
    """
    テストケース情報を見やすく表示し、instance_name を返す。

    要件:
      - file_paths / offsets が空、または長さ不一致なら日本語エラーで終了
      - exact_pd_pair_limit が未指定(None)なら日本語エラーで終了
      - 3本以上の入力にも対応
      - コロン位置を揃える / 罫線長を自動調整
      - exact_pd_pair_limit を強調表示（sys.maxsize は特別文言）
    """
    if not file_paths:
        print("エラー: テストケースが1つも指定されていません。", file=sys.stderr)
        sys.exit(1)
    if not offsets:
        print("エラー: オフセットが1つも指定されていません。", file=sys.stderr)
        sys.exit(1)
    if len(file_paths) != len(offsets):
        print(
            f"エラー: 入力ファイル数とオフセット数が一致していません。"
            f" (files={len(file_paths)}, offsets={len(offsets)})",
            file=sys.stderr,
        )
        sys.exit(1)
    if exact_pd_pair_limit is None:
        print("エラー: exact_pd_pair_limit が指定されていません。", file=sys.stderr)
        sys.exit(1)

    stems = [os.path.splitext(os.path.basename(p))[0] for p in file_paths]
    instance_name = "_".join(stems)
    instance_display = " + ".join(stems)

    offsets_display = " , ".join([f"({ox}, {oy})" for (ox, oy) in offsets])

    if exact_pd_pair_limit == sys.maxsize:
        gat_option = "すべての2車両VRPを自作VRPソルバーで厳密解を求解する"
    else:
        gat_option = f"PDペア数【{exact_pd_pair_limit}】ペア以下は自作VRPソルバーで厳密解を求解"

    labels = ["テストケース", "インスタンス", "オフセット", "GATオプション"]

    # 表示幅（日本語全角を考慮）
    label_width = 0
    for s in labels:
        w = 0
        for ch in s:
            w += 2 if unicodedata.east_asian_width(ch) in ("W", "F", "A") else 1
        label_width = max(label_width, w)

    def _pad_label(s: str) -> str:
        # ここだけは「内部関数」でなく、同関数内のローカル処理として完結させる
        w = 0
        for ch in s:
            w += 2 if unicodedata.east_asian_width(ch) in ("W", "F", "A") else 1
        return s + (" " * (label_width - w))

    lines = [
        f"{_pad_label('テストケース')}: {case_index}",
        f"{_pad_label('インスタンス')}: {instance_display}",
        f"{_pad_label('オフセット')}: {offsets_display}",
        f"{_pad_label('GATオプション')}: {gat_option}",
    ]

    # 罫線長を各行の表示幅に合わせる（日本語幅考慮）
    max_line_w = 0
    for s in lines:
        w = 0
        for ch in s:
            w += 2 if unicodedata.east_asian_width(ch) in ("W", "F", "A") else 1
        max_line_w = max(max_line_w, w)

    bar = "=" * max_line_w

    print("\n" + bar)
    for s in lines:
        print(s)
    print(bar)

    return instance_name


def build_state_for_case(file_paths: List[str], offsets: List[Tuple[float, float]], case_index: int, instance_name: str,) -> CollaborativeState:
    """
    2社の入力ファイルを読み込み、IDオフセットしつつ結合した CollaborativeState を構築する。
    """
    num_lsps = len(file_paths)
    assert num_lsps == len(offsets), "file_paths と offsets の長さが一致していません。"

    all_customers = []
    all_PD_pairs = {}
    companies: List[CompanyState] = []
    vehicle_capacity = None

    id_offset = 0
    for idx, (path, offset) in enumerate(zip(file_paths, offsets)):
        start_id = id_offset

        data = parse_lilim200(path, x_offset=offset[0], y_offset=offset[1], id_offset=id_offset)

        all_customers.extend(data["customers"])
        all_PD_pairs.update(data["PD_pairs"])

        max_id = max(c["id"] for c in data["customers"])
        next_offset = max_id + 1

        comp = CompanyState(
            idx=idx,
            depot_id=data["depot_id"],
            depot_coord=data["depot_coord"],
            num_vehicles=data["num_vehicles"],
            id_min=start_id,
            id_max=next_offset,
            routes=[],
            initial_total_route_length=0,
            current_total_route_length=0,
        )
        companies.append(comp)

        id_offset = next_offset

        if vehicle_capacity is None:
            vehicle_capacity = data["vehicle_capacity"]

    assert vehicle_capacity is not None, "vehicle_capacity が取得できませんでした。"

    return CollaborativeState(
        all_customers=all_customers,
        all_PD_pairs=all_PD_pairs,
        vehicle_capacity=vehicle_capacity,
        companies=companies,
        instance_name=instance_name,
        case_index=case_index,
        initial_total_route_length=0,
        current_total_route_length=0,
    )







def run_case(all_LSP_state: CollaborativeState, start_time: float, exact_pd_pair_limit: int) -> None:
    # ---- 視覚化ツール連番カウンタ ----
    plot_sequence = 0
    def save_figure(phase: str, round_number: int | None, elapsed_time: float | None = None) -> None:
        nonlocal plot_sequence
        if not ENABLE_PLOT:
            return
        plot_routes(
            all_LSP_state=all_LSP_state,
            phase=phase,                # "init" / "segment" / "ORTools" / "gat"
            output_file_index=plot_sequence,          # ファイル名用の通し番号
            round_number=round_number,      # 図中表示用（例：outer_iterやgat_iter）
            elapsed_time=elapsed_time,
        )
        plot_sequence += 1
    
    json_sequence = 0
    def save_json_data() -> None:
        nonlocal json_sequence
        if not ENABLE_EXPORT:
            return
        export_vrp_state(
            all_LSP_state.all_customers,
            all_LSP_state.get_all_routes(),
            all_LSP_state.all_PD_pairs,
            json_sequence,  # ← 単調増加の通し番号
            case_index=all_LSP_state.case_index,
            depot_id_list=all_LSP_state.depot_id_list,
            vehicle_num_list=all_LSP_state.vehicle_num_list,
            instance_name=all_LSP_state.instance_name,
            output_root="web_data",
        )
        json_sequence += 1
    
    
    
    
    # =================================
    # Step0：乱数生成
    # =================================
    if ENABLE_STOCHASTIC_IMP:
        utils.validate_stochastic_stddev(STOCH_NET_IMP_STDDEV)
    # ***ENABLE_STOCHASTIC_IMP=Falseでもデフォルトで生成されるが、実際には使用されない***
    stoch_rng = random.Random(STOCH_RANDOM_SEED) 
    
    
    
    
    # =================================
    # Step1：初期解生成
    # =================================
    proposed_method.initial_routes_generator(all_LSP_state)

    print("\n>>> 初期解が生成されました")
    for idx, comp in enumerate(all_LSP_state.companies):
        print(f"LSP {idx}: {comp.initial_total_route_length}")
    print(f"TOTAL: {all_LSP_state.initial_total_route_length}")

    # --- [データ保存] -> jsonファイル、pngファイル ---
    save_json_data()
    save_figure(phase="init", round_number=None)



    outer_iteration_index = 0
    while True:
        outer_iteration_index += 1
        print(f"\n==================== ループ {outer_iteration_index}回目 ====================")

        # ==========================================
        # STEP2：セグメント移管による全体最適化
        # ==========================================
        all_LSP_state = proposed_method.optimize_collectively_by_segment_transfar(
            all_LSP_state,
            enable_stochastic_imp=ENABLE_STOCHASTIC_IMP,
            stoch_net_imp_stddev=STOCH_NET_IMP_STDDEV,
            rng=stoch_rng,
            check_feasibility=FEASIBILITY_CHECK,
        )
        utils.print_cost_table(all_LSP_state, title=f">>> ラウンド{outer_iteration_index}：セグメント移管による全体改善が終了しました")


        # === 終了判定 ===
        if ENABLE_STOCHASTIC_IMP:
            # --- 条件1：反復上限 ---
            if outer_iteration_index >= NUM_REASSIGNMENT_LIMIT:
                print(
                    f"\n>>> 反復上限に到達したため終了します。"
                    f" (outer_iter={outer_iteration_index}, limit={NUM_REASSIGNMENT_LIMIT})"
                )
                break

            # --- 条件2：個別合理性破綻 ---
            violated = []
            for i, comp in enumerate(all_LSP_state.companies):
                if comp.current_total_route_length > comp.initial_total_route_length:
                    violated.append((i, comp.current_total_route_length, comp.initial_total_route_length))

            if violated:
                print("\n>>> 個別合理性が破綻したため終了します。")
                for i, aft, init_ in violated:
                    print(f"  - LSP{i+1}: after={aft} > initial={init_}")
                break

        else:
            # --- 各社の経路長に変化なし（previous と current が一致） ---
            if all(
                int(comp.current_total_route_length) == int(comp.previous_total_route_length)
                for comp in all_LSP_state.companies
            ):
                break

        save_figure(phase="segment", round_number=outer_iteration_index)
        save_json_data()
        
        
        # ==========================================
        # STEP3：社内最適化
        # ==========================================
        if not OPTIMIZE_BY_GAT:
            # --- ORToolsによる社内一括最適化 ---
            all_LSP_state=proposed_method.optimize_individually_by_ORTools(all_LSP_state, FEASIBILITY_CHECK)
            utils.print_cost_table(all_LSP_state, title=f">>> ラウンド{outer_iteration_index}：社内最適化（ORTools）が終了しました")
            save_figure(phase="ORTools", round_number=outer_iteration_index)
            save_json_data()
        else:
            # --- GATアルゴリズムによる社内最適化 ---
            print("")
            print(">>> 社内GATによる最適化")

            GAT_iteration_index = 0
            while all_LSP_state.current_total_route_length != all_LSP_state.previous_total_route_length:
                GAT_iteration_index += 1
                
                all_LSP_state = proposed_method.optimize_individually_by_GAT(
                    all_LSP_state,
                    exact_pd_pair_limit=exact_pd_pair_limit,
                    debug_2gat=DEBUG_2GAT
                )
                
                elapsed = time.time() - start_time
                utils.print_cost_table(all_LSP_state, title=f">>> ラウンド{outer_iteration_index}：社内GAT最適化{GAT_iteration_index}回目が終了しました")
                save_figure(phase="gat", round_number=GAT_iteration_index, elapsed_time=elapsed)
                save_json_data()

    return






def main() -> None:
    utils.setup_logging(show_progress=False)

    for case_index, (file_paths, offsets, exact_pd_pair_limit) in enumerate(test_cases, 1):
        # 各インスタンスに対して実行時間を計測
        start_time = time.time()
        # コンソールにタイトル表示
        instance_name = print_testcase_title(case_index, file_paths, offsets, exact_pd_pair_limit)
        # CollaborativeState 作成
        state = build_state_for_case(file_paths, offsets, case_index, instance_name)
        
        
        # === アルゴリズム本体 ===
        run_case(state, start_time, exact_pd_pair_limit)
        
        # 実行時間計測
        elapsed = time.time() - start_time
        # 最終経路の実行可能性チェック
        ok = utils.check_solution_feasibility(state, verbose=True)
        if not ok:
            print(f"⚠️ テストケース {state.case_index}：最終解が実行不可能です（詳細は上を参照）")
        # 実行時間表示
        print(f">>> テストケース {state.case_index} の実行時間: {elapsed:.2f} 秒")
        

if __name__ == "__main__":
    main()
