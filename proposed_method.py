from __future__ import annotations

import unicodedata
from typing import Dict, List, Tuple, Set, Optional
import math
import random

from model import CollaborativeState
from ortools_vrp_solver import solve_vrp_flexible
from exact_pdptw_solver_version1 import solve_exact_2vehicle_vrp
import utils
from ortools.sat.python import cp_model



def initial_routes_generator(all_LSP_state: CollaborativeState):
    """
    会社（LSP）単位で(初期)経路を一括生成し、CompanyState を更新して返す。

    ・initialize_individual_vrps の内部で CompanyState から必要な情報を適宜取得する
    ・initialize_individual_vrps の中で（１）の初期経路長のみ埋める
    ・返り値は更新された CompanyState
    """
    for company in all_LSP_state.companies:
        depot_id = company.depot_id
        num_vehicles = company.num_vehicles

        # sub_customersの抽出（会社のID範囲。デポを含む）
        sub_customers = [c for c in all_LSP_state.all_customers if company.id_min <= c['id'] < company.id_max]

        # sub_PD_pairsの抽出
        sub_customer_ids = {c['id'] for c in sub_customers}
        sub_PD_pairs = [
            (pickup, delivery)
            for pickup, delivery in all_LSP_state.all_PD_pairs.items()
            if pickup in sub_customer_ids or delivery in sub_customer_ids
        ]

        # 各車両の出発／終了デポ設定
        start_depot = [depot_id] * num_vehicles
        end_depot = [depot_id] * num_vehicles

        initial_routes = None
        # VRPを解く
        lsp_routes = solve_vrp_flexible(
            sub_customers,
            initial_routes,
            sub_PD_pairs,
            num_vehicles=num_vehicles,
            vehicle_capacity=all_LSP_state.vehicle_capacity,
            start_depots=start_depot,
            end_depots=end_depot,
            use_capacity=True,
            use_time=True,
            use_pickup_delivery=True,
            Warm_Start=False
        )

        company.routes = lsp_routes

        # --- 初期解の総経路長を記録 ---
        company.initial_total_route_length = int(sum(utils.route_cost(r, all_LSP_state.all_customers) for r in lsp_routes))
        company.previous_total_route_length =company.initial_total_route_length
        company.current_total_route_length = company.initial_total_route_length

    # --- 全社合計の経路長を記録 ---
    all_LSP_state.initial_total_route_length = int(sum(c.initial_total_route_length for c in all_LSP_state.companies))
    all_LSP_state.previous_total_route_length = all_LSP_state.initial_total_route_length
    all_LSP_state.current_total_route_length = all_LSP_state.initial_total_route_length
    
    
    return company



def optimize_collectively_by_segment_transfar(
    all_LSP_state: CollaborativeState,
    *,
    enable_stochastic_imp: bool = False,
    stoch_net_imp_stddev: float = 10.0,
    rng: Optional[random.Random] = None,
    check_feasibility: bool = False,
) -> CollaborativeState:


    # ============================
    # 出力ユーティリティ
    # ============================
    def _p(msg: str = "", indent: int = 0) -> None:
        if msg == "":
            print(">")
            return
        print("> " + ("  " * indent) + msg)

    def _vis_w(s: str) -> int:
        w = 0
        for ch in s:
            w += 2 if unicodedata.east_asian_width(ch) in ("F", "W", "A") else 1
        return w

    def _rule(char: str = "─", width: int = 56) -> None:
        _p(char * width)

    def _boxed_title(title: str, width: int = 56) -> None:
        w = max(width, _vis_w(title))
        _rule("─", w)
        _p(title)
        _rule("─", w)

    def _phase(title: str) -> None:
        _p(f"=== {title} ===")

    def _subsection(title: str) -> None:
        _p()
        _p(title)

    def _mode(warm: bool) -> str:
        return "ウォームスタート" if warm else "コールドスタート"
    
    def _print_cost_summary(title: str, prev_per: List[int], cur_per: List[int], prev_total: int, cur_total: int, ) -> None:
        _subsection(f"■ {title}：経路長の変化")

        def _rate(cur: int, base: int) -> float:
            return float("nan") if base <= 0 else (cur - base) / base

        def _ljust_vis(s: str, width: int) -> str:
            return s + " " * max(0, width - _vis_w(s))

        def _rjust_vis(s: str, width: int) -> str:
            return " " * max(0, width - _vis_w(s)) + s

        init_list = [int(getattr(comp, "initial_total_route_length", 0)) for comp in companies]
        prev_list = [int(x) for x in prev_per]
        cur_list = [int(x) for x in cur_per]
        init_total = int(getattr(all_LSP_state, "initial_total_route_length", 0))

        nums_for_width = init_list + prev_list + cur_list + [init_total, int(prev_total), int(cur_total)]
        w_num = max(4, max(len(str(n)) for n in nums_for_width))
        w_rate = 10
        name_candidates = [f"LSP{len(companies)}", "TOTAL"]
        w_name = max(_vis_w(s) for s in name_candidates)

        header = (
            _ljust_vis("", w_name)
            + "  "
            + _rjust_vis("初期", w_num)
            + "  "
            + _rjust_vis("直前", w_num)
            + "  "
            + _rjust_vis("現在", w_num)
            + "  "
            + _rjust_vis("直前比", w_rate)
            + "  "
            + _rjust_vis("初期比", w_rate)
        )
        _p(header, indent=1)

        total_width = _vis_w(header)
        _p("-" * total_width, indent=1)

        for i in range(n_comp):
            name = f"LSP{i+1}"
            init = init_list[i]
            prev = prev_list[i]
            cur = cur_list[i]
            r_prev = _rate(cur, prev)
            r_init = _rate(cur, init)

            s_prev = f"{r_prev:+.2%}"
            s_init = f"{r_init:+.2%}"

            line = (
                _ljust_vis(name, w_name)
                + "  "
                + _rjust_vis(str(init), w_num)
                + "  "
                + _rjust_vis(str(prev), w_num)
                + "  "
                + _rjust_vis(str(cur), w_num)
                + "  "
                + _rjust_vis(s_prev, w_rate)
                + "  "
                + _rjust_vis(s_init, w_rate)
            )
            _p(line, indent=1)

        _p("-" * total_width, indent=1)

        r_prev_t = _rate(cur_total, int(prev_total))
        r_init_t = _rate(cur_total, init_total)
        s_prev_t = f"{r_prev_t:+.2%}"
        s_init_t = f"{r_init_t:+.2%}"

        total_line = (
            _ljust_vis("TOTAL", w_name)
            + "  "
            + _rjust_vis(str(init_total), w_num)
            + "  "
            + _rjust_vis(str(int(prev_total)), w_num)
            + "  "
            + _rjust_vis(str(int(cur_total)), w_num)
            + "  "
            + _rjust_vis(s_prev_t, w_rate)
            + "  "
            + _rjust_vis(s_init_t, w_rate)
        )
        _p(total_line, indent=1)
        _p()
        _p()

    # =========================================================
    # Feasibility 検証機（check_feasibility=True のときのみ動作）
    # =========================================================
    def _run_feasibility_check_on_routes(label: str, routes_per_company: List[List[List[int]]]) -> None:
        """
        各社・各車両のルートを utils.check_single_route_feasible で個別に検査する。
        何らかの違反が1つでも見つかったら、その場で詳細を出力して例外で停止する。
        """
        _subsection(f"■ Feasibility check（{label}）")

        # 前提：routes_per_company は companies と整合（会社数・車両数が一致）していること
        for comp_idx, comp in enumerate(companies):
            depot_id = int(comp.depot_id)
            num_v = int(comp.num_vehicles)

            if comp_idx >= len(routes_per_company):
                _p(f"ERROR: routes_per_company に会社{comp_idx}の情報がありません", indent=1)
                raise RuntimeError(f"feasibility_failed: missing_company_routes comp_idx={comp_idx} label={label}")

            if len(routes_per_company[comp_idx]) != num_v:
                _p(
                    f"ERROR: 会社{comp_idx}のルート本数が車両台数と一致しません "
                    f"(routes={len(routes_per_company[comp_idx])}, vehicles={num_v})",
                    indent=1,
                )
                raise RuntimeError(
                    f"feasibility_failed: route_count_mismatch comp_idx={comp_idx} "
                    f"routes={len(routes_per_company[comp_idx])} vehicles={num_v} label={label}"
                )

            for v_idx, route in enumerate(routes_per_company[comp_idx]):
                ok, reason, detail = utils.check_single_route_feasible(
                    route,
                    customers=customers,
                    depot_id=depot_id,
                    all_PD_pairs=PD_map,
                    vehicle_capacity=vehicle_capacity,
                )
                if not ok:
                    _p(f"ERROR: INFEASIBLE at {label}", indent=1)
                    _p(f"LSP{comp_idx+1} vehicle{v_idx} reason={reason}", indent=1)
                    if detail is not None:
                        _p(f"detail={detail}", indent=2)
                    _p(f"route={route}", indent=2)
                    raise RuntimeError(
                        f"feasibility_failed: {label} LSP{comp_idx+1} vehicle{v_idx} reason={reason}"
                    )

        _p("判定：FEASIBLE", indent=1)



















    # ============================
    # 開始メッセージ
    # ============================
    _boxed_title("セグメント移管処理開始")
    
    # 開始直後の feasibility チェック
    if check_feasibility:
        _run_feasibility_check_on_routes("開始直後（入力ルート）", [list(comp.routes) for comp in companies])

    # ============================
    # データ準備（本関数で使う派生データを一括構築）
    # ============================
    customers = all_LSP_state.all_customers          # 顧客ノード（depot含む）の辞書列
    PD_map = all_LSP_state.all_PD_pairs              # pickup -> delivery の対応（Mapping）
    vehicle_capacity = int(all_LSP_state.vehicle_capacity)  # 車両容量（全社共通想定）
    companies = all_LSP_state.companies              # CompanyState の配列（会社順が重要）
    n_comp = len(companies)                          # 会社数

    id_to_coord = utils.build_id_to_coord(customers) # node_id -> (x,y) の座標辞書（距離計算用）
    depot_ids = [int(c.depot_id) for c in companies] # 各社デポID（会社順）

    # PDノード -> (pickup, delivery) の逆引き（候補列挙時に「このノードはどのPDペアか」を引く）
    node_to_pair: Dict[int, Tuple[int, int]] = {}
    for p, d in PD_map.items():
        p_i, d_i = int(p), int(d)
        node_to_pair[p_i] = (p_i, d_i)
        node_to_pair[d_i] = (p_i, d_i)

    # 各ノードが「最近傍デポ」の会社（Voronoi領域）に属するというラベル
    # node_region[nid] = 会社index
    node_region: Dict[int, int] = {}
    for c in customers:
        nid = int(c["id"])
        node_region[nid] = utils.nearest_company_of_node(
            node_id=nid,
            depot_ids=depot_ids,
            id_to_coord=id_to_coord,
        )

    # 現在割当（現状ルート）に基づく node -> 会社index の辞書（depotは除外）
    # node_owner_now[nid] = 会社index
    node_owner_now: Dict[int, int] = {}
    for ci, comp in enumerate(companies):
        dep = int(comp.depot_id)
        for r in comp.routes:
            for nid in r:
                nid_i = int(nid)
                if nid_i == dep:
                    continue
                node_owner_now[nid_i] = ci





















    # ============================
    # STEP1：候補列挙
    # ============================
    _phase("STEP1：セグメントの列挙")
    if enable_stochastic_imp:
        _p("セグメント移管に伴う改善量計算：正規分布で確率化する", indent=1)

    routes_per_company: List[List[List[int]]] = [[r[:] for r in comp.routes] for comp in companies]

    cost_per_company_before_proc1 = [
        int(sum(utils.route_cost(r, customers) for r in routes_per_company[i]))
        for i in range(n_comp)
    ]
    cost_all_company_before_proc1 = int(sum(cost_per_company_before_proc1))

    initial_cost_limit: List[int] = []
    for comp in companies:
        if hasattr(comp, "initial_total_route_length"):
            initial_cost_limit.append(int(comp.initial_total_route_length))
        else:
            initial_cost_limit.append(
                int(sum(utils.route_cost(r, customers) for r in comp.routes))
            )

    current_cost0: List[int] = [
        int(sum(utils.route_cost(r, customers) for r in routes_per_company[i]))
        for i in range(n_comp)
    ]

    avail_veh0: List[int] = []
    for i, comp in enumerate(companies):
        used = sum(1 for r in routes_per_company[i] if len(r) > 2)
        avail_veh0.append(max(0, int(comp.num_vehicles) - int(used)))

    candidates: List[Dict] = []

    for owner_idx, comp in enumerate(companies):
        depot_owner = int(comp.depot_id)
        for v_idx, r in enumerate(routes_per_company[owner_idx]):
            if len(r) <= 2:
                continue
            if int(r[0]) != depot_owner or int(r[-1]) != depot_owner:
                continue

            mid = [int(n) for n in r[1:-1] if int(n) != depot_owner]
            if not mid:
                continue

            invaded = any(node_region.get(n, owner_idx) != owner_idx for n in mid)
            if not invaded:
                continue

            n_mid = len(mid)
            for a in range(n_mid):
                open_pairs: Set[Tuple[int, int]] = set()
                cnt: Dict[Tuple[int, int], int] = {}

                for b in range(a, n_mid):
                    nid = mid[b]
                    pair = node_to_pair.get(nid)
                    if pair is None:
                        break

                    cnt[pair] = cnt.get(pair, 0) + 1
                    if cnt[pair] == 1:
                        open_pairs.add(pair)
                    elif cnt[pair] == 2:
                        open_pairs.discard(pair)
                    else:
                        break

                    if open_pairs:
                        continue

                    seg_nodes = mid[a : b + 1]
                    if not seg_nodes:
                        continue

                    tgt_set = {node_region.get(n, owner_idx) for n in seg_nodes}
                    tgt_set.discard(owner_idx)
                    if not tgt_set:
                        continue

                    seg_set = set(seg_nodes)
                    seg_pairs: List[Tuple[int, int]] = []
                    seen_pair: Set[Tuple[int, int]] = set()
                    for n in seg_nodes:
                        pr = node_to_pair[n]
                        if pr in seen_pair:
                            continue
                        p, d = pr
                        if p in seg_set and d in seg_set:
                            seg_pairs.append(pr)
                            seen_pair.add(pr)

                    if not seg_pairs:
                        continue

                    old_cost = int(utils.route_cost(r, customers))
                    new_mid = [n for n in mid if n not in seg_set]
                    new_route = [depot_owner] + new_mid + [depot_owner] if new_mid else [depot_owner, depot_owner]
                    new_cost = int(utils.route_cost(new_route, customers))
                    dec_owner = int(old_cost - new_cost)
                    if dec_owner <= 0:
                        continue

                    for tgt_idx in sorted(tgt_set):
                        depot_tgt = int(companies[tgt_idx].depot_id)
                        tgt_route = [depot_tgt] + seg_nodes + [depot_tgt]

                        ok, _, _ = utils.check_single_route_feasible(
                            tgt_route,
                            customers=customers,
                            depot_id=depot_tgt,
                            all_PD_pairs=PD_map,
                            vehicle_capacity=vehicle_capacity,
                        )
                        if not ok:
                            continue

                        inc_tgt = int(utils.route_cost(tgt_route, customers))
                        baseline_total_length_reduction = int(dec_owner - inc_tgt)

                        if enable_stochastic_imp:
                            expected_imp = utils.round(rng.gauss(baseline_total_length_reduction, stoch_net_imp_stddev))
                        else:
                            expected_imp = baseline_total_length_reduction
                        
                        candidates.append(
                            dict(
                                owner=owner_idx,
                                donor_vehicle=v_idx,
                                seg_nodes=seg_nodes,
                                seg_pairs=seg_pairs,
                                target=tgt_idx,

                                dec_owner=dec_owner,  # 送り出し側のコスト減少量
                                inc_target=inc_tgt,   # 受け入れ側のコスト増加量

                                baseline_imp=baseline_total_length_reduction,  # 全体の改善量（減少が正）
                                expected_imp=expected_imp,  # CP-SATから見える改善量（確率化ONならサンプル、OFFならbaseline）
                            )
                        )

    _p(f"見つかったセグメントの数：{len(candidates)} 件", indent=1)
    num_candidates_having_positive_imp = sum(1 for c in candidates if int(c.get("expected_imp", 0)) > 0)
    _p(f"改善量>0として扱われるセグメントの数：{num_candidates_having_positive_imp} 件", indent=1)

    """
    _subsection("■ candidates 詳細（処理1終了時点）")
    for k, c in enumerate(candidates):
        pair_cnt = int(len(c.get("seg_pairs", [])))
        dec_owner = int(c.get("dec_owner", 0))
        inc_target = int(c.get("inc_target", 0))
        baseline_imp = int(c.get("baseline_imp", 0))
        expected_imp = int(c.get("expected_imp", 0))
        
        if enable_stochastic_imp:
            _p(
                f"cand[{k}] PD数={pair_cnt}  送り出し側の減少={dec_owner}  受け入れ側の増加={inc_target}  "
                f"改善量(ナイーブ計算)={baseline_imp}  サンプリングした改善量={expected_imp}（CP-SATから見える改善量）",
                indent=1,
            )
        else:
            _p(
                f"cand[{k}] PD数={pair_cnt}  送り出し側の減少={dec_owner}  受け入れ側の増加={inc_target}  "
                f"改善量(ナイーブ計算)={baseline_imp}（CP-SATから見える改善量）",
                indent=1,
            )

    _p("会社ごとの空車両数：", indent=1)
    for i in range(n_comp):
        _p(f"LSP{i+1}（target={i}）：{int(avail_veh0[i])}", indent=2)

    if not candidates:
        _p("移管候補が無いため、現状維持で終了します。", indent=1)
        utils.refresh_allLSP_current_and_previous_cost_fields(all_LSP_state)
        _boxed_title("タスク再割り当て処理終了（現状維持）")
        return all_LSP_state
    else:
        _p()
        _p()
    """
















    # ============================
    # STEP2：CP-SAT選択＋暫定反映
    # ============================
    _p()
    _phase("STEP2：CP-SAT で移管の組合せを選択し、経路情報を更新")

    model = cp_model.CpModel()
    x = [model.NewBoolVar(f"cand_{k}") for k in range(len(candidates))]
    model.Maximize(sum(int(c["expected_imp"]) * xk for c, xk in zip(candidates, x)))

    # (a) PDペアは高々1回
    pair_to_cands: Dict[Tuple[int, int], List[int]] = {}
    for k, c in enumerate(candidates):
        for pr in c["seg_pairs"]:
            pair_to_cands.setdefault(pr, []).append(k)
    for pr, idxs in pair_to_cands.items():
        model.Add(sum(x[i] for i in idxs) <= 1)

    # (b) target の空車両数制約
    recv_count: List[List[int]] = [[] for _ in range(n_comp)]
    for k, c in enumerate(candidates):
        recv_count[c["target"]].append(k)
    for comp_idx in range(n_comp):
        if recv_count[comp_idx]:
            model.Add(sum(x[i] for i in recv_count[comp_idx]) <= int(avail_veh0[comp_idx]))

    # (c) 個別合理性（各社の現在コストが initial_total_route_length を超えない）
    for comp_idx in range(n_comp):
        effects = []
        for k, c in enumerate(candidates):
            coef = 0
            if c["owner"] == comp_idx:
                coef -= int(c["dec_owner"])
            if c["target"] == comp_idx:
                coef += int(c["inc_target"])
            if coef != 0:
                effects.append(coef * x[k])

        rhs = int(initial_cost_limit[comp_idx] - current_cost0[comp_idx])
        if effects:
            model.Add(sum(effects) <= rhs)
        else:
            model.Add(0 <= rhs)

    solver = cp_model.CpSolver()
    status = solver.Solve(model)

    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        _p("CP-SAT が実行可能な移管を見つけられませんでした。現状維持で終了します。", indent=1)
        utils.refresh_allLSP_current_and_previous_cost_fields(all_LSP_state)
        _boxed_title("タスク再割り当て処理終了（現状維持）")
        return all_LSP_state

    selected = [k for k in range(len(candidates)) if solver.Value(x[k]) == 1]
    if not selected:
        _p("移管が選ばれませんでした。現状維持で終了します。", indent=1)
        utils.refresh_allLSP_current_and_previous_cost_fields(all_LSP_state)
        _boxed_title("タスク再割り当て処理終了（現状維持）")
        return all_LSP_state

    # 暫定反映（donor:削除, target:空車へ挿入）
    incoming_segments_per_company: List[List[List[int]]] = [[] for _ in range(n_comp)]
    donor_remove_nodes: Dict[Tuple[int, int], Set[int]] = {}

    for k in selected:
        c = candidates[k]
        key = (c["owner"], c["donor_vehicle"])
        donor_remove_nodes.setdefault(key, set()).update(c["seg_nodes"])

        tgt = c["target"]
        depot_tgt = int(companies[tgt].depot_id)
        incoming_segments_per_company[tgt].append([depot_tgt] + c["seg_nodes"] + [depot_tgt])

    # donor反映（ノード削除）
    for (owner, v_idx), rmset in donor_remove_nodes.items():
        depot_owner = int(companies[owner].depot_id)
        r = routes_per_company[owner][v_idx]
        routes_per_company[owner][v_idx] = utils.remove_nodes_keep_depots(r, depot_owner, rmset)

    # target反映（空車へ挿入）
    for tgt in range(n_comp):
        segs = incoming_segments_per_company[tgt]
        if not segs:
            continue
        
        for seg_route in segs:
            placed = False
            for i in range(len(routes_per_company[tgt])):
                r = routes_per_company[tgt][i]
                # 厳密な「空車両」判定：ちょうど [depot, depot]
                if len(r) == 2 and r[0] == companies[tgt].depot_id and r[1] == companies[tgt].depot_id:
                    routes_per_company[tgt][i] = seg_route
                    placed = True
                    break

            if not placed:
                print(f"会社{tgt+1} の受け入れ車両が足りなかった。SATソルバーの実装ミスの可能性あり。")
                raise RuntimeError(f"会社{tgt+1} の受け入れ車両が足りなかった。SATソルバーの実装ミスの可能性あり。")
        
        if len(routes_per_company[tgt]) != companies[tgt].num_vehicles:
            print(f"会社{tgt+1} の部分経路移管後の経路本数と最大車両台数が一致していない")
            raise RuntimeError(f"route_count={len(routes_per_company[tgt])} != num_max_vehicle={companies[tgt].num_vehicles}")


    _subsection("■ 受け入れ状況")
    for i in range(n_comp):
        _p(f"LSP{i+1}：他社から引き受けた部分経路の数 = {len(incoming_segments_per_company[i])} 件", indent=1)

    after_proc2_per = [
        int(sum(utils.route_cost(r, customers) for r in routes_per_company[i]))
        for i in range(n_comp)
    ]
    after_proc2_total = int(sum(after_proc2_per))
    _print_cost_summary(
        "STEP2終了（移管選択＋暫定反映）",
        prev_per=cost_per_company_before_proc1,
        cur_per=after_proc2_per,
        prev_total=cost_all_company_before_proc1,
        cur_total=after_proc2_total,
    )
    
    if check_feasibility:
        _run_feasibility_check_on_routes("STEP2後（暫定反映後）", routes_per_company)

    # ============================
    # stateへ反映
    # ============================
    interim_routes_all: List[List[int]] = []
    for i in range(n_comp):
        interim_routes_all.extend([r[:] for r in routes_per_company[i]])

    all_LSP_state.set_all_routes(interim_routes_all)
    utils.refresh_allLSP_current_and_previous_cost_fields(all_LSP_state)

    if check_feasibility:
        _run_feasibility_check_on_routes("STEP2後（再割り当て終了済み）", routes_per_company)

    _boxed_title("セグメント移管処理終了")
    return all_LSP_state



def optimize_individually_by_ORTools(
    all_LSP_state: CollaborativeState,
    check_feasibility: bool = False,
) -> CollaborativeState:
    """
    ORToolsによる社内（個別）最適化
    """

    # ============================
    # 出力ユーティリティ（optimize_by_partial_route_reassignment と同等の体裁）
    # ============================
    def _p(msg: str = "", indent: int = 0) -> None:
        if msg == "":
            print(">")
            return
        print("> " + ("  " * indent) + msg)

    def _vis_w(s: str) -> int:
        w = 0
        for ch in s:
            w += 2 if unicodedata.east_asian_width(ch) in ("F", "W", "A") else 1
        return w

    def _rule(char: str = "─", width: int = 56) -> None:
        _p(char * width)

    def _boxed_title(title: str, width: int = 56) -> None:
        w = max(width, _vis_w(title))
        _rule("─", w)
        _p(title)
        _rule("─", w)

    def _phase(title: str) -> None:
        _p(f"=== {title} ===")

    def _subsection(title: str) -> None:
        _p()
        _p(title)

    def _mode(warm: bool) -> str:
        return "ウォームスタート" if warm else "コールドスタート"

    # =========================================================
    # Feasibility 検証機（check_feasibility=True のときのみ動作）
    # =========================================================
    def _run_feasibility_check_on_routes(label: str, routes_per_company: List[List[List[int]]]) -> None:
        _subsection(f"■ Feasibility check（{label}）")

        for comp_idx, comp in enumerate(companies):
            depot_id = int(comp.depot_id)
            num_v = int(comp.num_vehicles)

            if comp_idx >= len(routes_per_company):
                _p(f"ERROR: routes_per_company に会社{comp_idx}の情報がありません", indent=1)
                raise RuntimeError(f"feasibility_failed: missing_company_routes comp_idx={comp_idx} label={label}")

            if len(routes_per_company[comp_idx]) != num_v:
                _p(
                    f"ERROR: 会社{comp_idx}のルート本数が車両台数と一致しません "
                    f"(routes={len(routes_per_company[comp_idx])}, vehicles={num_v})",
                    indent=1,
                )
                raise RuntimeError(
                    f"feasibility_failed: route_count_mismatch comp_idx={comp_idx} "
                    f"routes={len(routes_per_company[comp_idx])} vehicles={num_v} label={label}"
                )

            for v_idx, route in enumerate(routes_per_company[comp_idx]):
                ok, reason, detail = utils.check_single_route_feasible(
                    route,
                    customers=customers,
                    depot_id=depot_id,
                    all_PD_pairs=PD_map,
                    vehicle_capacity=vehicle_capacity,
                )
                if not ok:
                    _p(f"ERROR: INFEASIBLE at {label}", indent=1)
                    _p(f"LSP{comp_idx+1} vehicle{v_idx} reason={reason}", indent=1)
                    if detail is not None:
                        _p(f"detail={detail}", indent=2)
                    _p(f"route={route}", indent=2)
                    raise RuntimeError(
                        f"feasibility_failed: {label} LSP{comp_idx+1} vehicle{v_idx} reason={reason}"
                    )

        _p("判定：FEASIBLE", indent=1)
        
        
        
        
        


    # ============================
    # データ準備（state から必要情報を取得）
    # ============================
    customers = all_LSP_state.all_customers
    PD_map = all_LSP_state.all_PD_pairs
    vehicle_capacity = int(all_LSP_state.vehicle_capacity)
    companies = all_LSP_state.companies
    n_comp = len(companies)

    # PDノード -> (pickup, delivery) の逆引き（ORTools入力の整合性チェック用）
    node_to_pair: Dict[int, Tuple[int, int]] = {}
    for p, d in PD_map.items():
        p_i, d_i = int(p), int(d)
        node_to_pair[p_i] = (p_i, d_i)
        node_to_pair[d_i] = (p_i, d_i)


    # ============================
    # 開始メッセージ
    # ============================
    print("")
    _boxed_title("社内最適化（OR-Tools）開始")

    routes_per_company: List[List[List[int]]] = [[r[:] for r in comp.routes] for comp in companies]
    cost_per_company_before = [
        int(sum(utils.route_cost(r, customers) for r in routes_per_company[i]))
        for i in range(n_comp)
    ]
    cost_all_before = int(sum(cost_per_company_before))

    if check_feasibility:
        _run_feasibility_check_on_routes("開始直後（入力ルート）", routes_per_company)

    # ============================
    # 処理3）各社ごとに全体経路再作成（assigned 全体）
    # ============================
    # --- 各社の暫定PDペアを再集計（ORToolsの入力に必要） ---
    all_pairs: List[Tuple[int, int]] = [(int(p), int(d)) for p, d in PD_map.items()]
    assigned_pairs_per_company: List[List[Tuple[int, int]]] = [[] for _ in range(n_comp)]

    for comp_idx, comp in enumerate(companies):
        depot_id = int(comp.depot_id)
        present_pairs: Set[Tuple[int, int]] = set()

        for v_idx, r in enumerate(routes_per_company[comp_idx]):
            mids = [int(n) for n in r if int(n) != depot_id]
            mid_set = set(mids)

            for nid in mids:
                pr = node_to_pair.get(nid)
                if pr is None:
                    print(f"ERROR: PDノードでないノードがルートに含まれています。company={comp_idx} vehicle={v_idx} nid={nid}")
                    print(f"route={r}")
                    raise RuntimeError(f"node_not_in_PD_map company={comp_idx} vehicle={v_idx} nid={nid}")

                p, d = pr
                if p not in mid_set or d not in mid_set:
                    missing = d if p in mid_set else p
                    print(
                        "ERROR: 兄弟PDが同一ルート内に存在しません。"
                        f" company={comp_idx} vehicle={v_idx} pair=({p},{d}) missing={missing}"
                    )
                    print(f"route={r}")
                    raise RuntimeError(
                        f"PD_pair_split_in_route company={comp_idx} vehicle={v_idx} pair=({p},{d}) missing={missing}"
                    )
                present_pairs.add((p, d))
        assigned_pairs_per_company[comp_idx] = [pr for pr in all_pairs if pr in present_pairs]

    routes_after_proc3: List[List[List[int]]] = [[] for _ in range(n_comp)]

    for comp_idx, comp in enumerate(companies):
        depot_id = int(comp.depot_id)
        num_available_vehicle = int(comp.num_vehicles)
        assigned_pairs = assigned_pairs_per_company[comp_idx]

        # init_routes_without_depot：ORToolsのウォームスタート時の初期解（出発・到着デポを削除した形）
        init_routes_without_depot = [utils.strip_depot(r, depot_id) for r in routes_per_company[comp_idx]]

        if len(init_routes_without_depot) != num_available_vehicle:
            print(f"会社{comp_idx}に保存されている経路の本数と車両台数が一致していない")
            raise RuntimeError(f"route_count={len(init_routes_without_depot)} != num_available_vehicle={num_available_vehicle}")

        # 各社の暫定顧客辞書を再集計（ORToolsの入力に必要）
        node_ids = {depot_id}
        for p, d in assigned_pairs:
            node_ids.add(int(p))
            node_ids.add(int(d))
        sub_customers = [c for c in customers if int(c["id"]) in node_ids]

        fallback_used = False
        reason_no_solution = False

        # --- ソルバー実行 経路再作成 ---
        new_routes = solve_vrp_flexible(
            customers=sub_customers,
            initial_routes=init_routes_without_depot,
            PD_pairs=assigned_pairs,
            num_vehicles=num_available_vehicle,
            vehicle_capacity=vehicle_capacity,
            start_depots=[depot_id] * num_available_vehicle,
            end_depots=[depot_id] * num_available_vehicle,
            use_capacity=True,
            use_time=True,
            use_pickup_delivery=True,
            Warm_Start=True,
        )

        if new_routes is None:
            new_routes = [r[:] for r in routes_per_company[comp_idx]]
            fallback_used = True
            reason_no_solution = True

        if len(new_routes) != num_available_vehicle:
            print(f"ソルバー（ORTools）の出力経路本数と車両台数が一致していない")
            raise RuntimeError(f"route_count={len(new_routes)} != num_available_vehicle={num_available_vehicle}")

        baseline_cost = int(sum(utils.route_cost(r, customers) for r in routes_per_company[comp_idx]))
        solver_cost = int(sum(utils.route_cost(r, customers) for r in new_routes))
        if solver_cost > baseline_cost:
            new_routes = [r[:] for r in routes_per_company[comp_idx]]
            fallback_used = True

        _subsection(f"● LSP{comp_idx+1}：全タスク（assigned）の最終再生成結果")
        _p(f"再生成前（暫定解）の経路長：{baseline_cost}", indent=1)
        _p(f"OR-Tools が生成した経路長：{solver_cost}", indent=1)

        if not fallback_used:
            _p("判定：OR-Tools 解を採用（悪化なし）", indent=1)
        else:
            if reason_no_solution:
                _p("判定：暫定解に戻す（フォールバック：解が得られなかったため）", indent=1)
            elif solver_cost > baseline_cost:
                _p("判定：暫定解に戻す（フォールバック：経路長が悪化したため）", indent=1)
            else:
                _p("判定：暫定解に戻す（フォールバック）", indent=1)

        routes_after_proc3[comp_idx] = new_routes


    if check_feasibility:
        _run_feasibility_check_on_routes("ORToolsによる個別最適化実行後", routes_after_proc3)

    # ============================
    # stateへ反映
    # ============================
    final_routes_all: List[List[int]] = []
    for i in range(n_comp):
        final_routes_all.extend([r[:] for r in routes_after_proc3[i]])

    all_LSP_state.set_all_routes(final_routes_all)
    utils.refresh_allLSP_current_and_previous_cost_fields(all_LSP_state)

    _p()
    _boxed_title("社内最適化（OR-Tools）終了")
    return all_LSP_state



def optimize_individually_by_GAT(
    all_LSP_state,
    exact_pd_pair_limit,
    debug_2gat: bool = False
):
    """
    社内限定GAT（2車両GATによる社内経路最適化）

    目的：
      会社内部で2車両GATを行い、社内経路最適化を目指す

    処理：
      1) 2車両合計PDペア数が exact_pd_pair_limit 以下なら自作ソルバー、
         それ以外は ORTools により2車両VRPを解く
      2) new_routes が None なら候補追加せず次のペアへ
      3) 返り値の先頭/最後尾デポを用いて new_routes[0],[1] を (i,j) に対応付け
      4) 改善されていれば solver 解をアクション集合に追加
      5) 改善があった場合のみ exchanged route を作成し、
         2車両とも feasible な場合のみアクション集合に追加（探索空間拡張）
      6) CP-SATで社内改善量最大となるマッチング（各車両は高々1回）を選択
         ※個別合理性制約は入れない
      7) 全社の更新が終わったら return

    デバッグ（debug=True のときのみ）：
      - デバッグ1：処理3と処理4の間で new_routes の2車両分を feasibility 検証し、
                   NGなら即 RuntimeError で停止
      - デバッグ2：処理7直後（全社更新後）に全ルートを feasibility 検証し、
                   NGなら即 RuntimeError で停止
    """

    # 各社ごとに最適化して、そのまま company.routes に反映
    for comp_idx, comp in enumerate(all_LSP_state.companies):
        comp_label = f"LSP{comp_idx+1}"
        original_routes = comp.routes
        num_vehicles = len(original_routes)

        # ----------------------------
        # 車両ごとの「その車両ルートに完全に含まれるPDペア」を前計算
        # ----------------------------
        PD_pairs_of_each_vehicle: List[List[Tuple[int, int]]] = []
        for vehicle_route in original_routes:
            visited = set(int(n) for n in vehicle_route)
            related_pairs = []
            for p, d in all_LSP_state.all_PD_pairs.items():
                p = int(p)
                d = int(d)
                if p in visited and d in visited:
                    related_pairs.append((p, d))
            PD_pairs_of_each_vehicle.append(related_pairs)

        feasible_actions: List[dict] = []

        # ----------------------------
        # 全ての2車両ペアで候補収集
        # ----------------------------
        for i in range(num_vehicles):
            for j in range(i + 1, num_vehicles):

                if debug_2gat:
                    print(f"\n[DEBUG] {comp_label} 2車両VRP: pair=({i},{j})")
                    print(f"        route{i}={original_routes[i]}")
                    print(f"        route{j}={original_routes[j]}")

                # 2車両の対象PD（重複除去）
                sub_PD_pairs = PD_pairs_of_each_vehicle[i] + PD_pairs_of_each_vehicle[j]
                if sub_PD_pairs:
                    sub_PD_pairs = list({(int(p), int(d)) for (p, d) in sub_PD_pairs})

                # デポ（車両ごとに異なる可能性を許容）
                start_i = int(original_routes[i][0])
                start_j = int(original_routes[j][0])
                start_depots = [start_i, start_j]
                end_depots = [start_i, start_j]

                # 対象ノード集合（両ルート + PD端点 + デポ）
                combined_node_ids = set(int(n) for n in original_routes[i]) | set(int(n) for n in original_routes[j])
                combined_node_ids.add(start_i)
                combined_node_ids.add(start_j)
                for p, d in sub_PD_pairs:
                    combined_node_ids.add(int(p))
                    combined_node_ids.add(int(d))

                # customers側に存在するノードだけに絞る
                sub_customers = [c for c in all_LSP_state.all_customers if int(c["id"]) in combined_node_ids]
                id_set = {int(c["id"]) for c in sub_customers}
                sub_PD_pairs = [(p, d) for (p, d) in sub_PD_pairs if (p in id_set and d in id_set)]

                # WarmStart用：デポ除去（ReadAssignmentFromRoutes想定）
                initial_routes = [
                    (r[1:-1] if len(r) >= 2 and r[0] == r[-1] else r)
                    for r in [original_routes[i], original_routes[j]]
                ]

                # ----- 分岐：PDペア数で「厳密」or「ORTools」 -----
                use_exact = (len(sub_PD_pairs) <= exact_pd_pair_limit)
                used_solver = "Exact" if use_exact else "ORTools"

                if debug_2gat:
                    print(f"        -> solver={used_solver}, |PD|={len(sub_PD_pairs)}, limit={exact_pd_pair_limit}")

                if use_exact:
                    new_routes = solve_exact_2vehicle_vrp(
                        sub_customers=sub_customers,
                        sub_PD_pairs=sub_PD_pairs,
                        start_depots=start_depots,
                        end_depots=end_depots,
                        vehicle_capacity=all_LSP_state.vehicle_capacity,
                    )
                else:
                    new_routes = solve_vrp_flexible(
                        sub_customers,
                        initial_routes,
                        sub_PD_pairs,
                        2,
                        all_LSP_state.vehicle_capacity,
                        start_depots,
                        end_depots,
                        use_capacity=True,
                        use_time=True,
                        use_pickup_delivery=True,
                        Warm_Start=True,
                    )

                # 処理2：解なしは候補追加せず次へ
                if new_routes is None:
                    continue
                if len(new_routes) != 2:
                    raise RuntimeError(f"{comp_label}: 2車両問題なのに new_routes の本数が不正: len={len(new_routes)}")

                # 処理3：返り値の対応付け（(i,j)順とは限らない）
                # デポが同一のときは対称性があり、順序入替えが意味を持たないため、そのまま受ける。
                # デポが異なるときのみ、start/end を見て (i,j) に対応付けする。
                if start_i != start_j:
                    r0, r1 = new_routes[0], new_routes[1]

                    def _match_start_end(r, s, e) -> bool:
                        return bool(r) and int(r[0]) == int(s) and int(r[-1]) == int(e)

                    if _match_start_end(r0, start_i, start_i) and _match_start_end(r1, start_j, start_j):
                        pass
                    elif _match_start_end(r0, start_j, start_j) and _match_start_end(r1, start_i, start_i):
                        new_routes = [r1, r0]
                    else:
                        raise RuntimeError(f"{comp_label}: 返ってきた2車両経路が想定外の形（デポ対応付け不能） pair=({i},{j})")

                # ----------------------------
                # デバッグ1：solver出力 new_routes のfeasibility（NGなら即停止）
                # ----------------------------
                if debug_2gat:
                    ok0, reason0, det0 = utils.check_single_route_feasible(
                        new_routes[0],
                        customers=all_LSP_state.all_customers,
                        depot_id=int(start_depots[0]),
                        all_PD_pairs=all_LSP_state.all_PD_pairs,
                        vehicle_capacity=int(all_LSP_state.vehicle_capacity),
                    )
                    ok1, reason1, det1 = utils.check_single_route_feasible(
                        new_routes[1],
                        customers=all_LSP_state.all_customers,
                        depot_id=int(start_depots[1]),
                        all_PD_pairs=all_LSP_state.all_PD_pairs,
                        vehicle_capacity=int(all_LSP_state.vehicle_capacity),
                    )
                    if (not ok0) or (not ok1):
                        if not ok0:
                            print(f"[DEBUG-2vehicle check] - vehicle{i} route={new_routes[0]}は実行不能解です")
                            print(f"                         reason={reason0}")
                            if det0:
                                for k, v in det0.items():
                                    print(f"    {k}={v}")
                        if not ok1:
                            print(f"[DEBUG-2vehicle check] - vehicle{j} route={new_routes[1]}")
                            print(f"                         reason={reason1}")
                            if det1:
                                for k, v in det1.items():
                                    print(f"    {k}={v}")
                        print("")
                        raise RuntimeError(
                            f"[DEBUG] ソルバーが実行不可能な2車両解を出力しています（{comp_label}, pair={i},{j}, solver={used_solver}）"
                        )
                    else:
                        print(f"        -> new route={new_routes}は実行可能な解です")
                        print("")

                # 処理4：改善判定（改善のみ solver 解を追加）
                old_cost = utils.route_cost(original_routes[i], all_LSP_state.all_customers) + \
                           utils.route_cost(original_routes[j], all_LSP_state.all_customers)
                new_cost = utils.route_cost(new_routes[0], all_LSP_state.all_customers) + \
                           utils.route_cost(new_routes[1], all_LSP_state.all_customers)

                improved = (new_cost < old_cost)

                if improved:
                    feasible_actions.append({
                        "vehicle_pair": (i, j),
                        "new_routes": new_routes,
                        "old_cost": old_cost,
                        "new_cost": new_cost,
                        "cost_improvement": old_cost - new_cost,
                        "meta": {"kind": "solver", "solver": used_solver, "pd_pairs": len(sub_PD_pairs)},
                    })

                # 処理5：改善があったときだけ exchanged を生成し、feasible のときだけ追加
                if improved:
                    depot_i = int(new_routes[0][0])
                    depot_j = int(new_routes[1][0])

                    # depot を除いた中身（途中に depot が混入していても除外）
                    mid_i = [int(n) for n in new_routes[0] if int(n) != depot_i]
                    mid_j = [int(n) for n in new_routes[1] if int(n) != depot_j]

                    exchanged_routes = [
                        [depot_i] + mid_j + [depot_i],
                        [depot_j] + mid_i + [depot_j],
                    ]

                    ok0x, _, _ = utils.check_single_route_feasible(
                        exchanged_routes[0],
                        customers=all_LSP_state.all_customers,
                        depot_id=depot_i,
                        all_PD_pairs=all_LSP_state.all_PD_pairs,
                        vehicle_capacity=int(all_LSP_state.vehicle_capacity),
                    )
                    ok1x, _, _ = utils.check_single_route_feasible(
                        exchanged_routes[1],
                        customers=all_LSP_state.all_customers,
                        depot_id=depot_j,
                        all_PD_pairs=all_LSP_state.all_PD_pairs,
                        vehicle_capacity=int(all_LSP_state.vehicle_capacity),
                    )

                    if ok0x and ok1x:
                        exchanged_cost = utils.route_cost(exchanged_routes[0], all_LSP_state.all_customers) + \
                                         utils.route_cost(exchanged_routes[1], all_LSP_state.all_customers)
                        feasible_actions.append({
                            "vehicle_pair": (i, j),
                            "new_routes": exchanged_routes,
                            "old_cost": old_cost,
                            "new_cost": exchanged_cost,
                            "cost_improvement": old_cost - exchanged_cost,  # 負もあり得るが除外しない
                            "meta": {"kind": "exchanged", "solver": used_solver, "pd_pairs": len(sub_PD_pairs)},
                        })

        # 候補が無ければ更新なし
        if not feasible_actions:
            comp.validate()
            continue

        # ----------------------------
        # 処理6：CP-SAT（各車両は高々1回、総改善量最大）
        # ※個別合理性制約は入れない
        # ----------------------------
        model = cp_model.CpModel()
        x = [model.NewBoolVar(f"action_{k}") for k in range(len(feasible_actions))]
        model.Maximize(sum(int(round(a["cost_improvement"])) * x[k] for k, a in enumerate(feasible_actions)))

        vehicle_to_actions: Dict[int, List[int]] = {}
        for k, a in enumerate(feasible_actions):
            vi, vj = a["vehicle_pair"]
            vehicle_to_actions.setdefault(vi, []).append(k)
            vehicle_to_actions.setdefault(vj, []).append(k)
        for v, idxs in vehicle_to_actions.items():
            model.Add(sum(x[k] for k in idxs) <= 1)

        solver = cp_model.CpSolver()
        status = solver.Solve(model)

        new_all_vehicles_routes = original_routes.copy()
        touched = set()

        if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            for k, a in enumerate(feasible_actions):
                if solver.Value(x[k]) == 1:
                    vi, vj = a["vehicle_pair"]
                    r0, r1 = a["new_routes"]
                    new_all_vehicles_routes[vi] = r0
                    new_all_vehicles_routes[vj] = r1
                    touched.add(vi)
                    touched.add(vj)
        else:
            print(f"[GAT] {comp_label}: 最適なアクションの組み合わせが見つかりませんでした。")

        comp.routes = new_all_vehicles_routes
        comp.validate()

        if debug_2gat and touched:
            print(f"[DEBUG] {comp_label}: touched vehicles = {sorted(touched)}")

    # 処理7：全社更新後に cost fields を更新
    utils.refresh_allLSP_current_and_previous_cost_fields(all_LSP_state)

    # ----------------------------
    # デバッグ2：全体スキャン（全社更新後の全ルートが feasible か）
    # ----------------------------
    if debug_2gat:
        total_bad = 0
        for comp_idx, comp in enumerate(all_LSP_state.companies):
            comp_label = f"LSP{comp_idx+1}"
            for vi, r in enumerate(comp.routes):
                ok, reason, det = utils.check_single_route_feasible(
                    r,
                    customers=all_LSP_state.all_customers,
                    depot_id=int(comp.depot_id),
                    all_PD_pairs=all_LSP_state.all_PD_pairs,
                    vehicle_capacity=int(all_LSP_state.vehicle_capacity),
                )
                if not ok:
                    total_bad += 1
                    print(f"[DEBUG-final check] {comp_label} vehicle={vi} の経路は実行不能解です")
                    print(f"                 route = {r}")
                    print(f"                 reason = {reason}")
                    if det:
                        for k, v in det.items():
                            print(f"                 {k}={v}")
                    raise RuntimeError(f"[DEBUG-final check] {comp_label} vehicle={vi} の経路は実行不能解です")

        print("[DEBUG] 社内GAT後の全経路はすべて実行可能です\n")

    return all_LSP_state

