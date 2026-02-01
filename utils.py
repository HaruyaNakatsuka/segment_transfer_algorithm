from __future__ import annotations

import logging
import math
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union, Set
from model import CollaborativeState, CompanyState, Customer, PDPairDict


def setup_logging(show_progress: bool = True) -> None:
    logging.basicConfig(
        level=logging.INFO if show_progress else logging.WARNING,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        force=True,
    )


def round(x: float) -> int:
    # 四捨五入（pythonのround関数は".5"のとき偶数丸めをするから使いたくない）
    if x >= 0:
        return int(math.floor(x + 0.5))
    else:
        return int(math.ceil(x - 0.5))


def validate_stochastic_stddev(stddev: float) -> None:
    if stddev < 0:
        raise ValueError(f"Invalid STOCH_NET_IMP_STDDEV={stddev}: must be >= 0.")
    if stddev == 0:
        raise ValueError(
            "Invalid STOCH_NET_IMP_STDDEV=0.0: stddev=0 means no stochasticity (cannot stochasticize)."
        )


def build_id_to_coord(customers: Sequence[Mapping[str, Any]]) -> Dict[int, Tuple[float, float]]:
    """customers から {node_id: (x, y)} を構築する。"""
    id_to_coord: Dict[int, Tuple[float, float]] = {}
    for c in customers:
        nid = int(c["id"])
        id_to_coord[nid] = (float(c["x"]), float(c["y"]))
    return id_to_coord


def euclid_int_dist(a: int, b: int, *, id_to_coord: Mapping[int, Tuple[float, float]]) -> int:
    """ユークリッド距離→切り捨て→int（プロジェクト標準の距離定義）。"""
    ax, ay = id_to_coord[int(a)]
    bx, by = id_to_coord[int(b)]
    return int(math.hypot(ax - bx, ay - by))  # int-cast = floor for non-negative


def route_cost(route, customers):
    """
    1台の車両経路の距離を計算する関数
    
    【重要】
    Google ORToolsのVRPソルバーは、距離としてint型しか扱えない
    よってこのプロジェクト内では2点間の距離は
      1. ユークリッド距離を計算
      2. 小数点以下切り捨て
      3. 明示的にint型に整形
    をしている
    """
    id_to_coord = {c['id']: (c['x'], c['y']) for c in customers}
    cost: int = 0
    for i in range(len(route) - 1):
        x1, y1 = id_to_coord[route[i]]
        x2, y2 = id_to_coord[route[i + 1]]
        cost += int(math.floor(((x2 - x1)**2 + (y2 - y1)**2)**0.5))
    return cost


def nearest_company_of_node(node_id: int, depot_ids: Sequence[int], id_to_coord: Mapping[int, Tuple[float, float]],) -> int:
    """node_id に最も近い depot を持つ会社 index を返す（Voronoi領域ラベル用）。"""
    best_i = 0
    best_d = 10**18
    nid = int(node_id)
    for i, dep in enumerate(depot_ids):
        dd = euclid_int_dist(nid, int(dep), id_to_coord=id_to_coord)
        if dd < best_d:
            best_d = dd
            best_i = i
    return int(best_i)


def strip_depot(route: Optional[Sequence[int]], depot_id: int) -> List[int]:
    """[depot, ..., depot] から depot を除去して中間ノード列だけを返す。"""
    if not route:
        return []
    dep = int(depot_id)
    r = [int(x) for x in route]
    if len(r) >= 2 and r[0] == dep and r[-1] == dep:
        return [n for n in r[1:-1] if n != dep]
    return [n for n in r if n != dep]


def remove_nodes_keep_depots(route: Optional[Sequence[int]], depot_id: int, remove_set: Set[int]) -> List[int]:
    """
    route から remove_set のノードを除去しつつ、先頭末尾の depot は保持する。
    結果として空になれば [depot, depot] を返す。
    """
    dep = int(depot_id)
    if not route or len(route) <= 2:
        return [dep, dep]
    mids = [int(n) for n in route[1:-1] if int(n) not in remove_set and int(n) != dep]
    if not mids:
        return [dep, dep]
    return [dep] + mids + [dep]


def flatten_routes_per_company(
    routes_per_company: Sequence[Sequence[Sequence[int]]],
    *,
    companies: Sequence[CompanyState],
) -> List[List[int]]:
    """routes_per_company を set_all_routes 用の 1次元 list に畳み込み（会社順に連結）。"""
    flat: List[List[int]] = []
    for ci, comp in enumerate(companies):
        assert ci < len(routes_per_company)
        assert len(routes_per_company[ci]) == int(comp.num_vehicles)

        flat.extend([list(map(int, r)) for r in routes_per_company[ci]])
    return flat


def refresh_allLSP_current_and_previous_cost_fields(All_LSP_state: CollaborativeState) -> None:
    """
    CompanyState / CollaborativeState の「暫定経路長」を、現在の routes に基づいて上書きする。
    """
    total = 0
    for company in All_LSP_state.companies:
        company.previous_total_route_length = company.current_total_route_length
        company.current_total_route_length = int(
            sum(route_cost(r, All_LSP_state.all_customers) for r in company.routes)
        )
        total += company.current_total_route_length
    All_LSP_state.previous_total_route_length = All_LSP_state.current_total_route_length
    All_LSP_state.current_total_route_length = int(total)


def print_cost_table(all_LSP_state: CollaborativeState, title: str) -> None:
    """
    all_LSP_state に記録された経路長（initial/previous/current）をもとに、
    会社ごと＋全体の情報をテーブル表示する。

    表示項目：
      - 現在の経路長
      - 直前の経路長からの短縮率（%表記。短縮→マイナス）
      - 初期経路長からの短縮率（%表記。短縮→マイナス）
    """
    # 出力の最上段に title を表示
    print("")
    print(title)

    colw = 13
    headers = ["現在経路長", "直前比(%)", "初期比(%)"]
    print(" " * 7 + "".join(f"{h:>{colw}}" for h in headers))

    colw = 15
    for comp in all_LSP_state.companies:
        cur = float(comp.current_total_route_length)
        prev = float(comp.previous_total_route_length)
        init = float(comp.initial_total_route_length)

        round_rate = ((cur - prev) / prev * 100.0) if prev > 0 else 0.0
        init_rate = ((cur - init) / init * 100.0) if init > 0 else 0.0

        print(
            f"LSP {comp.idx + 1:<2} "
            f"{cur:>{colw}.0f}"
            f"{round_rate:>{colw}.2f}"
            f"{init_rate:>{colw}.2f}"
        )

    # TOTAL 行
    cur_t = float(all_LSP_state.current_total_route_length)
    prev_t = float(all_LSP_state.previous_total_route_length)
    init_t = float(all_LSP_state.initial_total_route_length)

    round_total = ((cur_t - prev_t) / prev_t * 100.0) if prev_t > 0 else 0.0
    init_total = ((cur_t - init_t) / init_t * 100.0) if init_t > 0 else 0.0

    print(
        f"{'TOTAL':<6} "
        f"{cur_t:>{colw}.0f}"
        f"{round_total:>{colw}.2f}"
        f"{init_total:>{colw}.2f}"
    )


def count_tasks_per_company(companies: Sequence[CompanyState]) -> List[int]:
    """
    使用例）print(f"現在のタスク個数：{count_tasks_per_company(CollaborativeState.companies)}")
    """
    task_counts: List[int] = []
    for comp in companies:
        count = 0
        for r in comp.routes:
            count += max(0, len(r) - 2)
        task_counts.append(count)
    return task_counts


def rank_pd_pairs_by_midpoint_to_voronoi_boundary(
    all_customers: List[Customer],
    PD_pairs: PDPairDict,
    depot_id_list: List[int],
) -> List[Dict]:
    """
    PDペアの「重心→最寄りボロノイ境界」距離で昇順に並べる。
    """
    id2xy = {c["id"]: (c["x"], c["y"]) for c in all_customers}
    depot_xy = [(d, id2xy[d]) for d in depot_id_list if d in id2xy]

    boundary_dist_ranked = []
    for pu, de in PD_pairs.items():
        if pu not in id2xy or de not in id2xy:
            continue
        x1, y1 = id2xy[pu]
        x2, y2 = id2xy[de]
        mid = ((x1 + x2) * 0.5, (y1 + y2) * 0.5)

        dists: List[Tuple[float, int]] = []
        for depot_id, xy in depot_xy:
            dists.append((math.hypot(mid[0] - xy[0], mid[1] - xy[1]), depot_id))
        dists.sort(key=lambda t: t[0])

        if not dists:
            continue

        d1, dep1 = dists[0]
        d2, dep2 = dists[1] if len(dists) > 1 else (float("inf"), None)
        dist_to_boundary = (d2 - d1) * 0.5 if math.isfinite(d2) else float("inf")

        boundary_dist_ranked.append(
            {
                "pickup": pu,
                "delivery": de,
                "midpoint": mid,
                "d1": d1,
                "d2": d2,
                "dist_to_boundary": dist_to_boundary,
                "nearest_depot": dep1,
                "second_nearest_depot": dep2,
            }
        )

    boundary_dist_ranked.sort(key=lambda r: r["dist_to_boundary"])
    return boundary_dist_ranked


def check_single_route_feasible(
    route: Optional[Sequence[int]],
    *,
    customers: Sequence[Dict[str, Any]],
    depot_id: int,
    all_PD_pairs: Union[Mapping[int, int], Iterable[Tuple[int, int]]],
    vehicle_capacity: int,
) -> Tuple[bool, Optional[str], Optional[Dict[str, Any]]]:
    """
    単一路線のfeasibilityを検査する（depot return due を含む）。

    返り値:
      (ok, reason, detail)
        ok=True なら feasible
        ok=False なら infeasible（reasonは種別、detailはデバッグ用情報）

    チェック内容（単一路線）:
      - 形式: [depot, ..., depot]
      - 重複訪問なし（depot除く）
      - PD: 同一ルート内 & pickupがdeliveryより先
      - 容量: 0 <= load <= capacity
      - TW: 到着→待機→サービス開始が due を超えない（depot含む）
      - 余計なノード（PDに属さないノード）を含まない（depot除く）

    ※ customer dict のキー名は揺れを吸収する（ready/due/service/demand, x/y など）。
    ※ demand は「pickup正 / delivery負」を想定するが、符号が逆でもPDマップで正規化する。
    """

    READY_KEYS = ["ready", "ready_time", "tw_start", "start_time", "r"]
    DUE_KEYS = ["due", "due_time", "tw_end", "end_time", "d"]
    SERVICE_KEYS = ["service", "service_time", "s"]
    DEMAND_KEYS = ["demand", "q", "Q", "quantity", "load"]

    def _get_num(cust: Dict[str, Any], keys: List[str], default: float = 0.0) -> float:
        for k in keys:
            if k in cust and cust[k] is not None:
                try:
                    return float(cust[k])
                except Exception:
                    pass
        return float(default)

    id2c: Dict[int, Dict[str, Any]] = {int(c["id"]): c for c in customers}

    # PDマップ（pickup->delivery）
    if isinstance(all_PD_pairs, Mapping):
        pickup_to_delivery: Dict[int, int] = {int(p): int(d) for p, d in all_PD_pairs.items()}
    else:
        pickup_to_delivery = {int(p): int(d) for (p, d) in all_PD_pairs}

    delivery_to_pickup: Dict[int, int] = {d: p for p, d in pickup_to_delivery.items()}
    node_to_pair: Dict[int, Tuple[int, int]] = {}
    for p, d in pickup_to_delivery.items():
        node_to_pair[p] = (p, d)
        node_to_pair[d] = (p, d)

    def _dist(a: int, b: int) -> int:
        ca = id2c.get(int(a))
        cb = id2c.get(int(b))
        if ca is None or cb is None:
            return 10**9
        ax = _get_num(ca, ["x", "X"], 0.0)
        ay = _get_num(ca, ["y", "Y"], 0.0)
        bx = _get_num(cb, ["x", "X"], 0.0)
        by = _get_num(cb, ["y", "Y"], 0.0)
        return int(math.hypot(bx - ax, by - ay))  # floor(int-cast)

    def _norm_demand(nid: int, raw: float) -> float:
        # pickupは正、deliveryは負に正規化（データが逆符号でも吸収）
        if nid in pickup_to_delivery:
            return abs(raw)
        if nid in delivery_to_pickup:
            return -abs(raw)
        return raw

    # ----------------------------
    # 形式チェック
    # ----------------------------
    if route is None:
        return (False, "RouteNone", {"route": route})

    r = [int(x) for x in route]
    if len(r) < 2:
        # 空/短すぎるルートは「実行する仕事がない」扱いにする（呼び出し側の方針次第）
        return (True, None, None)

    if r[0] != int(depot_id) or r[-1] != int(depot_id):
        return (
            False,
            "RouteForm(depot head/tail mismatch)",
            {"route": r, "expected_depot": int(depot_id), "head": r[0], "tail": r[-1]},
        )

    # 中間にdepotが混ざるケースは「異常」だが、評価はdepotを除去した列で行う
    mids = [n for n in r[1:-1] if n != int(depot_id)]

    if len(mids) != len(set(mids)):
        return (False, "Duplicate(non-depot)", {"route": r})

    # PDノード以外が混ざっていないか（depot除く）
    for n in mids:
        if n not in node_to_pair:
            return (False, "NodeType(not in PD)", {"route": r, "node": n})

    # PD: 同一ルート内 & pickupが先
    pos = {n: i for i, n in enumerate(mids)}
    for p, d in pickup_to_delivery.items():
        in_p = p in pos
        in_d = d in pos
        if in_p != in_d:
            return (False, "PDSameVehicleBroken(one side missing)", {"route": r, "pair": (p, d)})
        if in_p and pos[p] > pos[d]:
            return (False, "PDOrderBroken(pickup after delivery)", {"route": r, "pair": (p, d), "pos_p": pos[p], "pos_d": pos[d]})

    # ----------------------------
    # TW / 容量シミュレーション
    # ----------------------------
    depot_c = id2c.get(int(depot_id), None)
    depot_ready = _get_num(depot_c, READY_KEYS, 0.0) if depot_c is not None else 0.0
    depot_due = _get_num(depot_c, DUE_KEYS, float("inf")) if depot_c is not None else float("inf")
    depot_service = _get_num(depot_c, SERVICE_KEYS, 0.0) if depot_c is not None else 0.0

    t = 0.0
    load = 0.0
    prev = int(depot_id)

    for k, nid in enumerate(mids, start=1):
        c = id2c.get(int(nid))
        if c is None:
            return (False, "UnknownNode", {"route": r, "node": int(nid), "pos": k})

        t += _dist(prev, int(nid))

        ready = _get_num(c, READY_KEYS, 0.0)
        due = _get_num(c, DUE_KEYS, float("inf"))
        service = _get_num(c, SERVICE_KEYS, 0.0)

        if t < ready:
            t = ready
        if t > due:
            return (False, "TimeWindow(node)", {"route": r, "node": int(nid), "pos": k, "start_service": t, "due": due})

        t += service

        dem_raw = _get_num(c, DEMAND_KEYS, 0.0)
        load += _norm_demand(int(nid), dem_raw)

        if load < -1e-9 or load - float(vehicle_capacity) > 1e-9:
            return (False, "Capacity", {"route": r, "node": int(nid), "pos": k, "load": load, "cap": int(vehicle_capacity)})

        prev = int(nid)

    # depotへ帰着（return due をチェック）
    t += _dist(prev, int(depot_id))

    if t < depot_ready:
        t = depot_ready
    if t > depot_due:
        return (False, "TimeWindow(depot_return)", {"route": r, "node": int(depot_id), "pos": len(r) - 1, "arrive": t, "due": depot_due})

    t += depot_service
    return (True, None, None)


def check_solution_feasibility(
    state: CollaborativeState,
    *,
    verbose: bool = True,
) -> bool:
    """
    最終ルート集合が実行可能（feasible）かどうかを検査する。

    チェック内容（主なもの）:
      - 全pickup/deliveryノードがちょうど1回ずつ訪問されている
      - 各会社の各車両ルートが [depot, ..., depot] 形式
      - 同一ノードの重複訪問なし（depot除く）
      - PD制約: pickup と delivery が同一車両にあり pickupが先
      - 容量制約: 0 <= load <= vehicle_capacity
      - TW制約: 到着→待機→サービス開始が due を超えない
      - 余計なノード（PDに属さないノード）を訪問していない（depot除く）

    ※ customer dict のキー名はインスタンス/実装差を吸収するため複数候補から取得する。
    """

    def _get_num(cust: Dict[str, Any], keys: List[str], default: float = 0.0) -> float:
        for k in keys:
            if k in cust and cust[k] is not None:
                try:
                    return float(cust[k])
                except Exception:
                    pass
        return float(default)

    def _get_int(cust: Dict[str, Any], keys: List[str], default: int = 0) -> int:
        return int(round(_get_num(cust, keys, default=default)))

    # --- customer辞書 ---
    id_to_customer: Dict[int, Dict[str, Any]] = {int(c["id"]): c for c in state.all_customers}

    depot_ids = set(state.depot_id_list)

    # --- PDペア（pickup->delivery）と逆引き（delivery->pickup） ---
    pickup_to_delivery: Dict[int, int] = {int(p): int(d) for p, d in state.all_PD_pairs.items()}
    delivery_to_pickup: Dict[int, int] = {int(d): int(p) for p, d in pickup_to_delivery.items()}

    required_nodes = set(pickup_to_delivery.keys()) | set(pickup_to_delivery.values())

    # --- 距離（ユークリッド→切り捨て→int） ---
    def dist(a: int, b: int) -> int:
        ca = id_to_customer[a]
        cb = id_to_customer[b]
        ax = _get_num(ca, ["x", "X"], 0.0)
        ay = _get_num(ca, ["y", "Y"], 0.0)
        bx = _get_num(cb, ["x", "X"], 0.0)
        by = _get_num(cb, ["y", "Y"], 0.0)
        return int(math.hypot(bx - ax, by - ay))  # floor by int-cast

    # --- demand（pickupで増え、deliveryで同量減る） ---
    # pickup側の需要を使う（delivery側が負値で入っていても吸収する）
    pair_demand: Dict[int, int] = {}
    for p in pickup_to_delivery.keys():
        cust = id_to_customer[p]
        dem = _get_int(cust, ["demand", "q", "Q", "quantity", "load"], default=0)
        pair_demand[p] = abs(dem)

    # --- TWキー候補 ---
    READY_KEYS = ["ready_time", "ready", "tw_start", "start_time", "r"]
    DUE_KEYS = ["due_time", "due", "tw_end", "end_time", "d"]
    SERVICE_KEYS = ["service_time", "service", "s"]

    errors: List[str] = []

    # =======================================================
    # 1) ルート全体の訪問回数チェック（node exactly-once）
    # =======================================================
    node_count: Dict[int, int] = {nid: 0 for nid in required_nodes}
    extra_nodes: List[int] = []

    for comp in state.companies:
        for r in comp.routes:
            for nid in r:
                if nid in depot_ids:
                    continue
                if nid in node_count:
                    node_count[nid] += 1
                else:
                    extra_nodes.append(nid)

    missing = [nid for nid, c in node_count.items() if c == 0]
    multi = [(nid, c) for nid, c in node_count.items() if c != 1]

    if missing:
        errors.append(f"[Coverage] 未訪問ノードが存在: count={len(missing)}, 例={missing[:10]}")
    if multi:
        # multi には missing も含まれるが見やすくするため別表示
        bad = [(nid, c) for nid, c in multi if c != 1]
        errors.append(f"[Coverage] ちょうど1回でないノードが存在: count={len(bad)}, 例={bad[:10]}")
    if extra_nodes:
        errors.append(f"[Coverage] PDに属さないノードを訪問している（depot除く）: count={len(extra_nodes)}, 例={extra_nodes[:10]}")

    # =======================================================
    # 2) 会社ごとのルート形式 + 制約チェック
    # =======================================================
    for comp_idx, comp in enumerate(state.companies):
        depot = comp.depot_id

        for v_idx, route in enumerate(comp.routes):
            # (2-1) ルート形式（depot始端終端）
            if not route or len(route) < 2:
                errors.append(f"[RouteForm] LSP{comp_idx+1} vehicle{v_idx}: ルートが空/短すぎる")
                continue

            if route[0] != depot or route[-1] != depot:
                errors.append(
                    f"[RouteForm] LSP{comp_idx+1} vehicle{v_idx}: 先頭末尾depot不一致 "
                    f"(expected {depot}, got head={route[0]}, tail={route[-1]})"
                )

            # (2-2) depot以外の重複訪問チェック
            mids = [n for n in route if n not in depot_ids]
            if len(mids) != len(set(mids)):
                # 重複しているノードを抽出
                seen = set()
                dups = []
                for n in mids:
                    if n in seen:
                        dups.append(n)
                    seen.add(n)
                errors.append(f"[Duplicate] LSP{comp_idx+1} vehicle{v_idx}: depot以外の重複訪問 例={dups[:10]}")

            # (2-3) PD制約 + TW + 容量
            picked: Dict[int, bool] = {}  # pickup_id -> picked?
            load = 0
            t = 0

            prev = route[0]
            for pos, nid in enumerate(route[1:], start=1):
                # travel
                if prev in id_to_customer and nid in id_to_customer:
                    t += dist(prev, nid)
                else:
                    errors.append(f"[Customer] LSP{comp_idx+1} vehicle{v_idx}: 未知ノードID {nid}")
                    prev = nid
                    continue

                # TW（depotにもTWが入っていることがあるので共通に処理）
                cust = id_to_customer[nid]
                ready = _get_int(cust, READY_KEYS, default=0)
                due = _get_int(cust, DUE_KEYS, default=10**9)
                service = _get_int(cust, SERVICE_KEYS, default=0)

                if t < ready:
                    t = ready  # wait allowed
                if t > due:
                    errors.append(
                        f"[TimeWindow] LSP{comp_idx+1} vehicle{v_idx}: node={nid} pos={pos} "
                        f"start_service={t} > due={due}"
                    )
                    print(f"vehicle{v_idx}.route = {route}")
                t += service

                # depotはPD/容量チェック対象外
                if nid in depot_ids:
                    prev = nid
                    continue

                # PD + 容量更新
                if nid in pickup_to_delivery:
                    # pickup
                    picked[nid] = True
                    load += pair_demand.get(nid, 0)
                elif nid in delivery_to_pickup:
                    # delivery
                    p = delivery_to_pickup[nid]
                    if not picked.get(p, False):
                        errors.append(
                            f"[PDOrder] LSP{comp_idx+1} vehicle{v_idx}: deliveryがpickupより先 "
                            f"(pickup={p}, delivery={nid})"
                        )
                    load -= pair_demand.get(p, 0)
                else:
                    # PDに属さないノード（上でextra扱いにしているが、ここでも明示）
                    errors.append(f"[NodeType] LSP{comp_idx+1} vehicle{v_idx}: PDに属さないnode={nid}")

                if load < 0 or load > state.vehicle_capacity:
                    errors.append(
                        f"[Capacity] LSP{comp_idx+1} vehicle{v_idx}: node={nid} pos={pos} "
                        f"load={load} (cap={state.vehicle_capacity})"
                    )

                prev = nid

            # (2-4) pickupしたのにdeliveryが同一車両に無い、を検出
            #  route内にdeliveryが出てきたかどうかは load が戻っているかでも分かるが、明示的にチェック
            visited_set = set(route)
            for p, d in pickup_to_delivery.items():
                if p in visited_set and d not in visited_set:
                    errors.append(
                        f"[PDSameVehicle] LSP{comp_idx+1} vehicle{v_idx}: pickupのみ存在 (p={p}, d={d})"
                    )
                if d in visited_set and p not in visited_set:
                    errors.append(
                        f"[PDSameVehicle] LSP{comp_idx+1} vehicle{v_idx}: deliveryのみ存在 (p={p}, d={d})"
                    )

    # =======================================================
    # 3) 結果表示
    # =======================================================
    ok = (len(errors) == 0)

    if verbose:
        if ok:
            print("✅ Feasibility check: OK（全ノード訪問・PD/TW/容量・形式チェックを通過）")
        else:
            print("❌ Feasibility check: FAILED")
            for e in errors[:50]:
                print("  - " + e)
            if len(errors) > 50:
                print(f"  ... and {len(errors) - 50} more errors")

    return ok

