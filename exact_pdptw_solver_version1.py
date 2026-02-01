from __future__ import annotations
from typing import Dict, List, Tuple, Optional, TypeAlias
from collections import defaultdict, Counter
import itertools
import math
import time


def build_dist_mat(coords: List[Tuple[int, int]]) -> List[List[int]]:
    """coords[index]=(x,y) に対する距離行列（float）を作る"""
    n = len(coords)
    dist_mat = [[0] * n for _ in range(n)]
    for i in range(n):
        xi, yi = coords[i]
        row = dist_mat[i]
        for j in range(n):
            xj, yj = coords[j]
            row[j] = int(math.floor(math.hypot(xj - xi, yj - yi)))
    return dist_mat


def build_move_time(service: List[int], dist_mat: List[List[int]]) -> List[List[int]]:
    """
    move_time[i][j] = service[i] + dist_mat[i][j]
    （あなたの定義：出発ノードiでのサービス後に移動する所要時間）
    """
    n = len(service)
    move_time = [[0] * n for _ in range(n)]
    for i in range(n):
        si = float(service[i])
        row_i = move_time[i]
        dist_row = dist_mat[i]
        for j in range(n):
            row_i[j] = si + dist_row[j]
    return move_time


def route_cost_idx(route_idx: List[int], dist_mat: List[List[int]]) -> int:
    """
    ルート（index列）の総移動距離を返す。
    route_idx: [start_idx, ..., end_idx]
    dist_mat: 事前計算した距離行列（float）
    """
    if route_idx is None or len(route_idx) < 2:
        return 0
    s = 0
    for a, b in zip(route_idx, route_idx[1:]):
        s += dist_mat[a][b]
    return s


# ============================================================
# solve_1vehicle_pdptw_exact_idxの探索フェーズで枝刈りを行うための
# 上界を簡易的(貪欲)に生成するための関数
# 解が得られない場合もある
# ============================================================
def build_any_feasible_route_greedy_idx(
    visit_idxs: List[int],
    idx_to_bit: dict[int, int],
    delivery_to_pickup: dict[int, int],
    demand: List[int],
    ready: List[int],
    due: List[int],
    service: List[int],
    dist_mat: List[List[float]],
    start_depot_idx: int,
    end_depot_idx: int,
    vehicle_capacity: int,
) -> Optional[Tuple[List[int], float]]:
    """
    1台分の feasible ルートを貪欲に1本作る。
    返り値: (route_idxs, total_dist) or None
    """
    # 状態
    mask = 0
    last = start_depot_idx
    load = 0
    t = max(0.0, float(ready[start_depot_idx]))
    if t > due[start_depot_idx]:
        return None

    route = [start_depot_idx]
    full_mask = (1 << len(visit_idxs)) - 1
    total_dist = 0.0

    # precedence check（deliveryはpickup済みが必要）
    def precedence_ok(m: int, nxt: int) -> bool:
        if nxt in delivery_to_pickup:
            p = delivery_to_pickup[nxt]
            pb = idx_to_bit[p]  # ※ visit_idxs 由来なので p は必ず入っている前提
            return ((m >> pb) & 1) == 1
        return True

    while mask != full_mask:
        candidates = []
        for nxt in visit_idxs:
            b = idx_to_bit[nxt]
            if (mask >> b) & 1:
                continue
            if not precedence_ok(mask, nxt):
                continue

            new_load = load + demand[nxt]
            if new_load < 0 or new_load > vehicle_capacity:
                continue

            travel = dist_mat[last][nxt]
            arrive = t + float(service[last]) + travel
            new_t = max(arrive, float(ready[nxt]))
            if new_t > due[nxt]:
                continue

            # スコア：近さ + 早い締切（好きに変えてOK）
            # 例）「行って戻るまでの見込み」+「締切を少し意識」
            score = travel + dist_mat[nxt][end_depot_idx] + 0.001 * float(due[nxt])
            candidates.append((score, nxt, new_load, new_t, travel))

        if not candidates:
            return None  # 貪欲では詰んだ

        candidates.sort(key=lambda x: x[0])
        _, nxt, new_load, new_t, travel = candidates[0]

        # 遷移確定
        b = idx_to_bit[nxt]
        mask |= (1 << b)
        route.append(nxt)
        total_dist += travel
        last = nxt
        load = new_load
        t = new_t

    # endへ戻れるかチェック
    travel_back = dist_mat[last][end_depot_idx]
    arrive_end = t + float(service[last]) + travel_back
    end_t = max(arrive_end, float(ready[end_depot_idx]))
    if end_t > due[end_depot_idx]:
        return None

    route.append(end_depot_idx)
    total_dist += travel_back
    return route, total_dist


# ============================================================
# 1台の PDPTW を「厳密に」解く（ラベリング/DP）
# - 目的: 距離最小
# - 制約: 容量 / タイムウィンドウ / PD順序（pickup -> delivery）
# - 入力は Index 基準
# ============================================================
def solve_1vehicle_pdptw_exact_idx(
    sub_customers: List[Dict],
    pd_pairs: List[Tuple[int, int]],      # indexペア (pickup_idx, delivery_idx)
    start_depot_idx: int,                 # index
    end_depot_idx: int,                   # index
    vehicle_capacity: int,
    dist_mat: List[List[float]],
    move_time: List[List[float]],
    *,
    debug: bool = False,
    debug_name: str = "1veh",
) -> Optional[List[int]]:
    """
    1車両の PDPTW を厳密に解いて、ルート（index列）を返す。
    返り値: [start_idx, ..., end_idx]（存在しなければ None）

    追加枝刈り（安全）:
      - 未訪問ノードのうち due 最小ノード u について、
          earliest(u) = max(time_sofar + move_time[last][u], ready[u])
        が due[u] を超えるなら、そのラベルから先は必ず infeasible → prune
      - 前提: 距離が三角不等式を満たす(ユークリッド等) & service>=0
    """

    t_start = time.perf_counter()

    # ========= デバッグ用カウンタ =========
    cnt = Counter()
    # 主要カウンタキー例:
    #   states_scanned, labels_scanned, transitions_tried
    #   prune_lb1, prune_due_min, prune_cap, prune_tw, prune_lb2
    #   insert_called, drop_dominated, removed_by_new, inserted
    #   ub_updates, end_feasible_found
    due_min_fail_node = Counter()  # due-min pruneで詰んだノード index の頻度
    due_min_fail_pair = Counter()  # (pair_i, is_pickup) の頻度

    # ========= 前処理 =========
    demand  = [int(c["demand"])  for c in sub_customers]
    ready   = [float(c["ready"]) for c in sub_customers]
    due     = [float(c["due"])   for c in sub_customers]
    service = [float(c["service"]) for c in sub_customers]

    # pd_pairs が空ならデポ直行
    if not pd_pairs:
        t0 = max(0.0, ready[start_depot_idx])
        if t0 > due[start_depot_idx]:
            if debug:
                print(f"[DBG][{debug_name}] empty-pairs: start depot infeasible")
            return None
        travel = dist_mat[start_depot_idx][end_depot_idx]
        arrive_end = t0 + service[start_depot_idx] + travel
        t_end = max(arrive_end, ready[end_depot_idx])
        if t_end > due[end_depot_idx]:
            if debug:
                print(f"[DBG][{debug_name}] empty-pairs: end depot infeasible")
            return None
        if debug:
            dt = time.perf_counter() - t_start
            print(f"[DBG][{debug_name}] empty-pairs solved in {dt:.3f}s")
        return [start_depot_idx, end_depot_idx]

    pairs: List[Tuple[int, int]] = list(pd_pairs)
    m = len(pairs)

    # ========= 3進マスク準備 =========
    pow3 = [1] * (m + 1)
    for i in range(m):
        pow3[i + 1] = pow3[i] * 3

    full_mask = 0
    for i in range(m):
        full_mask += 2 * pow3[i]

    def get_state(mask: int, i: int) -> int:
        return (mask // pow3[i]) % 3  # 0,1,2

    def inc_state(mask: int, i: int) -> int:
        return mask + pow3[i]

    # ========= ★ due最小未訪問ノード prune 用の準備 =========
    # entry = (due_value, node_idx, pair_index, is_pickup)
    due_order_nodes: List[Tuple[float, int, int, bool]] = []
    for i, (p, d) in enumerate(pairs):
        due_order_nodes.append((due[p], p, i, True))
        due_order_nodes.append((due[d], d, i, False))
    due_order_nodes.sort(key=lambda x: x[0])

    def is_node_visited(mask3: int, pair_i: int, is_pickup: bool) -> bool:
        st = get_state(mask3, pair_i)
        if is_pickup:
            return st >= 1
        else:
            return st == 2

    def due_min_prune(mask3: int, last: int, time_sofar: float) -> bool:
        """
        dueが最小の未訪問ノード u に、次に直行しても間に合わないなら prune。
        """
        cnt["due_min_checks"] += 1
        for _due_u, u, pair_i, is_p in due_order_nodes:
            if is_node_visited(mask3, pair_i, is_p):
                continue
            earliest = max(time_sofar + move_time[last][u], ready[u])
            if earliest > due[u]:
                cnt["prune_due_min"] += 1
                due_min_fail_node[u] += 1
                due_min_fail_pair[(pair_i, is_p)] += 1
                return True
            return False  # due最小の未訪問がOKなら、チェックはここで終わり
        return False      # 全訪問済み（ここは普通は full_mask 近辺）

    # ========= ラベルDP =========
    Label: TypeAlias = Tuple[float, float, Optional[Tuple[int, int, int, int]]]
    table: Dict[Tuple[int, int, int], List[Label]] = defaultdict(list)
    mask_to_keys: Dict[int, List[Tuple[int, int, int]]] = defaultdict(list)
    key_seen: set[Tuple[int, int, int]] = set()

    # 初期状態
    t0 = max(0.0, ready[start_depot_idx])
    if t0 > due[start_depot_idx]:
        if debug:
            print(f"[DBG][{debug_name}] start depot infeasible")
        return None

    init_key = (0, start_depot_idx, 0)
    table[init_key].append((0.0, t0, None))
    mask_to_keys[0].append(init_key)
    key_seen.add(init_key)

    def insert_label(key: Tuple[int, int, int], cand: Label) -> None:
        """
        同一 key 内で (dist,time,load) 支配関係で刈りつつ挿入。
        """
        cnt["insert_called"] += 1
        dist_c, time_c, _ = cand
        _m3, _last, _load = key
        lst = table[key]

        # 既存が cand を支配 → 捨て
        for dist_e, time_e, prev_e in lst:
            # load は key に固定なので比較不要
            if dist_e <= dist_c and time_e <= time_c:
                cnt["drop_dominated"] += 1
                return

        # cand が既存を支配 → 既存を除去
        new_lst: List[Label] = []
        removed = 0
        for dist_e, time_e, prev_e in lst:
            if dist_c <= dist_e and time_c <= time_e:
                removed += 1
                continue
            new_lst.append((dist_e, time_e, prev_e))
        if removed:
            cnt["removed_by_new"] += removed

        new_lst.append(cand)
        table[key] = new_lst
        cnt["inserted"] += 1

        if key not in key_seen:
            key_seen.add(key)
            mask_to_keys[key[0]].append(key)

    # ========= 上界（そのまま：inf） =========
    best_total = float("inf")

    # ========= 遷移 =========
    pending_masks = [0]
    pending_seen = {0}
    ptr = 0

    # 進捗ログ
    next_report = 0
    report_step = 2000  # m3処理数の目安ではなく、pending_masks(ptr)の進行で見る

    while ptr < len(pending_masks):
        mask = pending_masks[ptr]
        ptr += 1
        cnt["m3_popped"] += 1

        if debug and ptr >= next_report:
            next_report += report_step
            labels_now = sum(len(v) for v in table.values())
            keys_now = len(key_seen)
            dt = time.perf_counter() - t_start
            print(
                f"[DBG][{debug_name}] ptr={ptr} pending={len(pending_masks)} "
                f"seen_m3={len(pending_seen)} keys={keys_now} labels={labels_now} "
                f"prune_due_min={cnt['prune_due_min']} prune_tw={cnt['prune_tw']} dt={dt:.2f}s"
            )

        keys = mask_to_keys.get(mask, [])
        for (m3, last, load) in keys:
            cnt["states_scanned"] += 1
            for dist_sofar, time_sofar, _prev in list(table[(m3, last, load)]):
                cnt["labels_scanned"] += 1

                # 枝刈り1：最低でも end に戻る必要がある（距離LB）
                if dist_sofar + dist_mat[last][end_depot_idx] > best_total:
                    cnt["prune_lb1"] += 1
                    continue

                # ★追加枝刈り：due最小未訪問ノードに直行しても間に合わないなら prune
                if due_min_prune(m3, last, time_sofar):
                    continue

                # 次に動かせるペアを列挙
                for i, (p, d) in enumerate(pairs):
                    st = get_state(m3, i)
                    if st == 2:
                        continue

                    nxt = p if st == 0 else d
                    cnt["transitions_tried"] += 1

                    # 容量
                    new_load = load + demand[nxt]
                    if new_load < 0 or new_load > vehicle_capacity:
                        cnt["prune_cap"] += 1
                        continue

                    # 時刻（TW）
                    travel = dist_mat[last][nxt]
                    arrive = time_sofar + move_time[last][nxt]
                    new_time = max(arrive, ready[nxt])
                    if new_time > due[nxt]:
                        cnt["prune_tw"] += 1
                        continue

                    new_dist = dist_sofar + travel
                    new_mask = inc_state(m3, i)

                    # 枝刈り2：nxtに行っても最低でも end に戻る必要がある
                    if new_dist + dist_mat[nxt][end_depot_idx] > best_total:
                        cnt["prune_lb2"] += 1
                        continue

                    new_key = (new_mask, nxt, new_load)
                    insert_label(new_key, (new_dist, new_time, (m3, last, load, nxt)))

                    if new_mask not in pending_seen:
                        pending_seen.add(new_mask)
                        pending_masks.append(new_mask)

                    # 全完了に到達した瞬間、end に戻れるなら上界更新
                    if new_mask == full_mask:
                        cnt["end_state_reached"] += 1
                        back = dist_mat[nxt][end_depot_idx]
                        arrive_end = new_time + service[nxt] + back
                        t_end = max(arrive_end, ready[end_depot_idx])
                        if t_end <= due[end_depot_idx]:
                            cnt["end_feasible_found"] += 1
                            total = new_dist + back
                            if total < best_total:
                                best_total = total
                                cnt["ub_updates"] += 1

    # ========= 終了（endに戻れるものの中で最良を選ぶ） =========
    best_total_final: Optional[float] = None
    best_state = None
    best_label = None

    full_keys = mask_to_keys.get(full_mask, [])
    for (m3, last, load) in full_keys:
        for dist_sofar, time_sofar, prev in table[(m3, last, load)]:
            travel_back = dist_mat[last][end_depot_idx]
            arrive_end = time_sofar + service[last] + travel_back
            end_time = max(arrive_end, ready[end_depot_idx])
            if end_time > due[end_depot_idx]:
                continue

            total_dist = dist_sofar + travel_back
            if best_total_final is None or total_dist < best_total_final:
                best_total_final = total_dist
                best_state = (m3, last, load)
                best_label = (dist_sofar, time_sofar, prev)

    # ======= デバッグサマリ =======
    if debug:
        dt = time.perf_counter() - t_start
        total_labels = sum(len(v) for v in table.values())
        total_keys = len(key_seen)
        print(f"[DBG][{debug_name}] ========== SUMMARY ==========")
        print(f"[DBG][{debug_name}] time = {dt:.3f}s  m={m}")
        print(f"[DBG][{debug_name}] seen_m3={len(pending_seen)} popped_m3={cnt['m3_popped']}")
        print(f"[DBG][{debug_name}] keys={total_keys} labels={total_labels}")
        print(f"[DBG][{debug_name}] states_scanned={cnt['states_scanned']} labels_scanned={cnt['labels_scanned']}")
        print(f"[DBG][{debug_name}] transitions_tried={cnt['transitions_tried']}")
        print(f"[DBG][{debug_name}] prune: lb1={cnt['prune_lb1']} lb2={cnt['prune_lb2']} cap={cnt['prune_cap']} tw={cnt['prune_tw']}")
        print(f"[DBG][{debug_name}] due-min: checks={cnt['due_min_checks']} pruned={cnt['prune_due_min']}")
        print(f"[DBG][{debug_name}] insert: called={cnt['insert_called']} inserted={cnt['inserted']} drop_dominated={cnt['drop_dominated']} removed_by_new={cnt['removed_by_new']}")
        print(f"[DBG][{debug_name}] end_state_reached={cnt['end_state_reached']} end_feasible_found={cnt['end_feasible_found']} ub_updates={cnt['ub_updates']}")
        print(f"[DBG][{debug_name}] best_total={'inf' if best_total==float('inf') else f'{best_total:.6f}'}")
        if cnt["prune_due_min"] > 0:
            print(f"[DBG][{debug_name}] due-min fail node top10 (node_idx:count, due):")
            for u, c in due_min_fail_node.most_common(10):
                print(f"    u={u}  count={c}  due={due[u]}")
            print(f"[DBG][{debug_name}] due-min fail pair top10 ((pair_i,is_pickup):count):")
            for k, c in due_min_fail_pair.most_common(10):
                pair_i, is_p = k
                p, d = pairs[pair_i]
                which = "P" if is_p else "D"
                node = p if is_p else d
                print(f"    pair={pair_i}{which} node={node} due={due[node]} count={c}")
        print(f"[DBG][{debug_name}] =============================")

    if best_state is None or best_label is None:
        return None

    # ========= 経路復元（FIX版：整合する前ラベルを探索して辿る）=========
    EPS = 1e-6

    # best_label の dist/time も必要
    dist_cur, time_cur, prev = best_label

    route_rev = [end_depot_idx]
    m3, last, load = best_state
    route_rev.append(last)

    # prev は (pm, plast, pload, visited) を想定
    while prev is not None:
        pm, plast, pload, visited = prev

        # visited は基本 last と一致するはず（念のため）
        # assert visited == last

        # predecessor label を table から探す
        pred_label = None
        for dist_p, time_p, prev_p in table[(pm, plast, pload)]:
            # 距離整合
            if abs((dist_p + dist_mat[plast][visited]) - dist_cur) > EPS:
                continue

            # 時刻整合（待機込み）
            arrive = time_p + move_time[plast][visited]
            time_expected = max(arrive, ready[visited])
            if abs(time_expected - time_cur) > EPS:
                continue

            # これが「本当に使われた predecessor」
            pred_label = (dist_p, time_p, prev_p)
            break

        if pred_label is None:
            # ここに来るのは、復元不能（prevの持ち方が不足）か、
            # 浮動小数/丸めの不一致がある場合
            if debug:
                print(f"[DBG][{debug_name}] reconstruction failed at node={visited}")
            return None  # もしくは raise RuntimeError にして原因を顕在化させる

        # ひとつ戻る
        dist_p, time_p, prev_p = pred_label
        dist_cur, time_cur, prev = dist_p, time_p, prev_p

        # predecessor の last は plast
        route_rev.append(plast)
        m3, last, load = pm, plast, pload

    route = list(reversed(route_rev))

    # start/end は「整形で誤魔化す」のではなく、基本ここで成立しているのが望ましい
    # 必要なら assert にして早期発見
    if route[0] != start_depot_idx or route[-1] != end_depot_idx:
        if debug:
            print(f"[DBG][{debug_name}] bad start/end in reconstructed route: {route[:5]} ... {route[-5:]}")
        # ここで無理に整形すると順序が壊れる可能性があるので、返さない方が安全
        return None

    return route




def solve_exact_2vehicle_vrp(
    sub_customers: List[Dict],
    sub_PD_pairs: List[Tuple[int, int]],
    start_depots: List[int],
    end_depots: List[int],
    vehicle_capacity: int,
) -> List[List[int]]:
    """
    2車両ぶんの PDPTW（容量・タイムウィンドウ・PD順序）を「厳密に」解くための関数。

    入力（重要：すべて "ID" 基準）
    - sub_customers:
        ノード辞書のリスト。各辞書は少なくとも以下を含む想定:
        {
          'id': int, 'x': float, 'y': float,
          'demand': int,
          'ready': int, 'due': int,
          'service': int,
          ...（pickup_index/delivery_indexがあっても本関数ではsub_PD_pairsを使う）
        }
    - sub_PD_pairs:
        (pickup_id, delivery_id) のリスト。ここに入っている数値も "ID"。
    - start_depots / end_depots:
        長さ2のリスト。各車両の出発/到着デポの "ID"。
        例: [200, 200]
    - vehicle_capacity:
        車両容量（2台とも同じ前提）

    出力（ID基準）
    - routes: 長さ2の2次元リスト。
        routes[v] は [start_id, ..., end_id] で、訪問順序が入る。
        例: [[0, 113, ..., 0], [0, 45, ..., 0]]

    目的関数
    - Min( 2台の合計距離（ユークリッド） ）
    """

    # =========================================================
    # VERIFY SWITCHES
    # =========================================================
    VERIFY = False                 # まとめてON/OFF
    VERIFY_VEH_FEAS = True         # (A) veh_feas と solver の一致検証
    VERIFY_TRUE_OPT = True         # (B) 2車両の真の最適と best_total 一致検証
    # =========================================================

    # -----------------------------
    # 1) バリデーション
    # -----------------------------
    required_keys = {"id", "x", "y", "demand", "ready", "due", "service"}
    for c in sub_customers:
        missing = required_keys - set(c.keys())
        if missing:
            raise ValueError(f"sub_customers に必要キーが不足: missing={missing}, customer={c}")
        
    if len(start_depots) != 2 or len(end_depots) != 2:
        raise ValueError("start_depots/end_depots は長さ2（2車両）である必要があります。")
    
    ids = [c["id"] for c in sub_customers]
    if len(ids) != len(set(ids)):
        raise ValueError("sub_customers 内に 'id' の重複があります。")
    
    id2cust = {c["id"]: c for c in sub_customers}
    for dep_id in start_depots + end_depots:
        if dep_id not in id2cust:
            raise ValueError(f"デポID {dep_id} が sub_customers に存在しません。")
        
    for p_id, d_id in sub_PD_pairs:
        if p_id not in id2cust or d_id not in id2cust:
            raise ValueError(f"sub_PD_pairs に sub_customers 外のIDが含まれています: ({p_id},{d_id})")
        
    # -----------------------------------
    # 2) 内部計算用に index 化（ID→index）
    # -----------------------------------
    id2idx = {c["id"]: i for i, c in enumerate(sub_customers)}
    idx2id = [c["id"] for c in sub_customers]
    start_idx = [id2idx[dep_id] for dep_id in start_depots]
    end_idx = [id2idx[dep_id] for dep_id in end_depots]
    pd_idx_pairs: List[Tuple[int, int]] = [(id2idx[p], id2idx[d]) for p, d in sub_PD_pairs]
    
    # --------------------------------
    # 3) 特殊ケース：PDペアが存在しない
    # --------------------------------
    m = len(pd_idx_pairs)
    if m == 0:
        return [[start_depots[0], end_depots[0]], [start_depots[1], end_depots[1]]]
    
    # -----------------------------
    # 4) 各種必要データの作成
    # -----------------------------
    coords = [(float(c["x"]), float(c["y"])) for c in sub_customers]
    service = [float(c["service"]) for c in sub_customers]
    dist_mat = build_dist_mat(coords)
    move_time = build_move_time(service, dist_mat)

    FULL = (1 << m) - 1
    half = m / 2.0
    same_depot = (start_idx[0] == start_idx[1] and end_idx[0] == end_idx[1])

    # -----------------------------
    # 5) 車両ごとの各種キャッシュ
    # -----------------------------
    # veh_feas: 「車両vがmaskを単独で回れるか？」= solver一致対象
    vehicle_feasible: List[Dict[int, bool]] = [{}, {}]
    vehicle_cost: List[Dict[int, float]] = [{}, {}]
    vehicle_route: List[Dict[int, List[int]]] = [{}, {}]

    # useful: 「2車両探索でこのmask割当が意味あるか？」（comp_pruneでFalseになり得る）
    useful: List[Dict[int, bool]] = [{}, {}]

    # 既知 infeasible の「極小集合（antichain）」だけを保持し、
    #      部分集合チェックで上位集合を枝刈りするための集合
    infeasible_min: List[set[int]] = [set(), set()]

    # subset_prune の根拠となった下位集合のmaskを記録
    subset_pruned_bad: List[Dict[int, int]] = [{}, {}]  # mask -> bad

    # -----------------------------
    # 6) ユーティリティ
    # -----------------------------
    def build_pairs_from_mask(mask: int) -> List[Tuple[int, int]]:
        return [pd_idx_pairs[k] for k in range(m) if (mask >> k) & 1]

    def mask_to_pairs_id(mask: int) -> List[Tuple[int, int]]:
        return [(idx2id[p], idx2id[d]) for (p, d) in build_pairs_from_mask(mask)]

    def has_infeasible_subset(v: int, mask: int) -> int | None:
        """含んでいる infeasible subset を1つ返す（なければNone）"""
        for inf in infeasible_min[v]:
            if (mask & inf) == inf:
                return inf
        return None

    def register_infeasible(v: int, mask: int) -> None:
        """solverでNoneだったときだけ呼ぶ（重要：comp_pruneでは絶対呼ばない）"""
        if mask == 0:
            return
        for inf in infeasible_min[v]:
            if (mask & inf) == inf:
                return
        to_remove = {inf for inf in infeasible_min[v] if (inf & mask) == mask}
        if to_remove:
            infeasible_min[v].difference_update(to_remove)
        infeasible_min[v].add(mask)

    def solve_vehicle(v: int, mask: int) -> None:
        """veh_feas/veh_cost/veh_route を埋める（未計算なら計算）"""
        if mask in vehicle_feasible[v]:
            return
        route = solve_1vehicle_pdptw_exact_idx(
            sub_customers=sub_customers,
            pd_pairs=build_pairs_from_mask(mask),
            start_depot_idx=start_idx[v],
            end_depot_idx=end_idx[v],
            vehicle_capacity=vehicle_capacity,
            dist_mat=dist_mat,
            move_time=move_time,
            debug=False,
        )
        if route is None:
            vehicle_feasible[v][mask] = False
            useful[v][mask] = False  # 単独で無理なら当然2車両でも候補にならない
            register_infeasible(v, mask)
        else:
            c = route_cost_idx(route, dist_mat)
            vehicle_feasible[v][mask] = True
            useful[v][mask] = True
            vehicle_route[v][mask] = route
            vehicle_cost[v][mask] = c

    # ----------------------------------------
    # 7-1) 探索本体：PD割り当て数=0の場合の検証
    # ----------------------------------------
    # mask=0
    for v in (0, 1):
        solve_vehicle(v, 0)
        if not vehicle_feasible[v][0]:
            raise RuntimeError(f"Empty assignment (mask=0) infeasible for vehicle {v}")

    # -----------------------------------------
    # 7-2) 探索本体：PD割り当て数>=1の場合の検証 
    # -----------------------------------------
    # ここでは「PDペアの割当」を bitmask（長さm）で表す。
    #   - mask の bit=1 になっているPDペア集合を “車両0” が担当
    #   - comp = FULL ^ mask（補集合）を “車両1” が担当
    #
    # ■検証順序（重要）
    #   PDペア数 k の昇順（k=1,2,...,m）で、k個のペアを選ぶ全組合せを列挙し mask を作る。
    #   この順序により、任意の mask の「真部分集合」は必ず先に評価済みになる。
    #
    # ■枝刈り戦略
    #   1) subset prune（単調性に基づく枝刈り）
    #      すでに “infeasible と確定した集合” を含む（=部分集合として持つ）mask は、
    #      今回の問題設定では上位集合も必ず infeasible とみなせるためスキップできる。
    #      具体的には、infeasible_min（infeasibleの極小集合＝antichain）だけを保持し、
    #      ∃inf∈infeasible_min s.t. (mask & inf)==inf なら枝刈りする。
    #
    #   2) comp prune（2車両性に基づく枝刈り）
    #      k > m/2 の領域では補集合 comp のペア数は < m/2 なので、comp は既に評価済みのはず。
    #      もし comp が 1台で infeasible なら、mask が 1台で feasible でも 2車両解になり得ないため、
    #      mask の評価（solve_1vehicle呼び出し）自体を省略できる。
    #
    # ■妥当性（なぜ落としてよいか）
    #   - subset prune は「infeasible subset ⇒ infeasible superset」という単調性に基づく。
    #     （この性質が成り立つ前提のもとで、厳密性を損なわず探索数だけ減らす。）
    #   - comp prune は「2車両解には mask と comp の両方が1台で feasible である必要がある」という
    #     必要条件に基づくため、厳密性を損なわない。
    #
    # なお、best_total の更新は k >= m/2 の段階で行う（mask と comp の両方が揃う領域のため）。
    best_total = float("inf")
    best_r0_idx = None
    best_r1_idx = None

    for k in range(1, m + 1):
        #print(f"PDペア数 = {k}ペアを探索中")
        # 変更: 車両ごとに feasible mask を保持
        feasible_masks_k: List[List[int]] = [[], []]

        for comb in itertools.combinations(range(m), k):
            mask = 0
            for bitpos in comb:
                mask |= (1 << bitpos)
            comp = FULL ^ mask

            if same_depot:
                v = 0
                bad = has_infeasible_subset(v, mask)
                if bad is not None:
                    vehicle_feasible[0][mask] = False
                    vehicle_feasible[1][mask] = False
                    useful[0][mask] = False
                    useful[1][mask] = False
                    subset_pruned_bad[0][mask] = bad
                    subset_pruned_bad[1][mask] = bad
                    continue

                if k > half:
                    solve_vehicle(0, comp)
                    if not vehicle_feasible[0][comp]:
                        useful[0][mask] = False
                        useful[1][mask] = False
                        continue

                solve_vehicle(0, mask)
                vehicle_feasible[1][mask] = vehicle_feasible[0][mask]

                if vehicle_feasible[0].get(mask, False):
                    feasible_masks_k[0].append(mask)

            else:
                for v in (0, 1):
                    other = 1 - v

                    bad = has_infeasible_subset(v, mask)
                    if bad is not None:
                        vehicle_feasible[v][mask] = False
                        useful[v][mask] = False
                        subset_pruned_bad[v][mask] = bad
                        continue

                    if k > half:
                        solve_vehicle(other, comp)
                        if not vehicle_feasible[other][comp]:
                            useful[v][mask] = False
                            continue

                    solve_vehicle(v, mask)
                    if vehicle_feasible[v].get(mask, False):
                        feasible_masks_k[v].append(mask)

        # total 更新（k>=half）
        if k >= half:
            if same_depot:
                for mask in feasible_masks_k[0]:
                    comp = FULL ^ mask
                    solve_vehicle(0, comp)
                    if vehicle_feasible[0].get(mask, False) and vehicle_feasible[0].get(comp, False):
                        total = vehicle_cost[0][mask] + vehicle_cost[0][comp]
                        if total < best_total:
                            best_total = total
                            best_r0_idx = vehicle_route[0][mask]
                            best_r1_idx = vehicle_route[0][comp]
            else:
                for v in (0, 1):
                    other = 1 - v
                    for mask in feasible_masks_k[v]:
                        comp = FULL ^ mask
                        solve_vehicle(other, comp)
                        if vehicle_feasible[v].get(mask, False) and vehicle_feasible[other].get(comp, False):
                            total = vehicle_cost[v][mask] + vehicle_cost[other][comp]
                            if total < best_total:
                                best_total = total
                                best_r0_idx = vehicle_route[v][mask]
                                best_r1_idx = vehicle_route[other][comp]


    if best_r0_idx is None or best_r1_idx is None:
        raise RuntimeError("2車両の厳密解が見つかりません（入力が不可能制約の可能性）。")


    # =========================================================
    # VERIFY BLOCK (optional)  -- works for same_depot True/False
    # =========================================================
    if VERIFY:

        def _solve_mask_for_vehicle(v: int, mask: int) -> Optional[List[int]]:
            """車両vで mask を1台で回す（検証用の再計算）"""
            return solve_1vehicle_pdptw_exact_idx(
                sub_customers=sub_customers,
                pd_pairs=build_pairs_from_mask(mask),
                start_depot_idx=start_idx[v],
                end_depot_idx=end_idx[v],
                vehicle_capacity=vehicle_capacity,
                dist_mat=dist_mat,
                move_time=move_time,
            )

        # ---- (A) vehicle-feasible の完全検証 ----
        if VERIFY_VEH_FEAS:
            # solver_feasible[v][mask] を全maskで用意
            solver_feasible: List[Dict[int, bool]] = [{}, {}]
            for v in (0, 1):
                for mask in range(FULL + 1):
                    solver_feasible[v][mask] = (_solve_mask_for_vehicle(v, mask) is not None)

            # 探索中に comp_prune 等で未計算のmaskが残り得るので、強制的に埋めてから比較
            for v in (0, 1):
                for mask in range(FULL + 1):
                    solve_vehicle(v, mask)  # vehicle_feasible[v][mask] を必ず埋める

            mismatches = []
            for v in (0, 1):
                for mask in range(FULL + 1):
                    cache_val = bool(vehicle_feasible[v][mask])
                    sol_val = bool(solver_feasible[v][mask])
                    if cache_val != sol_val:
                        mismatches.append((v, mask, cache_val, sol_val))

            if mismatches:
                print(f"[VERIFY] vehicle_feasible mismatches = {len(mismatches)}")
                for (v, mask, cache_val, sol_val) in mismatches[:50]:
                    pairs_id = mask_to_pairs_id(mask)
                    print("--------------------------------------------------")
                    print(f"vehicle={v} mask={mask} bin={bin(mask)} k={mask.bit_count()}")
                    print(f"pairs(ID)={pairs_id}")
                    print(f"vehicle_feasible(cache)={cache_val} solver={sol_val}")
                    # subset_prune由来なら根拠も出す（記録している場合のみ）
                    if mask in subset_pruned_bad[v]:
                        bad = subset_pruned_bad[v][mask]
                        print(
                            f"subset_pruned_bad={bad} bin(bad)={bin(bad)} "
                            f"bad_pairs(ID)={mask_to_pairs_id(bad)}"
                        )
                raise RuntimeError(
                    f"[VERIFY FAILED] vehicle_feasible と solver の判定が一致しませんでした: mismatch={len(mismatches)}"
                )

            print("[VERIFY] vehicle_feasible matches solver for ALL masks (both vehicles). OK.")

        # ---- (B) 2車両の真の最適（総当たり） ----
        if VERIFY_TRUE_OPT:
            # (A)をOFFにして(B)だけ回す可能性があるので solver_feasible はここでも用意
            solver_feasible: List[Dict[int, bool]] = [{}, {}]
            for v in (0, 1):
                for mask in range(FULL + 1):
                    solver_feasible[v][mask] = (_solve_mask_for_vehicle(v, mask) is not None)

            # 車両ごとにルート/コストを必要時にキャッシュ
            solver_route_cache: List[Dict[int, List[int]]] = [{}, {}]
            solver_cost_cache: List[Dict[int, float]] = [{}, {}]

            def _get_route_cost(v: int, mask: int) -> Tuple[List[int], float]:
                if mask in solver_route_cache[v]:
                    return solver_route_cache[v][mask], solver_cost_cache[v][mask]
                r = _solve_mask_for_vehicle(v, mask)
                if r is None:
                    raise RuntimeError("internal: asked cost for infeasible mask")
                c = route_cost_idx(r, dist_mat)
                solver_route_cache[v][mask] = r
                solver_cost_cache[v][mask] = c
                return r, c

            best_total_true = float("inf")
            best_pair_true: Optional[Tuple[List[int], List[int]]] = None

            # 向き固定：車両0が mask、車両1が comp（mask全列挙で逆向きも必ず評価される）
            for mask in range(FULL + 1):
                comp = FULL ^ mask
                if not solver_feasible[0][mask]:
                    continue
                if not solver_feasible[1][comp]:
                    continue
                r0, c0 = _get_route_cost(0, mask)
                r1, c1 = _get_route_cost(1, comp)
                total = c0 + c1
                if total < best_total_true:
                    best_total_true = total
                    best_pair_true = (r0, r1)

            if best_pair_true is None:
                raise RuntimeError(
                    "[VERIFY TRUE-OPT] 全割当総当たりでも feasible な2車両解が見つかりません。入力か solver に問題があります。"
                )

            if best_total != best_total_true:
                r0_true, r1_true = best_pair_true
                r0_true_id = [idx2id[i] for i in r0_true]
                r1_true_id = [idx2id[i] for i in r1_true]
                print("[VERIFY TRUE-OPT] best_total mismatch!")
                print(f"  strategy best_total = {best_total}")
                print(f"  true best_total     = {best_total_true}")
                print(f"  true r0(ID)={r0_true_id}")
                print(f"  true r1(ID)={r1_true_id}")
                raise RuntimeError(
                    "[VERIFY TRUE-OPT FAILED] 本戦略のbest_totalが真の最適と一致しません。"
                )

            print("[VERIFY TRUE-OPT] strategy best_total matches true optimum. OK.")


    # -----------------------------
    # 8) index → IDに直して出力
    # -----------------------------
    best_r0_id = [idx2id[i] for i in best_r0_idx]
    best_r1_id = [idx2id[i] for i in best_r1_idx]
    return [best_r0_id, best_r1_id]



def debug_run_exact_2vehicle_vrp_on_two_routes(
    all_customers: List[Dict],
    all_PD_pairs: Dict[int, int],  # {pickup_id: delivery_id}
    vehicle_capacity: int = 200,
) -> List[List[int]]:
    # あなたが指定した2本のルート（ID列）
    r0 = [0, 19, 61, 62, 64, 68, 97, 100, 109, 173, 179, 192, 206, 209]
    r1 =  [209, 209]

    # ルートに登場するID集合（デポ含む）
    node_ids = set(r0) | set(r1)

    # customers 抽出
    id2cust_all = {c["id"]: c for c in all_customers}
    missing = [nid for nid in sorted(node_ids) if nid not in id2cust_all]
    if missing:
        raise ValueError(f"all_customers に存在しない node_id があります: {missing}")

    sub_customers = [id2cust_all[nid] for nid in sorted(node_ids)]

    # PD抽出：pickup,delivery の両方が node_ids にあるものだけ
    sub_PD_pairs: List[Tuple[int, int]] = []
    for p, d in all_PD_pairs.items():
        if p in node_ids and d in node_ids:
            sub_PD_pairs.append((p, d))

    # デポ（両車両とも 0→0 を想定）
    start_depots = [r0[0], r0[-1]]
    end_depots = [r1[0], r1[-1]]

    print("=== DEBUG INPUT for solve_exact_2vehicle_vrp ===")
    print(f"route0={r0}")
    print(f"route1={r1}")
    print(f"node_ids(sorted)={sorted(node_ids)}")
    print(f"sub_customers len={len(sub_customers)}")
    print(f"sub_PD_pairs={sub_PD_pairs}")
    print(f"start_depots={start_depots}, end_depots={end_depots}, cap={vehicle_capacity}")
    print("===============================================")

    # 直接実行
    new_routes = solve_exact_2vehicle_vrp(
        sub_customers=sub_customers,
        sub_PD_pairs=sub_PD_pairs,   # list[(pickup_id, delivery_id)] すべてID
        start_depots=start_depots,   # ID
        end_depots=end_depots,       # ID
        vehicle_capacity=vehicle_capacity,
    )

    print("=== OUTPUT from solve_exact_2vehicle_vrp ===")
    print(new_routes)
    return new_routes