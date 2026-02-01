from __future__ import annotations
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
import math
from utils import route_cost


def create_distance_matrix(customers):
    size = len(customers)
    matrix = [[0] * size for _ in range(size)]
    for i in range(size):
        for j in range(size):
            matrix[i][j] = int(math.hypot(customers[i]['x'] - customers[j]['x'], customers[i]['y'] - customers[j]['y']))
    return matrix


def solve_vrp_flexible(
    customers,
    initial_routes,
    PD_pairs,
    num_vehicles,
    vehicle_capacity,
    start_depots,
    end_depots,
    use_capacity: bool,
    use_time: bool,
    use_pickup_delivery: bool,
    Warm_Start: bool,
    *,
    debug: bool = False
):
    # 距離行列（customers の順序に対応）を構築
    distance_matrix = create_distance_matrix(customers)

    # 顧客ID -> customers 配列インデックス
    id_to_index = {c["id"]: i for i, c in enumerate(customers)}

    # 各車両の start/end デポ（顧客ID）を customers インデックスに変換
    starts = [id_to_index[depot_id] for depot_id in start_depots]
    ends = [id_to_index[depot_id] for depot_id in end_depots]

    # OR-Tools Routing モデル（車両ごとに start/end を与える構成）
    manager = pywrapcp.RoutingIndexManager(len(customers), num_vehicles, starts, ends)
    routing = pywrapcp.RoutingModel(manager)

    # アークコスト（距離）
    def distance_callback(from_idx, to_idx):
        from_node = manager.IndexToNode(from_idx)  # customers インデックス
        to_node = manager.IndexToNode(to_idx)
        return distance_matrix[from_node][to_node]

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    # debug 用：ID列の総距離を計算（距離行列と同じ定義で）
    def _route_distance_by_ids(route_ids):
        total = 0
        for a, b in zip(route_ids, route_ids[1:]):
            total += distance_matrix[id_to_index[a]][id_to_index[b]]
        return total

    # 容量制約（demand をそのまま使用：delivery が負の場合も想定）
    if use_capacity:
        demands = [c["demand"] for c in customers]

        def demand_callback(from_idx):
            node = manager.IndexToNode(from_idx)
            return demands[node]

        demand_cb = routing.RegisterUnaryTransitCallback(demand_callback)
        routing.AddDimensionWithVehicleCapacity(
            demand_cb,
            0,  # slack
            [vehicle_capacity] * num_vehicles,
            True,  # Start の累積値を 0 に固定
            "Capacity",
        )

    # 時間制約（移動 + 出発ノードでのサービス時間）
    time_dim = None
    if use_time:
        time_windows = [(c["ready"], c["due"]) for c in customers]
        service_times = [c["service"] for c in customers]

        def time_callback(from_idx, to_idx):
            from_node = manager.IndexToNode(from_idx)
            to_node = manager.IndexToNode(to_idx)
            return distance_matrix[from_node][to_node] + service_times[from_node]

        time_cb = routing.RegisterTransitCallback(time_callback)

        # Time 次元：待機（slack）を許し、Start の累積値は固定しない（TWで制御）
        routing.AddDimension(time_cb, 99999, 99999, False, "Time")
        time_dim = routing.GetDimensionOrDie("Time")

        # TW 付与：
        # - 通常ノードは NodeToIndex で一意に routing index を取得できる
        # - start/end は車両ごとに routing index が異なるため Start(v)/End(v) に対して付与する
        start_end_nodes = set(starts) | set(ends)

        for node in range(len(customers)):
            if node in start_end_nodes:
                continue
            idx = manager.NodeToIndex(node)
            time_dim.CumulVar(idx).SetRange(*time_windows[node])

        for v in range(num_vehicles):
            s_idx = routing.Start(v)
            e_idx = routing.End(v)
            s_node = manager.IndexToNode(s_idx)
            e_node = manager.IndexToNode(e_idx)
            time_dim.CumulVar(s_idx).SetRange(*time_windows[s_node])
            time_dim.CumulVar(e_idx).SetRange(*time_windows[e_node])

    # Pickup & Delivery 制約（同一車両 + pickup が先）
    # 重要：use_time=True のときは順序は Time 次元で書けるので、
    #       「Distance 次元（＋余計な目的関数）」を作らない。
    distance_dimension = None
    if use_pickup_delivery:
        if not (use_time and time_dim is not None):
            # Time 次元がない場合のみ、順序制約の代替として Distance 次元を作る
            routing.AddDimension(
                transit_callback_index,
                0,
                10000,
                True,  # Start cumul 0 固定
                "Distance",
            )
            distance_dimension = routing.GetDimensionOrDie("Distance")

        for pickup_id, delivery_id in PD_pairs:
            if pickup_id not in id_to_index or delivery_id not in id_to_index:
                if debug:
                    print(f"[ORTools][WARN] Invalid ID pair (not in sub-customers): {pickup_id}, {delivery_id}")
                continue

            pickup_idx = manager.NodeToIndex(id_to_index[pickup_id])
            delivery_idx = manager.NodeToIndex(id_to_index[delivery_id])

            routing.AddPickupAndDelivery(pickup_idx, delivery_idx)
            routing.solver().Add(routing.VehicleVar(pickup_idx) == routing.VehicleVar(delivery_idx))

            # 順序：Time 次元があれば Time、なければ Distance（代替）
            if use_time and time_dim is not None:
                routing.solver().Add(time_dim.CumulVar(pickup_idx) <= time_dim.CumulVar(delivery_idx))
            else:
                routing.solver().Add(distance_dimension.CumulVar(pickup_idx) <= distance_dimension.CumulVar(delivery_idx))

    # 探索パラメータ
    search_params = pywrapcp.DefaultRoutingSearchParameters()
    if debug:
        search_params.log_search = True

    # debug: WarmStart入力の距離統計（総距離 / 最大ルート長）
    if debug:
        try:
            init_full_routes = []
            for v in range(num_vehicles):
                s = start_depots[v]
                e = end_depots[v]
                mid = list(initial_routes[v])  # initial_routes は「デポ除去済み」想定
                init_full_routes.append([s] + mid + [e])

            init_dists = [_route_distance_by_ids(r) for r in init_full_routes]
            print(f"[ORTools][DEBUG] WarmStart input: sum_dist={sum(init_dists)}, max_dist={max(init_dists) if init_dists else 0}")
        except Exception as ex:
            print(f"[ORTools][DEBUG] failed to evaluate WarmStart input routes: {ex}")

    if Warm_Start:
        # 初期解を routes として与える（start/end は Routing が持つので mid 部分だけ渡す）
        # ※与えるノードが customers 内であることは呼び出し側で保証されている前提
        search_params.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.AUTOMATIC

        initial_routes_local = []
        for route in initial_routes:
            # route は顧客ID列
            initial_routes_local.append([id_to_index[node_id] for node_id in route])

        routing.CloseModelWithParameters(search_params)

        initial_solution = routing.ReadAssignmentFromRoutes(initial_routes_local, True)
        if initial_solution is None:
            if debug:
                print("[ORTools][DEBUG] ReadAssignmentFromRoutes returned None -> fallback to SolveWithParameters")
            solution = routing.SolveWithParameters(search_params)
        else:
            solution = routing.SolveFromAssignmentWithParameters(initial_solution, search_params)
    else:
        search_params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.AUTOMATIC
        search_params.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.AUTOMATIC
        solution = routing.SolveWithParameters(search_params)

    if not solution:
        return None

    # 解の復元（routing index -> customers index -> 顧客ID）
    result = []
    for vehicle_id in range(num_vehicles):
        idx = routing.Start(vehicle_id)
        route = []
        while not routing.IsEnd(idx):
            route.append(customers[manager.IndexToNode(idx)]["id"])
            idx = solution.Value(routing.NextVar(idx))
        route.append(customers[manager.IndexToNode(idx)]["id"])
        result.append(route)

    # debug: 出力解の距離統計（総距離 / 最大ルート長）と ObjectiveValue（取れれば）
    if debug:
        try:
            out_dists = [_route_distance_by_ids(r) for r in result]
            msg = f"[ORTools][DEBUG] Solver output: sum_dist={sum(out_dists)}, max_dist={max(out_dists) if out_dists else 0}"
            if hasattr(solution, "ObjectiveValue"):
                msg += f", obj={solution.ObjectiveValue()}"
            print(msg)

            if Warm_Start:
                # 目安として「総距離」が増えたかだけ通知（目的関数が総距離と一致していれば基本増えないはず）
                try:
                    init_sum = sum(init_dists)
                    out_sum = sum(out_dists)
                    if out_sum > init_sum:
                        print("[ORTools][DEBUG] NOTE: sum_dist increased vs WarmStart input (objective mismatch or other issue suspected).")
                except Exception:
                    pass
        except Exception as ex:
            print(f"[ORTools][DEBUG] failed to evaluate output routes: {ex}")

    return result
