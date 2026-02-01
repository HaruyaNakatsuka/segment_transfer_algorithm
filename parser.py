def parse_lilim200(filepath, x_offset=0, y_offset=0, id_offset=0, time_offset=0):
    customers = []
    P_to_D = {}

    with open(filepath, 'r') as f:
        lines = f.readlines()

    # 1行目から車両数・容量を読み取る
    header_parts = lines[0].strip().split()
    num_vehicles = int(header_parts[0])
    vehicle_capacity = int(header_parts[1])

    # 2行目以降のノード情報を処理
    for line in lines[1:]:
        parts = line.strip().split()
        if len(parts) < 9:
            continue

        cust_id = int(parts[0]) + id_offset
        x = float(parts[1]) + x_offset
        y = float(parts[2]) + y_offset
        demand = int(parts[3])
        ready = int(parts[4]) + time_offset
        due = int(parts[5]) + time_offset
        service = int(parts[6])
        if int(parts[7]) > 0:
            pickup_index = int(parts[7]) + id_offset
        else:
            pickup_index = int(parts[7])
        if int(parts[8]) > 0:
            delivery_index = int(parts[8]) + id_offset
        else:
            delivery_index = int(parts[8])

        node = {
            'id': cust_id,
            'x': x,
            'y': y,
            'demand': demand,
            'ready': ready,
            'due': due,
            'service': service,
            'pickup_index': pickup_index,
            'delivery_index': delivery_index
        }

        customers.append(node)

        if demand > 0 and delivery_index > 0:
            P_to_D[cust_id] = delivery_index

    return {
        'customers': customers,
        'PD_pairs': P_to_D,
        'num_vehicles': num_vehicles,
        'vehicle_capacity': vehicle_capacity,
        'depot_id': customers[0]['id'],
        'depot_coord': (customers[0]['x'], customers[0]['y'])
    }


def customers_to_lilim200_text(sub_customers, sub_PD_pairs, n_vehicles, vehicle_capacity):
    """
    sub_customers / sub_PD_pairs を LILIM200 風のテキスト行に再フォーマット。
      1行目: "<num_vehicles> <vehicle_capacity>"
      2行目以降: "<id> <x> <y> <demand> <ready> <due> <service> <pickup_index> <delivery_index>"

    注意:
    - 会社内サブセットなので、相方ノードが sub_customers にいない場合は pickup/delivery を 0 に落とす。
    - 座標は元ファイルが整数のことが多いので、整数に近ければ int で、そうでなければ小数で出力。
    - depot（最初の行にしたいノード）が sub_customers[0] 前提なら、先頭に来るように並べ替える。
      不明な場合は ID 昇順にします（必要なら depot を先頭にするロジックを追加してください）。
    """
    # 出力: 先頭行
    lines = [f"{n_vehicles} {int(vehicle_capacity)} 1"]

    # 会社サブセットに含まれる ID 集合
    include_ids = {c["id"] for c in sub_customers}

    # 並び順（ここでは ID 昇順。必要なら depot を最初にする処理を追加）
    sub_customers_sorted = sorted(sub_customers, key=lambda c: c["id"])

    def maybe_int(v):
        # ほぼ整数なら整数で、そうでなければそのまま
        if abs(v - round(v)) < 1e-9:
            return str(int(round(v)))
        return f"{v:.6f}".rstrip("0").rstrip(".")

    for c in sub_customers_sorted:
        cid   = c["id"]
        x     = c["x"]
        y     = c["y"]
        dem   = int(c["demand"])
        ready = int(c["ready"])
        due   = int(c["due"])
        svc   = int(c["service"])

        # 相方がサブセットにいなければ 0 に落とす
        pidx  = int(c.get("pickup_index", 0))
        didx  = int(c.get("delivery_index", 0))
        if pidx not in include_ids:
            pidx = 0
        if didx not in include_ids:
            didx = 0

        # 1レコード行を作成
        row = f"{cid} {maybe_int(x)} {maybe_int(y)} {dem} {ready} {due} {svc} {pidx} {didx}"
        lines.append(row)

    return "\n".join(lines)
