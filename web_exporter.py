import os
import json
import re
import glob
import shutil
import logging

logger = logging.getLogger(__name__)

def export_vrp_state(customers, routes, PD_pairs, step_index, case_index=None,
                     depot_id_list=None, vehicle_num_list=None, instance_name=None,
                     output_root="web_data"):
    """
    VRP状態をReactアプリ用にJSON形式で保存

    - instance_name が指定されていればそれをディレクトリ名として使う。
      例: instance_name="LC1_2_2_LC1_2_7" -> web_data/LC1_2_2_LC1_2_7/step_0.json
    - instance_name が None の場合は case_{case_index} を使用する。
    - depot_id_list, vehicle_num_list は未指定時に自動推定する。
    """

    
    if instance_name:
        folder_name = instance_name
    else:
        # 保険: case_index が None の場合はタイムスタンプを使う（安全策）
        if case_index is None:
            folder_name = f"case_{int(__import__('time').time())}"
        else:
            folder_name = f"case_{case_index}"

    output_dir = os.path.join(output_root, folder_name)
    # --- 初回のみフォルダをクリーンにする ---
    if step_index == 0:
        if os.path.exists(output_dir):
            logger.info(f"⚠️ 初回ステップのため既存フォルダを削除します: {output_dir}")
            shutil.rmtree(output_dir)
        os.makedirs(output_dir, exist_ok=True)
    else:
        os.makedirs(output_dir, exist_ok=True)

    if depot_id_list is None:
        depot_id_list = [c["id"] for c in customers if c.get("demand", 0) == 0]
    if vehicle_num_list is None:
        vehicle_num_list = [len(routes)]

    data = {
        "customers": customers,
        "routes": routes,
        "PD_pairs": PD_pairs,
        "depot_id_list": depot_id_list,
        "vehicle_num_list": vehicle_num_list,
        "step_index": step_index,
        "instance_name": folder_name
    }

    json_path = os.path.join(output_dir, f"step_{step_index}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    logger.info(f"✅ VRP状態を出力しました: {json_path}")
    return json_path  # 返しておくとテストやログに便利


def generate_index_json(instance_name: str,
                        output_root: str = "web_data",
                        target_root: str = "vrp-viewer/public/vrp_data"):
    """
    目的:
      直前に export_vrp_state で生成した「特定インスタンス」のJSON群だけを
      vrp-viewer/public/vrp_data に反映し、index.json を部分更新する。

    処理:
      1) web_data/<instance_name>/ を確認
      2) vrp-viewer/public/vrp_data/<instance_name>/ が既にあれば削除してから再コピー
      3) vrp-viewer/public/vrp_data/index.json を読み込み、同名エントリを削除
      4) コピー先の <instance_name> 内の JSON を列挙し、{"name": ..., "steps": [...]} を作成
      5) 既存 cases に新エントリを追加して index.json を保存

    引数:
      instance_name: 今回更新するケース名（例: "LC1_2_2_LC1_2_7"）
      output_root:   Python 側の出力ルート（web_data）
      target_root:   React 側の参照ルート（vrp-viewer/public/vrp_data）

    戻り値:
      index.json のパス
    """
    if not instance_name or not isinstance(instance_name, str):
        raise ValueError("generate_index_json: 'instance_name' は必須です。")

    src_case_dir = os.path.join(output_root, instance_name)
    if not os.path.isdir(src_case_dir):
        raise FileNotFoundError(f"ソースが見つかりません: {src_case_dir}")

    os.makedirs(target_root, exist_ok=True)

    # 1) 先に対象インスタンスのコピー先をキレイにする
    dst_case_dir = os.path.join(target_root, instance_name)
    if os.path.exists(dst_case_dir):
        logger.info(f"⚠️ 既存インスタンスを削除します: {dst_case_dir}")
        shutil.rmtree(dst_case_dir)

    # 2) 当該インスタンスだけコピー
    shutil.copytree(src_case_dir, dst_case_dir)
    logger.info(f"📁 コピー完了: {src_case_dir} → {dst_case_dir}")

    # 3) 既存 index.json を読み込み（なければ空テンプレート）
    index_path = os.path.join(target_root, "index.json")
    if os.path.exists(index_path):
        try:
            with open(index_path, "r", encoding="utf-8") as f:
                index_data = json.load(f)
            if not isinstance(index_data, dict) or "cases" not in index_data or not isinstance(index_data["cases"], list):
                # 想定外形式のときはリセット
                index_data = {"cases": []}
        except Exception:
            # 壊れていた場合もリセット
            index_data = {"cases": []}
    else:
        index_data = {"cases": []}

    # 4) index.cases から同名インスタンスを削除
    cases = [c for c in index_data.get("cases", []) if not (isinstance(c, dict) and c.get("name") == instance_name)]

    # 5) コピー先のファイルから steps を作成（数値でソート）
    step_paths = glob.glob(os.path.join(dst_case_dir, "step_*.json"))

    def step_num(fname: str) -> int:
        m = re.search(r"step_(\d+)\.json$", os.path.basename(fname))
        return int(m.group(1)) if m else 10**9   # マッチしない場合は末尾へ

    steps = [os.path.basename(p) for p in sorted(step_paths, key=step_num)]
    
    # 6) 新しいエントリを追加
    cases.append({"name": instance_name, "steps": steps})

    # 7) index.json を保存（上書き）
    index_data = {"cases": cases}
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(index_data, f, indent=2, ensure_ascii=False)

    logger.info(f"✅ index.json を更新しました → {index_path}")
    return index_path
