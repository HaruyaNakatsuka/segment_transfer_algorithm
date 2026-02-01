from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Set, Tuple

Customer = Dict[str, Any]          # parse_lilim200 の返す customer dict を想定
PDPairDict = Dict[int, int]        # {pickup_id: delivery_id}
Route = List[int]                 # [node_id, ...]（先頭末尾デポ）
Routes = List[Route]


@dataclass
class CompanyState:
    """
    会社（LSP）単位の状態。
    """
    idx: int
    depot_id: int
    depot_coord: Tuple[float, float]
    num_vehicles: int

    # この会社に属するノードID範囲（[id_min, id_max)）
    # ※ parse_lilim200 の id_offset 方式に合わせて main 側で設定する
    id_min: int
    id_max: int

    routes: Routes = field(default_factory=list)

    # 初期解の総経路長（int）
    initial_total_route_length: int = 0
    # 直前の総経路長
    previous_total_route_length: int = 0
    # 暫定（最新）総経路長（int）
    current_total_route_length: int = 0

    @property
    def label(self) -> str:
        return f"LSP {self.idx + 1}"

    def node_ids(self) -> Set[int]:
        s: Set[int] = set()
        for r in self.routes:
            s.update(r)
        return s

    def validate(self) -> None:
        if len(self.routes) != 0 and len(self.routes) != self.num_vehicles:
            raise ValueError(
                f"{self.label}: routes length mismatch "
                f"(expected {self.num_vehicles}, got {len(self.routes)})"
            )


@dataclass
class CollaborativeState:
    """
    collaborative PDPTW の全体状態。
    """
    all_customers: List[Customer]
    all_PD_pairs: PDPairDict
    vehicle_capacity: int
    companies: List[CompanyState]
    instance_name: str = ""
    case_index: int = 0

    # 初期解の総経路長（int）
    initial_total_route_length: int = 0
    # 直前の総経路長
    previous_total_route_length: int = 0
    # 暫定（最新）総経路長（int）
    current_total_route_length: int = 0

    @property
    def depot_id_list(self) -> List[int]:
        return [c.depot_id for c in self.companies]

    @property
    def depot_coords(self) -> List[Tuple[float, float]]:
        return [c.depot_coord for c in self.companies]

    @property
    def vehicle_num_list(self) -> List[int]:
        return [c.num_vehicles for c in self.companies]

    def get_all_routes(self) -> Routes:
        """
        全会社, 全車両の経路をフラットな2次元リストとして取得できる
        """
        return [r for c in self.companies for r in c.routes]

    def set_all_routes(self, routes: Routes) -> None:
        """フラットな全車両ルート配列を companies に分配して保持する。"""
        idx = 0
        for c in self.companies:
            c.routes = routes[idx: idx + c.num_vehicles]
            idx += c.num_vehicles
            c.validate()
        if idx != len(routes):
            raise ValueError(
                f"routes length mismatch: consumed={idx}, total={len(routes)})"
            )

    def company_customers(self, comp: CompanyState) -> List[Customer]:
        """その会社のID範囲に属する customers を抽出（デポを含む）"""
        return [c for c in self.all_customers if comp.id_min <= c.get("id") < comp.id_max]

    def company_pd_pairs(self, comp: CompanyState) -> PDPairDict:
        """pickup/delivery のどちらかが会社のID範囲に含まれるペアを抽出"""
        # ※ gat.py と同じ条件（旧 initialize_individual_vrps と一致）に合わせる
        sub_ids = {c["id"] for c in self.company_customers(comp)}
        return {
            p: d for p, d in self.all_PD_pairs.items()
            if p in sub_ids or d in sub_ids
        }



