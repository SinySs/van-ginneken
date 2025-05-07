import json
from dataclasses import dataclass
from typing import List, Dict, Optional, Any
from enum import Enum, auto

@dataclass
class Module:
    kind: str
    name: str
    C: float
    R: float
    DELAY: float

@dataclass
class Technology:
    unit_wire_resistance: float
    unit_wire_capacitance: float
    unit_wire_resistance_comment: str
    unit_wire_capacitance_comment: str

@dataclass
class Config:
    modules: List[Module]
    technology: Technology

def read_config(file_path: str) -> Config:
    with open(file_path, 'r') as file:
        data = json.load(file)
    
    module_arr = data["module"]
    module_obj = module_arr[0]
    input_obj = module_obj["input"][0]
    
    module = Module(
        kind="buf1x",  
        name=module_obj["name"],
        R=input_obj["R"],
        C=input_obj["C"],
        DELAY=input_obj["intrinsic_delay"]
    )
    
    tech_obj = data["technology"]
    technology = Technology(
        unit_wire_resistance=tech_obj["unit_wire_resistance"],
        unit_wire_capacitance=tech_obj["unit_wire_capacitance"],
        unit_wire_resistance_comment=tech_obj.get("unit_wire_resistance_comment0", ""),
        unit_wire_capacitance_comment=tech_obj.get("unit_wire_capacitance_comment0", "")
    )
    
    return Config(modules=[module], technology=technology)


class NodeKind:
    STEINER = "s"
    TERMINAL = "t"
    BUFFER = "b"


class Point:
    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y
    
    def __repr__(self):
        return f"({self.x}, {self.y})"


class Node:
    def __init__(self, node_id: int, kind: str, name: str, point: Point, 
                 capacitance: float = 0.0, rat: float = 0.0):
        self.id = node_id
        self.kind = kind
        self.name = name
        self.point = point
        self.capacitance = capacitance
        self.rat = rat
    
    @property
    def is_buffer(self) -> bool:
        return self.kind == NodeKind.BUFFER
    
    @property
    def is_terminal(self) -> bool:
        return self.kind == NodeKind.TERMINAL
    
    def __repr__(self):
        params = [
            f"id={self.id}",
            f"kind='{self.kind}'",
            f"name='{self.name}'",
            f"at={self.point}",
        ]
        if self.is_terminal:
            params.extend([
                f"C={self.capacitance}",
                f"RAT={self.rat}"
            ])
        return f"Node({', '.join(params)})"


class Edge:
    def __init__(self, edge_id: int, source: int, target: int, segments: List[Point]):
        self.id = edge_id
        self.source = source
        self.target = target
        self.segments = segments
    
    def __repr__(self):
        return (f"Edge({self.id}: {self.source}→{self.target}, "
                f"points={self.segments})")


class Graph:
    def __init__(self):
        self._nodes: Dict[int, Node] = {}
        self._edges: Dict[int, Edge] = {}
        self._root_id: Optional[int] = None
        self.technology: Optional[Technology] = None  # Добавляем хранение технологии
        self.modules: Dict[str, Module] = {}  # Словарь модулей по их типу

    def set_technology(self, technology: Technology):
        self.technology = technology

    def add_module(self, module: Module):
        self.modules[module.kind] = module
    
    @classmethod
    def read_from_file(cls, file_path: str) -> 'Graph':
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        graph = cls()
        
        for node_data in data["node"]:
            node = Node(
                node_id=node_data["id"],
                kind=node_data["type"],
                name=node_data["name"],
                point=Point(node_data["x"], node_data["y"]),
                capacitance=node_data.get("capacitance", 0.0),
                rat=node_data.get("rat", 0.0)
            )
            graph.add_node(node)
        
        for edge_data in data["edge"]:
            edge = Edge(
                edge_id=edge_data["id"],
                source=edge_data["vertices"][0],
                target=edge_data["vertices"][1],
                segments=[Point(p[0], p[1]) for p in edge_data["segments"]]
            )
            graph.add_edge(edge)
        
        return graph
    
    def add_node(self, node: Node):
        self._nodes[node.id] = node
        if node.is_buffer:
            self._root_id = node.id
    
    def add_edge(self, edge: Edge):
        self._edges[edge.id] = edge
    
    @property
    def nodes(self) -> Dict[int, Node]:
        return self._nodes
    
    @property
    def edges(self) -> Dict[int, Edge]:
        return self._edges
    
    @property
    def root(self) -> Optional[Node]:
        return self._nodes.get(self._root_id)
    
    def __repr__(self):
        return (f"Graph({len(self._nodes)} nodes, {len(self._edges)} edges, "
                f"root={self._root_id})")


class PointKind(Enum):
    SIMPLE = auto()
    BUFFER = auto()

@dataclass(frozen=True)
class PointRecord:
    kind: PointKind
    start: Point
    p: Point

    def __eq__(self, other):
        if not isinstance(other, PointRecord):
            return False
        return (self.kind == other.kind and 
                self.start == other.start and 
                self.p == other.p)

    def __lt__(self, other):
        return manhattan_distance(self.start, self.p) < manhattan_distance(other.start, other.p)

    def __hash__(self):
        return hash((self.kind, self.start, self.p))


class SolutionInserter:
    def __init__(self, graph):
        self.graph = graph
        self._next_node_id = self._find_max_node_id() + 1 if graph.nodes else 0

    def _find_max_node_id(self) -> int:
        return max(self.graph.nodes.keys()) if self.graph.nodes else -1

    def _get_next_node_id(self) -> int:
        new_id = self._next_node_id
        self._next_node_id += 1
        return new_id

    def _split_points(self, points: List['Point'], solutions: List['Candidate']) -> List[List['Point']]:
        if len(points) <= 1 or not solutions:
            return [points]

        start = points[0]
        records_set = set()

        for candidate in solutions:
            if candidate.has_buffer:
                record = PointRecord(PointKind.BUFFER, start, candidate.position)
                records_set.add(record)

        for point in points[1:-1]:
            record = PointRecord(PointKind.SIMPLE, start, point)
            if record not in records_set:
                records_set.add(record)

        records = [PointRecord(PointKind.SIMPLE, start, points[0])]
        records.extend(sorted(records_set))
        records.append(PointRecord(PointKind.SIMPLE, start, points[-1]))

        result = []
        prev_idx = 0
        for i, record in enumerate(records):
            if record.kind == PointKind.BUFFER:
                segment = [r.p for r in records[prev_idx:i+1]]
                if len(segment) > 1:
                    result.append(segment)
                prev_idx = i

        last_segment = [r.p for r in records[prev_idx:]]
        if len(last_segment) > 1:
            result.append(last_segment)

        return result

    def insert_solution(self, solution: List['Candidate']) -> None:
        if not solution:
            return

        grouped = {}
        for candidate in solution:
            if candidate.has_buffer:
                if candidate.edge_id not in grouped:
                    grouped[candidate.edge_id] = []
                grouped[candidate.edge_id].append(candidate)

        for edge_id, candidates in grouped.items():
            if edge_id not in self.graph.edges:
                continue

            edge = self.graph.edges[edge_id]
            start_point = edge.segments[0]
            
            candidates_sorted = sorted(
                candidates,
                key=lambda c: manhattan_distance(start_point, c.position)
            )

            splitted_segments = self._split_points(edge.segments, candidates_sorted)

            first_node = edge.source
            last_node = edge.target
            buffer_nodes = [first_node]
            
            for candidate in candidates_sorted:
                buffer_node = Node(
                    node_id=self._get_next_node_id(),
                    kind=NodeKind.BUFFER,
                    name="buf1x",
                    point=candidate.position,
                    capacitance=candidate.capacity,
                    rat=candidate.rat
                )
                self.graph.add_node(buffer_node)
                buffer_nodes.append(buffer_node.id)
            
            buffer_nodes.append(last_node)

            del self.graph.edges[edge_id]

            for i in range(len(buffer_nodes)-1):
                segment = splitted_segments[i] if i < len(splitted_segments) else [
                    self.graph.nodes[buffer_nodes[i]].point,
                    self.graph.nodes[buffer_nodes[i+1]].point
                ]
                new_edge_id = max(self.graph.edges.keys()) + 1 if self.graph.edges else 0
                new_edge = Edge(
                    edge_id=new_edge_id,
                    source=buffer_nodes[i],
                    target=buffer_nodes[i+1],
                    segments=segment
                )
                self.graph.edges[new_edge_id] = new_edge

def manhattan_distance(p1: 'Point', p2: 'Point') -> float:
    return abs(p2.x - p1.x) + abs(p2.y - p1.y)


def save_graph_to_json(graph, filename: str) -> None:
    nodes_data = []
    for node in graph.nodes.values():
        node_data = {
            "id": node.id,
            "name": node.name,
            "type": node.kind,
            "x": node.point.x,
            "y": node.point.y
        }
        
        if node.is_terminal:
            node_data["capacitance"] = node.capacitance
            node_data["rat"] = node.rat
            
        nodes_data.append(node_data)

    edges_data = []
    for edge in graph.edges.values():
        edge_data = {
            "id": edge.id,
            "segments": [[p.x, p.y] for p in edge.segments],
            "vertices": [edge.source, edge.target]
        }
        edges_data.append(edge_data)

    graph_data = {
        "node": nodes_data,
        "edge": edges_data
    }

    with open(filename, 'w') as f:
        json.dump(graph_data, f, indent=4)

