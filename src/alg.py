import math
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from src.parse_json import Graph, Node, Edge, Point, NodeKind

@dataclass
class Candidate:
    capacity: float
    rat: float
    position: Point
    edge_id: int
    has_buffer: bool = False

    def __str__(self):
        return (f"Candidate(capacity={self.capacity:.4f}, rat={self.rat:.4f}, "
                f"pos=({self.position.x}, {self.position.y}), "
                f"edge_id={self.edge_id}, has_buffer={self.has_buffer})")

class BufferInserter:
    def __init__(self, graph: Graph, step: int = 1):
        self.graph = graph
        self.step = step
        self.visited: Dict[int, List[List[Candidate]]] = {}
        self.invalid_id = -1

    def manhattan_distance(self, p1: Point, p2: Point) -> float:
        return abs(p1.x - p2.x) + abs(p1.y - p2.y)

    def split_edge(self, edge: Edge, step: int) -> List[Point]:
        points = edge.segments
        candidates = []

        for i in range(len(points)-1, 0, -1):
            start = points[i]
            end = points[i-1]
            
            if start.x != end.x:
                step = self.step if end.x > start.x else -self.step
                steps = int((end.x - start.x) / step)
                for cnt in range(1, steps):
                    x = start.x + cnt * step
                    candidates.append(Point(x, start.y))
            else:
                step = self.step if end.y > start.y else -self.step
                steps = int((end.y - start.y) / step)
                for cnt in range(1, steps):
                    y = start.y + cnt * step
                    candidates.append(Point(start.x, y))

        if points:
            candidates.append(points[0])
        return candidates

    def calculate_wire_delay(self, length: float, C: float) -> float:
        R = self.graph.technology.unit_wire_resistance
        Cw = self.graph.technology.unit_wire_capacitance
        return (R * Cw * length**2)/2 + R * length * C

    def calculate_buffer_delay(self, C: float) -> float:
        buffer = self.graph.modules["buf1x"]
        return buffer.DELAY + buffer.R * C
    
    def insert_buffer(self, solution: List[Candidate]) -> None:
        if not solution or "buf1x" not in self.graph.modules:
            return

        last = solution[-1]
        buffer_module = self.graph.modules["buf1x"]

        buffer_delay = buffer_module.DELAY + buffer_module.R * last.capacity
        last.rat -= buffer_delay
        last.capacity = buffer_module.C
        last.has_buffer = True

    def insert_wire(self, solution: List[Candidate], length: float, position: Point, edge_id: int) -> None:
        if not solution or not self.graph.technology:
            return

        last = solution[-1]
        tech = self.graph.technology

        wire_delay = (tech.unit_wire_resistance * tech.unit_wire_capacitance * (length ** 2)) / 2
        wire_delay += tech.unit_wire_resistance * length * last.capacity

        solution.append(Candidate(
            capacity=last.capacity + tech.unit_wire_capacitance * length,
            rat=last.rat - wire_delay,
            position=position,
            edge_id=edge_id,
            has_buffer=False
        ))

    def elimination(self, solutions: List[List[Candidate]]) -> List[List[Candidate]]:
        if not solutions:
            return []

        def is_dominated(dominant: List[Candidate], candidate: List[Candidate]) -> bool:
            last_dom = dominant[-1]
            last_cand = candidate[-1]
            
            return (last_dom.rat >= last_cand.rat and 
                    last_dom.capacity <= last_cand.capacity)

        indexed_solutions = list(enumerate(solutions))

        dominated_indices = set()
        
        for i, (idx_a, sol_a) in enumerate(indexed_solutions[:-1]):
            for j, (idx_b, sol_b) in enumerate(indexed_solutions[i+1:], start=i+1):
                if is_dominated(sol_a, sol_b):
                    dominated_indices.add(idx_b)
                elif is_dominated(sol_b, sol_a):
                    dominated_indices.add(idx_a)

        result = [
            solution 
            for idx, solution in indexed_solutions 
            if idx not in dominated_indices
        ]

        return result


    def merge_pair(self, lhs: List[List[Candidate]], 
                          rhs: List[List[Candidate]], 
                          position: Point) -> List[List[Candidate]]:
        merged = []
        for lhs_sol in lhs:
            for rhs_sol in rhs:
                new_sol = lhs_sol.copy() + rhs_sol.copy()
                lhs_last = lhs_sol[-1]
                rhs_last = rhs_sol[-1]
                
                new_sol.append(Candidate(
                    capacity=lhs_last.capacity + rhs_last.capacity,
                    rat=min(lhs_last.rat, rhs_last.rat),
                    position=position,
                    edge_id=self.invalid_id,
                    has_buffer=False
                ))
                merged.append(new_sol)
        return merged

    def merge_solutions(self, children_solutions: List[List[List[Candidate]]], 
                       node: Node) -> List[List[Candidate]]:
        if node.kind == NodeKind.TERMINAL:
            return [[Candidate(
                capacity=node.capacitance,
                rat=node.rat,
                position=node.point,
                edge_id=self.invalid_id,
                has_buffer=False
            )]]

        if not children_solutions:
            return []

        if len(children_solutions) == 1:
            return children_solutions[0]

        if len(children_solutions) == 2:
            return self.merge_pair(children_solutions[0], 
                                          children_solutions[1], 
                                          node.point)

        solutions = children_solutions[0]
        for child_solutions in children_solutions[1:]:
            solutions = self.merge_pair(solutions, child_solutions, node.point)
            solutions = self.elimination(solutions)
        
        return solutions

    def buffer_insertion(self, step: int = 1) -> List[Candidate]:
        root_id = self.graph.root.id
        
        backtrack = [root_id]
        self.visited = {None: []} 
        
        while backtrack:
            node_id = backtrack[-1]
            node = self.graph.nodes[node_id]
            
            children_solutions = []
            children = [e.target for e in self.graph.edges.values() if e.source == node_id]
            
            for child_id in children:
                if child_id not in self.visited:
                    backtrack.append(child_id)
                else:
                    children_solutions.append(self.visited[child_id])
            
            if len(children_solutions) < len(children):
                continue
                
            solutions = self.merge_solutions(children_solutions, node)
            solutions = self.elimination(solutions)
            
            if node_id == root_id:
                for solution in solutions:
                    self.insert_buffer(solution)
                    solution[-1].has_buffer = False
                self.visited[node_id] = solutions
                backtrack.pop()
                continue
                
            parent_edge = next((e for e in self.graph.edges.values() if e.target == node_id), None)
            if not parent_edge:
                self.visited[node_id] = solutions
                backtrack.pop()
                continue
                
            points = self.split_edge(parent_edge, step)
            
            for point in points:
                for solution in solutions:
                    length = self.manhattan_distance(solution[-1].position, point)
                    self.insert_wire(solution, length, point, parent_edge.id)
                
                solutions = self.elimination(solutions)
                
                buffered_solutions = [sol.copy() for sol in solutions]
                for solution in buffered_solutions:
                    length = self.manhattan_distance(solution[-1].position, point)
                    self.insert_wire(solution, length, point, parent_edge.id)
                    self.insert_buffer(solution)
                
                solutions.extend(buffered_solutions)
                solutions = self.elimination(solutions)
            
            self.visited[node_id] = solutions
            backtrack.pop()
        
        root_solutions = self.visited.get(root_id, [])
        if not root_solutions:
            raise ValueError("No valid solutions found")
        
        return max(root_solutions, key=lambda s: s[-1].rat)


def insert_buffers(graph: Graph, step: int = 1) -> List[Candidate]:
    inserter = BufferInserter(graph, step)
    return inserter.buffer_insertion()

