import matplotlib.pyplot as plt
from matplotlib.patches import Circle, ConnectionPatch
from typing import Dict, List

from src.parse_json import Graph, Node, NodeKind

class GraphVisualizer:
    def __init__(self, graph: Graph):
        self.graph = graph
        self.fig, self.ax = plt.subplots(figsize=(12, 8))
        self._configure_axes()
        self._processed_locations = {}  

    def _configure_axes(self):
        self.ax.set_title('Graph Visualization (Nested Circles for Co-located Nodes)')
        self.ax.set_xlabel('X coordinate')
        self.ax.set_ylabel('Y coordinate')
        self.ax.grid(True, linestyle='--', alpha=0.7)
        self.ax.set_aspect('equal')

    def _draw_node(self, node: Node):
        pos_key = (round(node.point.x, 2), round(node.point.y, 2))
        
        if pos_key in self._processed_locations:
            self._processed_locations[pos_key].append(node)
            return
        
        self._processed_locations[pos_key] = [node]

    def _draw_colocated_nodes(self, nodes: List[Node]):
        if not nodes:
            return
            
        base_node = nodes[0]
        x, y = base_node.point.x, base_node.point.y
        
        nodes_sorted = sorted(nodes, key=lambda n: 1 if n.is_terminal else 0)
        
        for i, node in enumerate(nodes_sorted):
            radius = 0.5 - i * 0.12  
            
            fill_color = {
                NodeKind.BUFFER: 'limegreen',
                NodeKind.TERMINAL: 'tomato',  
                NodeKind.STEINER: 'dodgerblue'
            }.get(node.kind, 'lightgray')
            
            edge_color = {
                NodeKind.TERMINAL: 'darkred',
                NodeKind.BUFFER: 'darkgreen',
                NodeKind.STEINER: 'darkblue'
            }.get(node.kind, 'gray')
            
            circle = Circle(
                (x, y), 
                radius=radius,
                facecolor=fill_color,  
                edgecolor=edge_color,   
                alpha=0.8,
                linewidth=1.5 if node.is_terminal else 1.0
            )
            
            self.ax.add_patch(circle)
            
            if i == 0:
                label = f"{node.name}\nID:{node.id}"
                if node.is_terminal:
                    label += f"\nC:{node.capacitance:.1f}"
                
                self.ax.text(
                    x, y + radius + 0.2,  
                    label,
                    ha='center', va='center',
                    fontsize=8, 
                    bbox=dict(
                        facecolor='white', 
                        alpha=0.7,
                        edgecolor='none',
                        boxstyle='round,pad=0.2'
                    )
                )

    def _draw_edge(self, edge):
        for i in range(len(edge.segments) - 1):
            start = edge.segments[i]
            end = edge.segments[i + 1]
            line = ConnectionPatch(
                (start.x, start.y), (end.x, end.y),
                coordsA='data', coordsB='data',
                mutation_scale=15, color='black', 
                alpha=0.5, linewidth=1
            )
            self.ax.add_patch(line)

    def visualize(self, show=True, save_path=None):
        for edge in self.graph.edges.values():
            self._draw_edge(edge)
        
        for node in self.graph.nodes.values():
            self._draw_node(node)
        
        for nodes in self._processed_locations.values():
            self._draw_colocated_nodes(nodes)
        
        self.ax.autoscale_view()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Graph saved to {save_path}")
        
        if show:
            plt.show()

def visualize_graph(graph: Graph, show=True, save_path=None):
    visualizer = GraphVisualizer(graph)
    visualizer.visualize(show=show, save_path=save_path)