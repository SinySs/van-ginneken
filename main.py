import argparse
import time

from src.parse_json import *  
#from src.visual import visualize_graph
from src.alg import insert_buffers



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('tech_file')
    parser.add_argument('test_file')
    parser.add_argument('--plot', action='store_true')
    args = parser.parse_args()

    config = read_config(args.tech_file)

    graph = Graph.read_from_file(args.test_file)
    inserter = SolutionInserter(graph)

    graph.set_technology(config.technology)
    for module in config.modules:
        graph.add_module(module)

    start_time = time.time()
    solution = insert_buffers(graph)
    end_time = time.time()


    '''
    print("Оптимальное решение:")
    for cand in solution:
        if cand.has_buffer:
            print(f"Buffer at ({cand.position.x}, {cand.position.y})")
            print(f"  RAT: {cand.rat:.4f}")
            print(f"  EdgeId: {cand.edge_id}")
        
    if graph.root:
        print(f"Корневой узел: {graph.root}")
    

    print(f" Result RAT: {solution[-1].rat:.4f}")
    execution_time = end_time - start_time  
    print(f"Exec time: {execution_time:.4f} sec")
    '''

    inserter.insert_solution(solution)

    # if args.plot:
    #     visualize_graph(inserter.graph, save_path="")
    #     return
    
    out_file = args.test_file.replace('.json', '_out.json')
    save_graph_to_json(inserter.graph, out_file)


if __name__ == "__main__":
    main()