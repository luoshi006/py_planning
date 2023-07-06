#https://github.com/melkir/A-Star-Python/blob/master/Algorithms.py

from queue import PriorityQueue
import numpy as np

def np_hash(pt):
    return f'{pt[0]}a{pt[1]}'
def hash_pt(hash):
    num_str = hash.split("a",1)
    return np.array([int(num_str[0]), int(num_str[1])])

class SearchGraph:
    def __init__(self) -> None:
        self.vertex = []
        self.neighbor_dict = {}

    def add_edge(self, pt1, pt2, bi_direct=False):
        if np_hash(pt1) not in self.vertex:
            self.vertex.append(np_hash(pt1))
            self.neighbor_dict[np_hash(pt1)] = []
        if np_hash(pt2) not in self.vertex:
            self.vertex.append(np_hash(pt2))
            self.neighbor_dict[np_hash(pt2)] = []

        self.neighbor_dict[np_hash(pt1)].append(pt2)
        if bi_direct:
            self.neighbor_dict[np_hash(pt2)].append(pt1)

    def neighbors(self, pt_hash):
        return self.neighbor_dict[pt_hash]

    def cost(self, pt1, pt2):
        return np.linalg.norm(pt1-pt2)

def edge_to_search_graph(edge_nx4):
    sg = SearchGraph()
    for i in range(edge_nx4.shape[0]):
        seg = edge_nx4[i,:]
        pt1 = seg[:2]
        pt2 = seg[-2:]
        sg.add_edge(pt1, pt2, bi_direct=True)
    return sg

def reconstruct_path(came_from, start, goal):
    start_hash = np_hash(start)
    goal_hash = np_hash(goal)

    cur = goal
    cur_hash = goal_hash
    path = [cur]
    while cur_hash != start_hash:
        cur_hash = came_from[cur_hash]
        cur = hash_pt(cur_hash)
        path.append(cur)
    path.reverse()
    return path

def dijkstra_search(graph, start, goal):
    start_hash = np_hash(start)
    goal_hash = np_hash(goal)
    frontier = PriorityQueue()
    frontier.put(start_hash, 0)
    came_from = {start_hash: None}
    cost_so_far = {start_hash: 0}

    while not frontier.empty():
        cur_hash = frontier.get()
        cur_pt = hash_pt(cur_hash)
        if cur_hash == goal_hash:
            break
        for next_pt in graph.neighbors(cur_hash):
            next_hash = np_hash(next_pt)
            new_cost = cost_so_far[cur_hash] + graph.cost(cur_pt, next_pt)
            if next_hash not in cost_so_far or new_cost < cost_so_far[next_hash]:
                cost_so_far[next_hash] = new_cost
                priority = new_cost
                frontier.put(next_hash, priority)
                came_from[next_hash] = cur_hash

    return reconstruct_path(came_from, start, goal), cost_so_far[goal_hash]