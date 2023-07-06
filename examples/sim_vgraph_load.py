import sys
# sys.path.append('..')
# sys.path.append('../thirdparty/ir_sim')
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))+"/thirdparty/ir_sim")
from ir_sim.env import EnvBase
from lib.utils import scan_to_pointcloud
from planner.global_planner.v_graph.visibility_graph import VisibilityGraph, Point, Segment

import cv2
import csv
import numpy as np
import matplotlib.pyplot as plt

# init param
start_point = np.array([1, 1, 0]).reshape((3, 1))
goal_point = np.array([9, 9, 0]).reshape(3, 1)

fig, ax = plt.subplots()

env = EnvBase('../resource/cave.yaml'  # cave, map_ningde
              # , control_mode='keyboard'
              , init_args={'no_axis': False}
              , collision_mode='stop'
              , save_ani=False
              , save_fig=True
              , robot_args={'state': start_point, 'goal': goal_point})

r = env.get_robot_info().shape
vehicle_poly_nx2 = np.array([r, r
                        , -r, r
                        , -r, -r
                        , r, -r]).reshape((4,2))
reso = env.world.reso
r_px = 2 * r / reso
r_px = r_px.astype(np.uint8)
start_px = start_point[:2]/reso
start_px = start_px.astype(np.int64)
goal_px = goal_point[:2]/reso
goal_px = goal_px.astype(np.int64)


# get static map
map_static = env.get_grid_map()*(2.55)
# map_static = 255 - map_static.astype(np.uint8)
map_static = map_static.astype(np.uint8)

map_static_cv = cv2.Mat(map_static)


# load from csv
graph_filename = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))+"/resource/cave.csv"

v_graph = VisibilityGraph()
v_graph.load(graph_filename)
v_graph.set_start_pt(start_px[0,0], start_px[1,0])
v_graph.set_goal_pt(goal_px[0,0], goal_px[1,0])
v_graph.update_pt()

path, cost = v_graph.get_shortest_path()

ax = v_graph.draw(ax, True)

for i in range(len(path)-1):
    p1 = path[i]
    p2 = path[i+1]
    if 0 == i: ax.plot([p1[1],p2[1]], [p1[0],p2[0]], 'r-', linewidth=1, label='shortest path')
    else: ax.plot([p1[1],p2[1]], [p1[0],p2[0]], 'r-', linewidth=1)

ax.imshow(map_static.T, cmap='Greys', origin='lower')
fig.legend()
fig.show()
fig.waitforbuttonpress()

fig.savefig("examples/fig/map_visibility_graph.png", dpi=400)
print("done")
