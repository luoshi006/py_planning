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
import numpy as np
import matplotlib.pyplot as plt

# init param
start_point = np.array([1, 1, 0]).reshape((3, 1))
goal_point = np.array([9, 9, 0]).reshape(3, 1)

fig, ax = plt.subplots()

env = EnvBase('../resource/grid_map.yaml'  # grid_map, map_ningde
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

# get static map
map_static = env.get_grid_map()*(2.55)
# map_static = 255 - map_static.astype(np.uint8)
map_static = map_static.astype(np.uint8)

map_static_cv = cv2.Mat(map_static)

kernel = np.ones((r_px[0,0], r_px[1,0]),np.uint8)
#以机器人形状腐蚀
# map_static_cv_inflate = cv2.erode(map_static_cv, kernel)
#以机器人形状膨胀
map_static_cv_inflate = cv2.dilate(map_static_cv, kernel)

contours, hierarchy = cv2.findContours(map_static_cv_inflate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1)

contours_refine = []
for i in range(len(contours)):
    contour_cur = contours[i]
    # epsilon = 0.01 * cv2.arcLength(contour_cur, True)
    # print(epsilon)
    epsilon = 1.5
    # cv::approxPolyDP(raw_contours[i], refined_contours[i], DIST_LIMIT, true)
    contour_new = cv2.approxPolyDP(contour_cur, epsilon, True)
    contours_refine.append(contour_new)

# construct V-Graph
contours_dict = {}
edge = []
for i in range(len(contours_refine)):
    # each obstacle polygon
    contour_i = contours_refine[i]
    pts_list = []
    for j in range(len(contour_i)):
        pt = contour_i[j].flatten()
        pt1 = Point(pt[0], pt[1])
        pts_list.append(pt1)

        if i==0 and j == 0: ax.plot(pt[1], pt[0], 'r.', alpha=0.7, label='vertex')
        else: ax.plot(pt[1], pt[0], 'r.', alpha=0.7)

        # print("({:.3f},{:.3f}) ".format(pt[0], pt[1]))



        pt_idx = j+1
        if j == len(contour_i)-1: pt_idx = 0
        pt = contour_i[pt_idx].flatten()
        pt2 = Point(pt[0], pt[1])
        edge.append(Segment(pt1, pt2))

        if i==0 and j == 0: ax.plot([pt1.y, pt2.y], [pt1.x, pt2.x], 'b-', alpha=1, label="contour")
        else: ax.plot([pt1.y, pt2.y], [pt1.x, pt2.x], 'b-', alpha=1)
    contours_dict[i] = pts_list

v_graph = VisibilityGraph(contours_dict, edge)
v_edge = v_graph.v_edge

for i in range(len(v_edge)):
    seg_i = v_edge[i]
    if i==0: ax.plot([seg_i.p1.y, seg_i.p2.y], [seg_i.p1.x, seg_i.p2.x], 'g-', alpha=0.4, linewidth='0.3', label="visibility edge")
    else: ax.plot([seg_i.p1.y, seg_i.p2.y], [seg_i.p1.x, seg_i.p2.x], 'g-', alpha=0.4, linewidth='0.3')


# plot vertex

# map_static_cv_inflate_rgb = 255 - cv2.cvtColor(map_static_cv_inflate, cv2.COLOR_GRAY2RGB)
# cv2.drawContours(map_static_cv_inflate_rgb, contours_refine, -1, (0,255,0), 2)

# cv2.imshow("dbg", map_static_cv_inflate_rgb)
# cv2.waitKey(0)
# cv2.imwrite("fig/map_inflate_contours1.png", map_static_cv_inflate_rgb)


ax.imshow(map_static.T, cmap='Greys', origin='lower')
fig.legend()
fig.show()
fig.waitforbuttonpress()
# plt.pause(20)

fig.savefig("map_contours_edge.png", dpi=400)

print("map_static")

# env.end(ani_name='grid_map', ending_time=20, ani_kwargs={'subrectangles': True})