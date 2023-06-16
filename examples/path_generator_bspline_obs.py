import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon

import sys
sys.path.append('..')
from lib.path.cubic_uniform_bspline_2d import solver_cubic_uniform_bspline_2d_v2
from lib.path.bspline_path import BSplinePath2D
from lib.collision_detect.point_in_poly import point_in_polygon

# generate path
dis = 1.0
angle = 27
delta_angle = (angle/3)
scale = 0.65
angle_ori = 90

fov = 30
fov_half = ((fov/2) // angle)*angle
idx_path = 0

# obstacle
obs_poly_nx2 = np.array([ 1.6, 1.6
                        , 1.8, 1.9
                        , 1.5, 2.3
                        , 0.6, 1.8
                        , 1.6, 1.6]).reshape((5,2))    # close vertex

# draw gif
flg_draw = True
if flg_draw:
    fig, ax = plt.subplots()  # Create a figure containing a single axes.
    obs_patch = Polygon(obs_poly_nx2, alpha=0.8, fc='k')
    ax.add_patch(obs_patch)

for shift0 in np.arange(angle_ori-fov_half, angle_ori+fov_half+1, angle):
    for shift1 in np.arange(-angle+shift0, angle+shift0+1, delta_angle):
        shift1 = round(shift1, 2)
        shift1_rad = np.deg2rad(shift1)

        angle2 = (angle * scale)
        angle3 = (angle2 * scale)
        delta_angle2 = (delta_angle * scale)
        delta_angle3 = (delta_angle2 * scale)
        for shift2 in np.arange(-angle2+shift1, angle2+shift1+1, delta_angle2):
            shift2 = round(shift2, 2)
            shift2_rad = np.deg2rad(shift2)

            for shift3 in np.arange(-angle3+shift2, angle3+shift2+1, delta_angle3):
                shift3 = round(shift3, 2)
                shift3_rad = np.deg2rad(shift3)

                sample_step = 1/2
                cur_rad = 0
                path_waypoints_list = []
                flg_collision = False
                for dis_cur in np.arange(0,3+sample_step,sample_step):
                    if dis_cur < 1.0:
                        angle_range = shift1 - shift0
                        step_scale = dis_cur
                        cur_rad = np.deg2rad(shift0 + angle_range*step_scale)
                    elif dis_cur < 2.0:
                        angle_range = shift2 - shift1
                        step_scale = dis_cur - 1
                        cur_rad = np.deg2rad(shift1 + angle_range*step_scale)
                    else:
                        angle_range = shift3 - shift2
                        step_scale = dis_cur - 2
                        cur_rad = np.deg2rad(shift2 + angle_range*step_scale)

                    cur_way_pts_x = round(dis_cur*np.cos(cur_rad), 4)
                    cur_way_pts_y = round(dis_cur*np.sin(cur_rad), 4)
                    cur_pts = np.array([cur_way_pts_x,cur_way_pts_y])
                    # check way point collision
                    if point_in_polygon(cur_pts, obs_poly_nx2):
                        flg_collision = True
                        break
                    path_waypoints_list.append(cur_pts)

                if flg_collision:
                    continue
                path_waypoints_array = np.array(path_waypoints_list)
                vel_s = sample_step*5
                vel0 = np.array([[np.cos(np.deg2rad(shift0))], [np.sin(np.deg2rad(shift0))]]) * vel_s
                vel1 = np.array([[np.cos(np.deg2rad(shift3))], [np.sin(np.deg2rad(shift3))]]) * vel_s

                # solve uniform BSpline ctrl pts from waypoints
                ctrl_pts = solver_cubic_uniform_bspline_2d_v2(path_waypoints_array.T, vel0, vel1)
                # constructor for bspline
                spl_cur = BSplinePath2D(ctrl_pts)

                if spl_cur.collision_check(obs_poly_nx2):
                    flg_collision = True
                    continue

                idx_path = idx_path + 1

                # draw spline
                if flg_draw:
                    show_pts = spl_cur.eval_list()
                    ax.plot(show_pts[0,:], show_pts[1,:], linewidth='0.5')
                    plt.axis('equal')
                    plt.show(block=False)
                    plt.pause(0.001)

print("====")
print("%d path generated." % (idx_path))

if flg_draw:
    # plt.savefig('path_generator_bspline_obs.png', dpi=600)
    plt.show()
