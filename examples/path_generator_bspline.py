import matplotlib.pyplot as plt
import numpy as np
from queue import PriorityQueue

import sys
sys.path.append('..')
from lib.path.cubic_uniform_bspline_2d import solver_cubic_uniform_bspline_2d_v2, solver_cubic_uniform_bspline_2d_v3
from lib.path.bspline_path import BSplinePath2D

# generate path
dis = 1.0
angle = 27
delta_angle = (angle/3)
scale = 0.65
angle_ori = 90

fov = 30
fov_half = ((fov/2) // angle)*angle
idx_path = 0
path_dict = {}      # key: pathid, value: path
path_cost_dq = PriorityQueue()

# draw gif
flg_draw = True
if flg_draw:
    fig, ax = plt.subplots()  # Create a figure containing a single axes.

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
                path_waypoints_angle_list = []
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

                    cur_way_pts_x = dis_cur*np.cos(cur_rad)
                    cur_way_pts_y = dis_cur*np.sin(cur_rad)
                    cur_pts = np.array([cur_way_pts_x,cur_way_pts_y])
                    path_waypoints_list.append(cur_pts)
                    path_waypoints_angle_list.append(cur_rad)
                path_waypoints_array = np.array(path_waypoints_list)
                path_waypoints_angle_array = np.unwrap(np.array(path_waypoints_angle_list))
                # calc cost
                cost_angle = np.mean(np.abs(np.diff(path_waypoints_angle_array/np.pi)))     # [0,1]
                cost_ref_goal = 0

                vel_s = sample_step*5
                vel0 = np.array([[np.cos(np.deg2rad(shift0))], [np.sin(np.deg2rad(shift0))]]) * vel_s*0.5
                vel1 = np.array([[np.cos(np.deg2rad(shift3))], [np.sin(np.deg2rad(shift3))]]) * vel_s

                # solve uniform BSpline ctrl pts from waypoints
                ctrl_pts = solver_cubic_uniform_bspline_2d_v3(path_waypoints_array.T, vel0, vel1)
                # constructor for bspline
                spl_cur = BSplinePath2D(ctrl_pts)
                path_dict[idx_path] = spl_cur

                cost_len   = spl_cur.arclen
                cost_total = cost_angle + cost_ref_goal + cost_len*0
                path_cost_dq.put((cost_total, idx_path))

                idx_path = idx_path + 1

                # calc curvate
                flg_prt_kpa = False
                if flg_prt_kpa:
                    kappa_u = np.linspace(0,0.5,100)
                    kappa = spl_cur.curvature(kappa_u)
                    print("first half traj kappa max: {}".format(np.round(np.max(np.abs(kappa)),2)))

                # draw spline
                if flg_draw:
                    show_pts = spl_cur.eval_list()
                    ax.plot(show_pts[0,:], show_pts[1,:], linewidth='0.5')
                    ax.plot(path_waypoints_array[:,0], path_waypoints_array[:,1], 'r.', markersize='2')
                    plt.axis('equal')
                    plt.show(block=False)
                    plt.pause(0.001)
# get the best path
if not path_cost_dq.empty():
    cost_ch, idx_ch = path_cost_dq.get()
    print("get the choosed path idx {}, cost: {}".format(idx_ch, cost_ch))
    if flg_draw:
        show_pts_ch = path_dict[idx_ch].eval_arclen()
        ax.plot(show_pts_ch[0,:], show_pts_ch[1,:], 'g', linewidth='2')


print("====")
print("%d path generated." % (idx_path))

if flg_draw:
    # plt.savefig('path_generator_bspline_default.png', dpi=600)
    plt.show()
