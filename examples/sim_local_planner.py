import sys

sys.path.append('..')
sys.path.append('../thirdparty/ir_sim')
from ir_sim.env import EnvBase
from lib.utils import scan_to_pointcloud
from planner.local_planner.local_planner import LocalPlanner
import numpy as np

# init param
start_point = np.array([1, 1, 0]).reshape((3, 1))
goal_point = np.array([9, 9, 0]).reshape(3, 1)

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

lp = LocalPlanner(vehicle_poly_nx2)

for i in range(300):
    # get info
    robot_info = env.get_robot_info()
    robot_pos = robot_info.state
    lidar_scan = env.get_lidar_scan()  # already transform to body frame in lidar.scan_matrix
    lidar_pts_2xn_b = scan_to_pointcloud(lidar_scan)

    # set planner
    lp.set_robot_pose(robot_pos[0], robot_pos[1], robot_pos[2])
    lp.set_goal(goal_point[0], goal_point[1], goal_point[2])
    lp.set_obs_pts_body(lidar_pts_2xn_b.T)
    if lp.run():
        path_choosed = lp.get_best_path()
        path_feasible = lp.get_feasible_path()

    vel_cmd = np.array([[2], [0.6]]);
    ref_path_pts_2xn = path_choosed.eval_arclen()
    env.draw_trajectory(ref_path_pts_2xn, traj_type='-b', refresh=True)

    env.step(vel_cmd)
    env.render(show_traj=True)

    if env.done(): break

env.end(ani_name='grid_map', ending_time=20, ani_kwargs={'subrectangles': True})
