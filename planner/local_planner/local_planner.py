import numpy as np
from queue import PriorityQueue
from collections import namedtuple
import time

import sys
sys.path.append('../..')
from lib.path.cubic_uniform_bspline_2d import solver_cubic_uniform_bspline_2d_v3
from lib.path.bspline_path import BSplinePath2D
from lib.collision_detect.point_in_poly import point_in_polygon
from lib.utils import get_transform, wrap_pi

PathInfo = namedtuple('PathInfo', 'point_arr, vel0, vel1')

class LocalPlanner:
    def __init__(self, robot_poly_nx2=[]):
        self.dis = 1.0
        self.sample_step = 1/2
        self.angle = 27
        self.delta_angle = (self.angle/3)
        self.scale = 0.65
        self.angle_ori = 0
        self.fov = 30

        self.dict_id_path = {}
        self.dq_cost_id = PriorityQueue()

        self.pose = np.zeros(3)         # robot pose
        self.goal_g = np.array([0,0,0])
        self.flg_goal_psi_enable = False

        self.obs_poly_nx2s_b = []
        self.obs_pts_nx2_b = np.array([])
        self.robot_poly_nx2_b = robot_poly_nx2

        self.path_alternative_b = {}
        self.path_best_g = []

        self.dist_traveled = 0
        self.pose_last = np.zeros(3)
        self.flg_need_replan = True

    def R_gb(self):
        return np.array([[np.cos(self.pose[2]), -np.sin(self.pose[2])],
                         [np.sin(self.pose[2]),  np.cos(self.pose[2])]])
    def trans_to_body(self, x_g, y_g, psi=[]):
        x_g = np.atleast_1d(x_g).flatten()
        y_g = np.atleast_1d(y_g).flatten()
        psi = np.atleast_1d(psi).flatten()

        trans_g = np.vstack((x_g, y_g))
        xy_2xn_b = self.R_gb().T @ trans_g - self.R_gb().T@self.pose[:2].reshape((2,1))
        psi_b = []

        if len(psi) > 0:
            assert len(x_g) == len(psi), "angle size should be equal to position"
            psi_b = psi - self.pose[2]
        return xy_2xn_b.T, psi_b
    def trans_to_global(self, x_b, y_b, psi=[]):
        x_b = np.atleast_1d(x_b).flatten()
        y_b = np.atleast_1d(y_b).flatten()
        psi = np.atleast_1d(psi).flatten()

        trans_b = np.vstack((x_b, y_b))
        xy_2xn_g = self.R_gb() @ trans_b + self.pose[:2].reshape((2,1))
        psi_g = []

        if len(psi) > 0:
            assert len(x_b) == len(psi), "angle size should be equal to position!"
            psi_g = self.pose[2] + psi
        return xy_2xn_g.T, psi_g

    def set_robot_pose(self, x, y, psi):
        self.pose_last = self.pose
        self.pose = np.array([x,y,psi])
        if np.linalg.norm(self.pose_last) > 0:
            self.dist_traveled = self.dist_traveled + np.linalg.norm(self.pose[:2]-self.pose_last[:2])

    def set_goal(self, x, y, psi=[]):
        if psi == []:
            psi = 0
            self.flg_goal_psi_enable = False
        else:
            self.flg_goal_psi_enable = True

        if np.linalg.norm(self.goal_g[:2] - np.array([x,y])) < 0.001 and wrap_pi(self.goal_g[2]-psi) < np.deg2rad(1):
            pass
        else:
            self.goal_g = np.array([x,y,psi])
            self.flg_need_replan = True

    def set_obs_polys(self, obs_poly_nx2s_g):
        obs_poly_nx2s_b = []
        for i in obs_poly_nx2s_g:
            obs_poly_nx2 = obs_poly_nx2s_g[i]
            obs_poly_nx2_b = self.trans_to_body(obs_poly_nx2[:,0], obs_poly_nx2[:,1])
            obs_poly_nx2s_b.append(obs_poly_nx2_b)
        self.obs_poly_nx2s_b = obs_poly_nx2s_b
    def set_obs_pts_global(self, obs_pts_nx2_g):
        self.obs_pts_nx2_b = self.trans_to_body(obs_pts_nx2_g[:,0], obs_pts_nx2_g[:,1])
    def set_obs_pts_body(self, obs_pts_nx2_b):
        self.obs_pts_nx2_b = obs_pts_nx2_b

    def get_best_path(self):
        return self.path_best_g
    def get_feasible_path(self):
        return self.transform_spline_dict_to_global(self.path_alternative_b)

    def check_waypts_collision(self, x, y, psi):
        trans, rot = get_transform(x,y,psi)
        vehicle_contour_delta = []
        if [] != self.robot_poly_nx2_b:
            vehicle_contour_delta = self.robot_poly_nx2_b * 1.5
            waypts_contour_2xn = rot @ vehicle_contour_delta.T + trans
            waypts_contour_close_nx2 = np.vstack((waypts_contour_2xn.T, waypts_contour_2xn.T[0,:]))
            for i in range(self.obs_pts_nx2_b.shape[0]):
                if point_in_polygon(self.obs_pts_nx2_b[i,:], waypts_contour_close_nx2):
                    return True
        else:
            for i in range(self.obs_pts_nx2_b.shape[0]):
                if np.linalg.norm(self.obs_pts_nx2_b[i,:] - np.array([x,y])) < 0.01:
                    return True
        return False

    def check_waypts_polyline_collision(self, waypts_nx2):
        num = waypts_nx2.shape[0]
        r = 0.2
        if [] != self.robot_poly_nx2_b:
            r = np.max(self.robot_poly_nx2_b[:,1]) # y axis max value
        for i in range(num-1):
            start_pt = waypts_nx2[i,:]
            end_pt = waypts_nx2[i+1,:]
            vec_diff_1x2 = end_pt - start_pt
            angle_rad = np.arctan2(vec_diff_1x2[1],vec_diff_1x2[0])
            # start point
            trans,rot = get_transform(start_pt[0], start_pt[1], angle_rad)
            start_lr_2xn = rot @ np.array([[0,-r], [0,r]]).T + trans
            # end point
            trans,rot = get_transform(end_pt[0],end_pt[1],angle_rad)
            end_rl_2xn = rot @ np.array([[0,r], [0,-r]]).T + trans

            poly_open_nx2 = np.vstack((start_lr_2xn.T, end_rl_2xn.T))
            poly_close_nx2 = np.vstack((poly_open_nx2, poly_open_nx2[0,:]))
            # print("####", poly_open_nx2.T)
            for j in range(self.obs_pts_nx2_b.shape[0]):
                obs_pt_1x2 = self.obs_pts_nx2_b[j, :]
                if point_in_polygon(obs_pt_1x2, poly_close_nx2):
                    return True
        return False

    def check_spline_collision(self, spl):
        vehicle_contour_delta = []
        spl_hulls = []
        delta_x = 0
        delta_y = 0
        if [] != self.robot_poly_nx2_b:
            vehicle_contour_delta = self.robot_poly_nx2_b * 1.5
            delta_x = np.max(np.abs(vehicle_contour_delta[:,0]))
            delta_y = np.max(np.abs(vehicle_contour_delta[:,1]))

        for j in range(self.obs_pts_nx2_b.shape[0]):
            obs_pt_1x2 = self.obs_pts_nx2_b[j,:]
            # check in bounding box
            if (obs_pt_1x2[0] > spl.bbox_leftdown[0] and obs_pt_1x2[1] > spl.bbox_leftdown[1]
                and obs_pt_1x2[0] < spl.bbox_rightup[0] and obs_pt_1x2[1] < spl.bbox_rightup[1]) :
                if [] == spl_hulls:
                    spl_hulls = spl.convex_hulls_of_curve(vehicle_contour_delta)
                if self.check_hulls_collision(spl_hulls, obs_pt_1x2):
                    return True
        return False

    def check_hulls_collision(self, hulls, pt):
        for i in range(len(hulls)):
            poly_nx2 = hulls[i]
            poly_close_nx2 = np.vstack((poly_nx2, poly_nx2[0,:]))
            if point_in_polygon(pt, poly_close_nx2):
                return True
        return False

    def transform_spline_to_global(self, spline):
        trans, rot = get_transform(self.pose[0], self.pose[1], self.pose[2])
        ctrl_pts_2xn_b = spline.ctrl_pts_2xn
        ctrl_pts_2xn_g = rot @ ctrl_pts_2xn_b + trans
        return BSplinePath2D(ctrl_pts_2xn_g)

    def transform_spline_dict_to_global(self, spl_dict):
        spls_g = {}
        for path_idx, path_spl_b in spl_dict.items():
            spls_g[path_idx] = self.transform_spline_to_global(path_spl_b)
        return spls_g

    def generate_paths(self, pathScale = 1, fovScale = 1):
        idx_path = 0
        xy,psi = self.trans_to_body(self.goal_g[0], self.goal_g[1], self.goal_g[2])
        goal_b = np.append(xy,psi).flatten()
        goal_dir = np.arctan2(goal_b[1], goal_b[0])
        fov_scale = fovScale
        # if abs(goal_dir) > self.fov:
        #     fov_scale = goal_dir // self.fov + 1
        fov = fov_scale * self.fov
        fov_half = ((fov/2) // self.angle) * self.angle
        path_inf_dict = {}
        path_cost_dq = PriorityQueue()
        t_gen = 0
        t_spl = 0
        t_hul = 0
        t_chk = 0
        t_cst = 0

        for shift0 in np.arange(self.angle_ori-fov_half, self.angle_ori+fov_half+1, self.angle):
            for shift1 in np.arange(-self.angle+shift0, self.angle+shift0+1, self.delta_angle):
                shift1 = round(shift1,2)
                shift1_rad = np.deg2rad(shift1)

                angle2 = self.angle*self.scale
                delta_angle2 = self.delta_angle*self.scale
                angle3 = angle2*self.scale
                delta_angle3 = delta_angle2*self.scale
                for shift2 in np.arange(-angle2+shift1, angle2+shift1+1, delta_angle2):
                    shift2 = round(shift2,2)
                    shift2_rad = np.deg2rad(shift2)
                    for shift3 in np.arange(-angle3+shift2, angle3+shift2+1, delta_angle3):
                        shift3 = round(shift3,2)
                        shift3_rad = np.deg2rad(shift3)

                        cur_rad = 0
                        flg_collision = False
                        path_waypoints_list = []
                        path_waypoints_angle_list = []
                        t_0 = time.time()
                        for dis_cur in np.arange(0,3+self.sample_step, self.sample_step):
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

                            cur_way_pts_x = pathScale*dis_cur*np.cos(cur_rad)
                            cur_way_pts_y = pathScale*dis_cur*np.sin(cur_rad)
                            cur_pts = np.array([cur_way_pts_x,cur_way_pts_y])

                            if self.check_waypts_collision(cur_way_pts_x, cur_way_pts_y, cur_rad):
                                flg_collision = True
                                break

                            path_waypoints_list.append(cur_pts)
                            path_waypoints_angle_list.append(cur_rad)

                        if flg_collision:
                            continue
                        path_waypoints_array = np.array(path_waypoints_list)
                        path_waypoints_angle_array = np.unwrap(np.array(path_waypoints_angle_list))
                        t_1 = time.time()

                        # check collision for polygonal line
                        if self.check_waypts_polyline_collision(path_waypoints_array):
                            flg_collision = True
                            continue

                        # approx uniform BSpline from waypoints
                        vel_s = self.sample_step*5
                        vel0 = np.array([[np.cos(np.deg2rad(shift0))], [np.sin(np.deg2rad(shift0))]]) * vel_s * 0.5
                        vel1 = np.array([[np.cos(np.deg2rad(shift3))], [np.sin(np.deg2rad(shift3))]]) * vel_s

                        path_info = PathInfo(path_waypoints_array.T, vel0, vel1)
                        # ctrl_pts = solver_cubic_uniform_bspline_2d_v3(path_waypoints_array.T, vel0, vel1)
                        # spl_cur = BSplinePath2D(ctrl_pts)
                        t_2 = time.time()

                        # # check spline collistion here
                        # if self.check_spline_collision(spl_cur):
                        #     flg_collision = True
                        #     continue

                        path_inf_dict[idx_path] = path_info
                        t_3 = time.time()

                        # calc cost
                        cost_angle = np.mean(np.abs(np.diff(path_waypoints_angle_array/np.pi)))
                        cost_ref_goal = np.linalg.norm(path_waypoints_array[-1,:] - goal_b[:2])
                        # cost_curvature = np.max(np.abs(spl_cur.curvature(np.linspace(0,1,100))))
                        # cost_len = spl_cur.arclen
                        cost_total = cost_angle + cost_ref_goal
                        path_cost_dq.put((cost_total, idx_path))
                        idx_path = idx_path + 1
                        t_4 = time.time()
                        t_gen = t_gen + (t_1-t_0)
                        t_spl = t_spl + (t_2-t_1)
                        t_hul = t_hul + (t_3-t_2)
                        t_cst = t_cst + (t_4-t_3)
        flg_success = False
        spl_best = []
        # get the best path
        if path_cost_dq.empty():
            print("lp generate NULL path! scale: {:.3f}, fov: {:.3f}".format(pathScale, fov))
            return False

        while not path_cost_dq.empty():
            cost_best, idx_best = path_cost_dq.get()
            path_info = path_inf_dict[idx_best]

            ctrl_pts = solver_cubic_uniform_bspline_2d_v3(path_info.point_arr, path_info.vel0, path_info.vel1)
            spl_cur = BSplinePath2D(ctrl_pts)
            if self.check_spline_collision(spl_cur):
                continue
            else:
                flg_success = True
                spl_best = spl_cur
                break

        if flg_success:
            self.path_alternative_b = path_inf_dict
            self.path_best_g = self.transform_spline_to_global(spl_best)
        # lp_gen: 0.0558, spl: 0.285, hull: 2.35, cost: 0.0321s
        # print("lp_gen: {:.3}, spl: {:.3}, hull: {:.3}, cost: {:.3}s".format(t_gen,t_spl,t_hul,t_cst))
        return flg_success

    def check_need_replan(self):
        res = False
        if [] == self.path_best_g:
            res = True
        elif self.flg_need_replan:
            res = True
        else:
            # check path reach goal
            path_goal = self.path_best_g.eval(1)    # 1x2
            gap = np.linalg.norm(path_goal - self.goal_g[:2])
            if gap < 0.02 :
                res = False
            elif not res:
                res = self.dist_traveled > 1.2

        if res:
            self.flg_need_replan = False    # reset replan flg
        return res

    def run(self):
        #TODO: check input param correct
        # print("Local Planner Run ...")
        path_scale_step = 0.25
        path_scale_min = 0.7
        if self.check_need_replan():
            path_scale = 1.0
            dist_to_goal = np.linalg.norm(self.pose[:2] - self.goal_g[:2])
            if dist_to_goal < 3 :
                path_scale = dist_to_goal/3
            if path_scale < path_scale_min:
                print("path scale should lager than 0.7, too close")
                return False

            if self.generate_paths(path_scale):
                self.dist_traveled = 0
            else:
                if path_scale - path_scale_step > path_scale_min: path_scale = path_scale - path_scale_step
                if self.generate_paths(path_scale, fovScale=3):
                    self.dist_traveled = 0
                else:
                    print("debug:")
                    print(self.goal_g.T, self.pose.T)
                    print(self.obs_pts_nx2_b.T)



            return True
        return False


# ============ test =========================
if __name__ == "__main__":
    r = 0.2
    vehicle_poly_nx2 = np.array([r, r
                                    , -r, r
                                    , -r, -r
                                    , r, -r]).reshape((4, 2))

    lp = LocalPlanner(vehicle_poly_nx2)
    lp.set_robot_pose(1,1,np.deg2rad(90))
    xx0 = lp.trans_to_body(1,1,np.deg2rad(90))
    print(xx0)
    xx1 = lp.trans_to_global(0,0,np.deg2rad(90))
    print(xx1)

    print("===============")
    lp.set_robot_pose(2.103,  1.599, 0.603)
    lp.set_goal(9,9,0)
    scan = np.array([[ 3.41404722e+00,  2.84511605e+00,  2.42747239e+00,  2.32057790e+00, 2.30470485e+00,  2.28585626e+00,  2.32268190e+00,  2.35748940e+00, 2.40995955e+00,  2.46044446e+00,  2.50881615e+00,  2.57490896e+00, 2.61870719e+00,  2.68000000e+00,  2.73864798e+00,  2.83439591e+00, 1.14880414e+00,  1.04300463e+00,  1.01707749e+00,  9.75144401e-01, 9.47659227e-01,  9.05096692e-01,  8.76220309e-01,  8.46479202e-01, 8.15902722e-01,  7.84521044e-01,  7.52365140e-01,  7.19466741e-01, 6.85858316e-01,  6.51573032e-01,  6.26279797e-01,  5.90187671e-01, 5.62028688e-01,  5.24235240e-01,  4.93286926e-01,  4.53908839e-01, 4.26443480e-01,  3.96167401e-01,  3.58113469e-01],
                     [-1.47739013e+00, -1.12646111e+00, -8.73943820e-01, -7.54001454e-01, -6.69578643e-01, -5.86908124e-01, -5.19180906e-01, -4.49715147e-01, -3.81700088e-01, -3.10826414e-01, -2.37152946e-01, -1.61999538e-01, -8.22961874e-02,  5.95079541e-16,  8.60654784e-02,  1.78325072e-01, 8.34655046e-01,  8.09037299e-01,  8.41399654e-01,  8.59705413e-01, 8.89911226e-01,  9.05096668e-01,  9.33079831e-01,  9.60142157e-01, 9.86256939e-01,  1.01139840e+00,  1.03554174e+00,  1.05866312e+00, 1.08073973e+00,  1.10174978e+00,  1.13919867e+00,  1.15830847e+00, 1.19437170e+00,  1.21143610e+00,  1.24590048e+00,  1.26078022e+00, 1.31245798e+00,  1.36361702e+00,  1.39475974e+00]])
    lp.set_obs_pts_body(scan.T)
    lp.run()
    path_choosed = lp.get_best_path()
    path_u = path_choosed.eval_arc2u(0.01)