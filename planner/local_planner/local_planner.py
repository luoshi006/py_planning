import numpy as np
from queue import PriorityQueue
import time

import sys
sys.path.append('../..')
from lib.path.cubic_uniform_bspline_2d import solver_cubic_uniform_bspline_2d_v3
from lib.path.bspline_path import BSplinePath2D
from lib.collision_detect.point_in_poly import point_in_polygon
from lib.utils import get_transform

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
        self.goal_b = np.array([0,0,0])
        self.flg_goal_psi_enable = False

        self.obs_poly_nx2s_b = []
        self.obs_pts_nx2_b = np.array([])
        self.robot_poly_nx2_b = robot_poly_nx2

        self.path_feasible_b = {}
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
        goal_xy_b, goal_psi_b = self.trans_to_body(x,y,psi)
        self.goal_b = np.append(goal_xy_b,goal_psi_b).flatten()
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
        return self.transform_spline_dict_to_global(self.path_feasible_b)

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

    def generate_paths(self):
        idx_path = 0
        goal_dir = np.arctan2(self.goal_b[1], self.goal_b[0])
        fov_scale = 1
        if abs(goal_dir) > self.fov:
            fov_scale = goal_dir // self.fov + 1
        fov = fov_scale * self.fov
        fov_half = ((fov/2) // self.angle) * self.angle
        path_dict = {}
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

                            cur_way_pts_x = dis_cur*np.cos(cur_rad)
                            cur_way_pts_y = dis_cur*np.sin(cur_rad)
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

                        # approx uniform BSpline from waypoints
                        vel_s = self.sample_step*5
                        vel0 = np.array([[np.cos(np.deg2rad(shift0))], [np.sin(np.deg2rad(shift0))]]) * vel_s * 0.5
                        vel1 = np.array([[np.cos(np.deg2rad(shift3))], [np.sin(np.deg2rad(shift3))]]) * vel_s

                        ctrl_pts = solver_cubic_uniform_bspline_2d_v3(path_waypoints_array.T, vel0, vel1)
                        spl_cur = BSplinePath2D(ctrl_pts)
                        t_2 = time.time()

                        # check spline collistion here
                        if self.check_spline_collision(spl_cur):
                            flg_collision = True
                            continue

                        path_dict[idx_path] = spl_cur
                        t_3 = time.time()

                        # calc cost
                        cost_angle = np.mean(np.abs(np.diff(path_waypoints_angle_array/np.pi)))
                        cost_ref_goal = np.linalg.norm(path_waypoints_array[-1,:] - self.goal_b[:2])
                        cost_curvature = np.max(np.abs(spl_cur.curvature(np.linspace(0,1,100))))
                        cost_len = spl_cur.arclen
                        cost_total = cost_angle + cost_ref_goal + cost_len*0
                        path_cost_dq.put((cost_total, idx_path))
                        idx_path = idx_path + 1
                        t_4 = time.time()
                        t_gen = t_gen + (t_1-t_0)
                        t_spl = t_spl + (t_2-t_1)
                        t_hul = t_hul + (t_3-t_2)
                        t_cst = t_cst + (t_4-t_3)
        # get the best path
        if path_cost_dq.empty():
            print("lp generate NULL path!")
            return False
        cost_best, idx_best = path_cost_dq.get()
        self.path_feasible_b = path_dict
        self.path_best_g = self.transform_spline_to_global(path_dict[idx_best])
        # lp_gen: 0.0558, spl: 0.285, hull: 2.35, cost: 0.0321s
        print("lp_gen: {:.3}, spl: {:.3}, hull: {:.3}, cost: {:.3}s".format(t_gen,t_spl,t_hul,t_cst))
        return True

    def check_need_replan(self):
        res = False
        if [] == self.path_best_g:
            res = True
        if self.flg_need_replan:
            res = True
        if not res:
            res = self.dist_traveled > 0.5*self.path_best_g.arclen
        if res:
            self.flg_need_replan = False    # reset replan flg
        return res

    def run(self):
        #TODO: check input param correct
        # print("Local Planner Run ...")
        if self.check_need_replan():
            t_start = time.time()
            if self.generate_paths():
                self.dist_traveled = 0
            else:
                #TODO:
                assert False, "impl"
            t_end = time.time()
            print("lp time: {:.2f}s".format(t_end-t_start))
            return True
        return False


# ============ test =========================
if __name__ == "__main__":
    lp = LocalPlanner()
    lp.set_robot_pose(1,1,np.deg2rad(90))
    xx0 = lp.trans_to_body(1,1,np.deg2rad(90))
    print(xx0)
    xx1 = lp.trans_to_global(0,0,np.deg2rad(90))
    print(xx1)