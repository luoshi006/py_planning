import numpy as np
from collections import namedtuple

import sys
sys.path.append('../..')
from lib.path.bspline_path import BSplinePath2D
from lib.utils import get_transform, wrap_pi

"""
ref: 
- https://github.com/leggedrobotics/se2_navigation/tree/master/pure_pursuit_core
- Kuwata, Yoshiaki, et al. "Motion planning in complex environments using closed-loop prediction." AIAA Guidance, Navigation and Control Conference and Exhibit. 2008.
"""

Traj = namedtuple('Traj', 'position, tangent_angle, curvature, vel')

class PurePursuit:
    def __init__(self, dt=0.1):
        self.anchor_distance = 0.05      # 可以增加稳定性
        self.lookahead_distance = 0.2   # 需大于 速度 * 上升时间
        self.max_acc = 3.0
        self.dt = dt

        self.pose = np.zeros(3)
        self.pose_last = np.zeros(3)

        self.traj = []
        self.idx_near = 0               # idx of current point in traj

        self.cmd = [0,0]

    def set_robot_pose(self, x, y, psi):
        self.pose_last = self.pose
        self.pose = np.array([x,y,psi])

    def set_traj(self, traj_pos_nx2, traj_psi, traj_curvature, traj_vel):
        #TODO: check shape
        self.traj = Traj(traj_pos_nx2, traj_psi, traj_curvature, traj_vel)
        self.idx_near = 0

    def find_nearest_idx(self, pt):
        # not global nearest, only local nearest
        dist_min = np.inf
        dist_last = np.inf
        idx_min = self.idx_near
        range = min(self.traj.position.shape[0], self.idx_near+100)
        # print("### {}, {}, {}".format(self.traj.position.shape[0], self.idx_near, range))
        traj_pos = self.traj.position
        robot_pos = pt.flatten()
        for i in np.arange(self.idx_near, range, 1):
            dist = np.linalg.norm(traj_pos[i,:] - robot_pos)
            if dist < dist_min:
                dist_min = dist
                idx_min = i
            # quick quit for local nearest
            if dist_last < dist:
                break
            dist_last = dist
        return idx_min

    def calc_anchor_pos(self):
        anchor_dist = self.anchor_distance
        robot_pos = self.pose[:2]
        robot_ang = self.pose[2]
        trans,rot = get_transform(robot_pos[0], robot_pos[1], robot_ang)
        anchor_pt_b = np.array([anchor_dist, 0]).reshape((2,1))
        anchor_pt_g = rot @ anchor_pt_b + trans
        return anchor_pt_g.flatten()

    def calc_angle(self, pt_s, pt_e):
        assert pt_s.shape == pt_e.shape, "shape of two points should equal"
        pt_diff = pt_e - pt_s
        return np.arctan2(pt_diff[1], pt_diff[0])

    def find_lookahead_pt(self, pt, idx_start):
        range = self.traj.position.shape[0]
        traj_pos = self.traj.position
        traj_tan = self.traj.tangent_angle
        dist_lookahead = self.lookahead_distance
        res = pt
        for i in np.arange(idx_start, range, 1):
            dist = np.linalg.norm(traj_pos[i,:] - pt)
            if dist > dist_lookahead:
                res = traj_pos[i,:]
                # print("pp look from {} to {} in {} ".format(idx_start, i, range))
                break
            if range-1 == i and dist < dist_lookahead:
                """
                Extension cord of last two points
                and the last point should inside the lookahead circle
                ref: https://math.stackexchange.com/a/894518
                @ r^2 = (t(p2x-p1x) + p1x - cx)^2 + (t(p2y-p1y) + p1y - cy)^2
                @     = (deltax * t + p1cx)^2 + (deltay * t + p1cy)^2
                @  0  = (deltax^2 + deltay^2)*t^2 + 2(deltax*p1cx + deltay*p1cy)*t + (p1cx^2 + p1cy^2 - r^2)
                @  0  = ax^2 + bx + c
                @  res = (-b + sqrt(b^2 - 4ac)) / 2a
                """
                r = dist_lookahead
                c0 = pt.reshape((2,1))
                p1 = traj_pos[-1,:].reshape((2,1))     # last point, inside circle
                # cord_angle = self.calc_angle(traj_pos[-2,:], traj_pos[-1,:])
                cord_angle = traj_tan[-1]
                trans,rot = get_transform(p1[0], p1[1],cord_angle)
                p2 = rot @ np.array([2*r,0]).reshape((2,1)) + trans
                print("cord: r: {}, c0:{:.4f},{:.4f}, p1:{:.4f},{:.4f}, p2:{:.4f},{:.4f}, ang:{:.4f}".format(r, c0[0,0], c0[1,0], p1[0,0], p1[1,0], p2[0,0],p2[1,0], cord_angle))
                
                delta = p2 - p1
                p1c = p1 - c0
                a = delta[0]**2 + delta[1]**2
                b = 2*(delta.T @ p1c).flatten()
                c = p1c[0]**2 + p1c [1]**2 - r**2

                t = (np.sqrt(b**2 - 4*a*c) - b) / (2*a)
                resT = p1 + t * delta
                res = resT.flatten()

        return res

    def run(self):
        self.idx_near = self.find_nearest_idx(self.pose[:2])
        anchor_pt_g = self.calc_anchor_pos()
        idx_anchor = self.find_nearest_idx(anchor_pt_g)
        tps = self.traj.position
        # print("near: {:.4f},{:.4f}, anch: {:.4f},{:.4f}".format(tps[self.idx_near,0], tps[self.idx_near,1], tps[idx_anchor,0], tps[idx_anchor,1]))
        lookahead_pt_g = self.find_lookahead_pt(anchor_pt_g, idx_anchor)
        # print(anchor_pt_g, lookahead_pt_g)
        lookahead_angle = self.calc_angle(anchor_pt_g, lookahead_pt_g)
        angle_err = wrap_pi(lookahead_angle - self.pose[2])
        angle_err = angle_err.item()
        print("pose:{:.4f},{:.4f},{:.4f},{}, anchor: {:.4f},{:.4f},{}, look: {:.3f},{:.3f}, look ang: {:.3f}, ang err: {:.3f}".format(
           self.pose[0,0], self.pose[1,0], self.pose[2,0],self.idx_near, anchor_pt_g[0], anchor_pt_g[1], idx_anchor, lookahead_pt_g[0], lookahead_pt_g[1], lookahead_angle, angle_err
        ))
        if abs(angle_err) > 1:
            print("tan: {:.3f}".format(self.traj.tangent_angle[-1]))


        # feedforward
        v_last = self.cmd[0]
        w_last = self.cmd[1]
        v_ref = self.traj.vel[self.idx_near]
        k_ref = self.traj.curvature[self.idx_near]

        v_cur = v_ref
        if abs(v_ref - v_last) > self.max_acc*self.dt :
            v_cur = v_last + np.sign(v_ref-v_last) * self.max_acc*self.dt
        w_ff = v_cur * k_ref

        # p controller
        kp = 3.
        # print("INF: pp ctrl {:.3f} + {:.3f}, kappa: {:.3f}".format(kp*angle_err, w_ff, k_ref))
        w_cur = kp * angle_err + w_ff

        # k_cur = w_cur/v_cur
        # if abs(w_cur - w_last) > self.max_acc*self.dt :
        #     w_cur = w_last + np.sign(w_cur-w_last) * self.max_acc*self.dt
        #TODO: should constrain to vel


        self.cmd = [v_cur, w_cur]
        return True

    def get_cmd(self):
        return self.cmd



# ============ test =========================
if __name__ == "__main__":
    pp = PurePursuit()

    num_traj = 100
    position_nx2 = np.linspace(np.array([0.5,0.5]), np.array([5,5]), num_traj)
    tan = np.ones(num_traj) * np.deg2rad(45)
    kappa = np.zeros(num_traj)
    vel = np.ones(num_traj)
    pp.set_traj(position_nx2, tan, kappa, vel)
    pp.set_robot_pose(1,1,np.deg2rad(45))

    if pp.run():
        cmd = pp.get_cmd()
        print("cmd: [{:.3f}, {:.3f}]".format(cmd[0], cmd[1]))