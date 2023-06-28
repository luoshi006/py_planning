import numpy as np
from math import sin, cos

def scan_to_pointcloud(scan_in):
    N = len(scan_in['ranges'])
    ranges = np.array(scan_in['ranges'])
    eps = 1E-6
    ranges[ranges > scan_in['range_max'] - eps] = np.inf
    # for i in ranges:
    #     if ranges[i] > scan_in['range_max'] - eps:
    #         ranges[i] = np.inf
    angles = scan_in['angle_min'] + np.arange(N)*scan_in['angle_increment']
    cos_sin_map = np.array([np.cos(angles), np.sin(angles)])
    out_pts = ranges * cos_sin_map
    # delete inf pts
    out_pts = out_pts[:,~np.isinf(out_pts)[0,:]]
    return out_pts

def get_transform(x, y, psi):
    # trans_point = np.linalg.inv(rot) @ ( point - trans)
    # vertex = rot @ self.init_vertex + trans
    trans = np.array([x,y]).reshape((2,1))
    rot = np.array([[cos(psi), -sin(psi)], [sin(psi), cos(psi)]])
    return trans, rot

def wrap_pi(angle):
    # https://stackoverflow.com/a/32266181
    # wrap angle to (-pi, pi]
    return (( -angle + np.pi) % (2.0 * np.pi ) - np.pi) * -1.0
