import numpy as np
import matplotlib.pyplot as plt

def solver_cubic_uniform_bspline_2d(way_pts):
    """
    ref: https://github.com/dfki-asr/morphablegraphs/blob/master/morphablegraphs/constraints/spatial_constraints/splines/b_spline.py#L41
    :param way_pts: np.array([[x0, x1,...],[y0, y1,...]])
    :return: tck
    """
    debug = True
    if debug:
        fig, ax = plt.subplots()
        ax.plot(way_pts[0,:], way_pts[1,:], 'r.', markersize=2)
        plt.axis('equal')
        plt.show(block=False)

    p_ = 3  # degree
    outer_knots = p_ + 1
    n_ = way_pts.shape[1] - 1   # control points number
    m_ = n_ + p_ + 1
    num_knots = m_ + 1
    num_inner_knots = num_knots - (p_*2)
    knots_inner = np.linspace(0,1,num_inner_knots).tolist()
    knots_ = knots_inner[:1]*p_ + knots_inner + knots_inner[-1:]*p_

    u_sample_pts = np.linspace(0,1,n_+1)

    spline_basis = spcol(u_sample_pts, knots_, p_)
    spline_basis_inv = np.linalg.inv(spline_basis)

    ctrl_pts_x = spline_basis_inv @ way_pts[0,:]
    ctrl_pts_y = spline_basis_inv @ way_pts[1,:]
    if debug:
        print(ctrl_pts_x)
        print(ctrl_pts_y)
        ax.plot(ctrl_pts_x, ctrl_pts_y, 'b.')
        u_show = np.linspace(0,1,100)
        spl_show_basis = spcol(u_show, knots_, p_)
        show_x = spl_show_basis @ ctrl_pts_x
        show_y = spl_show_basis @ ctrl_pts_y
        ax.plot(show_x, show_y, 'g-', linewidth='1')
    return np.vstack((ctrl_pts_x, ctrl_pts_y)).T

def solver_cubic_uniform_bspline_2d_v2(way_pts, vel0=[], vel1=[]):
    """施法中, 计算机辅助几何设计与非均匀有理B样条（修订版） P304 """
    p_ = 3  # degree
    n_ = way_pts.shape[1] + 1  # control points idx
    dim_ = way_pts.shape[0]
    knots_inner = np.linspace(0,1, n_-1).tolist()
    knots_ = knots_inner[:1] * p_ + knots_inner + knots_inner[-1:] * p_

    if len(vel0) == 0:
        diff_10 = way_pts[:,1] - way_pts[:,0]
        vel0 = (diff_10) / np.linalg.norm(diff_10)
    if len(vel1) == 0:
        diff_n = way_pts[:,-1] - way_pts[:,-2]
        vel1 = diff_n / np.linalg.norm(diff_n)

    # A*D = E
    A = np.zeros((n_-1,n_-1))
    E = np.zeros((n_-1,dim_))
    Q = way_pts.T

    # start point
    A[0,0] = 1  # b1
    A[0,1] = 0  # c1
    A[0,2] = 0  # a1
    E[0,:]   = Q[0,:] + delta(knots_,3) * vel0.T / 3.0
    # end point
    A[-1,-1] = 1    # b_{n-1}
    A[-1,-2] = 0    # a_{n-1}
    A[-1,-3] = 0    # c_{n-1}
    E[-1,:] = Q[-1,:] - delta(knots_,n_) * vel1.T / 3.0
    for i in np.arange(2,n_-2+1):
        idx = i-1
        ai = (delta(knots_,i+2)**2) / (delta(knots_,i)+delta(knots_,i+1)+delta(knots_,i+2))
        bi = ((delta(knots_,i+2)*(delta(knots_,i)+delta(knots_,i+1)))/(delta(knots_,i)+delta(knots_,i+1)+delta(knots_,i+2))) + ((delta(knots_,i+1)*(delta(knots_,i+2)+delta(knots_,i+3)))/(delta(knots_,i+1)+delta(knots_,i+2)+delta(knots_,i+3)))
        ci = (delta(knots_,i+1)**2) / (delta(knots_,i+1)+delta(knots_,i+2)+delta(knots_,i+3))
        ei = (delta(knots_,i+1)+delta(knots_,i+2)) * Q[i-1,:]

        A[idx,idx] = bi
        A[idx,idx-1] = ai
        A[idx,idx+1] = ci
        E[idx,:] = ei
    # solve the ctrl pts
    D_inner = np.linalg.inv(A) @ E
    D = np.vstack((Q[0,:],D_inner,Q[-1,:]))
    ctrl_pts_x = D[:,0]
    ctrl_pts_y = D[:,1]

    if __name__ == "__main__":
        fig, ax = plt.subplots()
        ax.plot(way_pts[0,:], way_pts[1,:], 'r*', markersize=9)
        ax.plot(ctrl_pts_x, ctrl_pts_y, 'k.')
        u_show = np.linspace(0,1,100)
        spl_show_basis = spcol(u_show, knots_, p_)
        show_x = spl_show_basis @ ctrl_pts_x
        show_y = spl_show_basis @ ctrl_pts_y
        ax.plot(show_x, show_y, 'g-', linewidth='1')
        plt.axis('equal')
        # plt.savefig('test.png', dpi=300)
        plt.show()
    return D.T

def solver_cubic_uniform_bspline_2d_v3(way_pts_2xn, vel0=[], vel1=[], acc0=[], acc1=[]):
    acc_constrain = 0
    flg_acc1 = False
    if len(vel0) == 0:
        diff_10 = way_pts_2xn[:, 1] - way_pts_2xn[:, 0]
        vel0 = (diff_10) / np.linalg.norm(diff_10)
    if len(vel1) == 0:
        diff_n = way_pts_2xn[:, -1] - way_pts_2xn[:, -2]
        vel1 = diff_n / np.linalg.norm(diff_n)
    if len(acc0) == 0:
        acc0 = np.zeros((2, 1))
    acc_constrain = acc_constrain +1
    if len(acc1) != 0:
        flg_acc1 = True
        acc_constrain = acc_constrain +1

    p_ = 3
    n_ = way_pts_2xn.shape[1] + 1 + acc_constrain
    dim_ = way_pts_2xn.shape[0]
    knots_inner = np.linspace(0,1,n_-1).tolist()
    knots_ = knots_inner[:1]*p_ + knots_inner + knots_inner[-1:]*p_

    # A*D = E
    A = np.zeros((n_-1,n_-1))
    E = np.zeros((n_-1,dim_))
    Q = way_pts_2xn.T

    # start point vel
    A[0,0] = 1  # b1
    E[0,:]   = Q[0,:] + delta(knots_,3) * vel0.T / 3.0
    # end point vel
    A[-1,-1] = 1    # b_{n-1}
    E[-1,:] = Q[-1,:] - delta(knots_,n_) * vel1.T / 3.0
    # start point acc
    #https://pages.mtu.edu/~shene/COURSES/cs3621/NOTES/spline/B-spline/bspline-derv.html
    A[1,0] = -6
    A[1,1] = 6
    E[1,:] = 3*(knots_[5]-knots_[2])*vel0.T - (knots_[5]-knots_[2])*(acc0.T*(knots_[2]-knots_[4]) + vel0.T)
    # end point acc
    if flg_acc1:
        A[-2,-1] = -6
        A[-2,-2] = 6
        E[-2,:] = -3*(knots_[n_+2]-knots_[n_-1])*vel1.T + (knots_[n_+2]-knots_[n_-1])*(acc1.T*(knots_[n_+2]-knots_[n_]) + vel1.T)
    for i in np.arange(3,way_pts_2xn.shape[1]+1):
        idx = i-1
        ai = (delta(knots_,i+2)**2) / (delta(knots_,i)+delta(knots_,i+1)+delta(knots_,i+2))
        bi = ((delta(knots_,i+2)*(delta(knots_,i)+delta(knots_,i+1)))/(delta(knots_,i)+delta(knots_,i+1)+delta(knots_,i+2))) + ((delta(knots_,i+1)*(delta(knots_,i+2)+delta(knots_,i+3)))/(delta(knots_,i+1)+delta(knots_,i+2)+delta(knots_,i+3)))
        ci = (delta(knots_,i+1)**2) / (delta(knots_,i+1)+delta(knots_,i+2)+delta(knots_,i+3))
        ei = (delta(knots_,i+1)+delta(knots_,i+2)) * Q[i-2,:]

        A[idx,idx] = bi
        A[idx,idx-1] = ai
        A[idx,idx+1] = ci
        E[idx,:] = ei
    # solve the ctrl pts
    D_inner = np.linalg.inv(A) @ E
    D = np.vstack((Q[0,:],D_inner,Q[-1,:]))
    ctrl_pts_x = D[:,0]
    ctrl_pts_y = D[:,1]

    if __name__ == "__main__":
        fig, ax = plt.subplots()
        ax.plot(way_pts_2xn[0,:], way_pts_2xn[1,:], 'r*', markersize=9)
        ax.plot(ctrl_pts_x,ctrl_pts_y, 'k.',markersize=4)
        # ax.plot(ctrl_pts_x, ctrl_pts_y, 'b.')
        u_show = np.linspace(0,1,100)
        spl_show_basis = spcol(u_show, knots_, p_)
        show_x = spl_show_basis @ ctrl_pts_x
        show_y = spl_show_basis @ ctrl_pts_y
        ax.plot(show_x, show_y, 'g-', linewidth='1')
        plt.axis('equal')
        plt.show()
    return D.T


def solver_cubic_uniform_bspline_2d_v4(way_pts_2xn, way_angles, weight_smooth=0.1):
    import osqp
    from scipy import sparse
    assert weight_smooth <= 1 and weight_smooth >= 0, "weight_smooth should in [0,1]"

    def wrap_pi(angle):
        # wrap angle to (-pi, pi]
        return (( -angle + np.pi) % (2.0 * np.pi ) - np.pi) * -1.0    # calc chord for knots
    p_ = 3
    n_ = way_pts_2xn.shape[1] +2 -1
    dim_ = way_pts_2xn.shape[0]

    knots_inner = np.linspace(0,1,n_-1).tolist()
    knots_ = knots_inner[:1]*p_ + knots_inner + knots_inner[-1:]*p_

    # calculate basis matrix for break points
    MB = spcol(np.array(knots_inner), np.array(knots_), p_)

    # calculate derivative basis matrix
    # https://pages.mtu.edu/~shene/COURSES/cs3621/NOTES/spline/B-spline/bspline-derv.html
    # https://github.com/lewisli/PFA-CFCA/blob/master/thirdparty/fda_matlab/notneeded/bspline.m
    Q2P = np.zeros((n_,n_+1))
    for i in np.arange(n_):
        a = p_/(knots_[i+p_+1] - knots_[i+1])
        Q2P[i,i] = -1*a
        Q2P[i,i+1] = 1*a
    MBV = spcol(np.array(knots_inner), np.array(knots_[1:-1]), p_-1)
    MBVV = MBV @ Q2P
    A = np.vstack((MB, MBVV[0,:]))
    A = np.vstack((A, MBVV[-1,:]))

    vel_dir = [way_angles[0], way_angles[-1]]
    v = np.array([np.cos(vel_dir), np.sin(vel_dir)]).T
    B = np.vstack((way_pts_2xn.T, v))

    ctrl_pts_nx2 = np.linalg.inv(A) @ B

    # smooth
    # 固定两侧端点各两个控制点，优化变量 2*(n_+1-4)
    # minimize        0.5 x' P x + q' x
    # subject to      l <= A x <= u

    # cost_smooth
    # sum{ ||2P_k - P_{k-1} - P_{k+1}||^2 }
    # 4*p_k**2 - 4*p_k*p_km1 - 4*p_k*p_kp1 + p_km1**2 + 2*p_km1*p_kp1 + p_kp1**2
    # 对长度不均匀的采样点，会引起波动，参考 case 2

    # cost_deviation
    # MB 去掉头尾各两行，计算 L2 范数
    # p_k**2 - 2*p_k*p_kref + p_kref**2

    X = np.hstack((ctrl_pts_nx2[:,0], ctrl_pts_nx2[:,1]))   # [x0, x1,..., xn, y0, y1,..., yn]
    pt_ref = np.hstack((way_pts_2xn[0,:],way_pts_2xn[1,:]))
    c_off = n_+1
    w_off = way_pts_2xn.shape[1]
    P = np.zeros(((n_+1)*dim_, (n_+1)*dim_))
    Q = np.zeros((n_+1)*dim_)

    A = np.identity((n_+1)*dim_)
    win_size = 0.5
    delta = np.ones((n_+1)*dim_)*win_size
    delta[0:2] = 0
    delta[n_-1:n_+1] = 0
    delta[c_off:c_off+2] = 0
    delta[n_-1+c_off:n_+1+c_off] = 0

    lb = X - delta
    ub = X + delta
    # ub = np.ones_like(lb)*2   # test for max value

    w_smooth = weight_smooth
    w_deviation = 1-w_smooth
    err_range_sq = 0.1**2
    eps_proj = 0.5**2 * np.sin(np.deg2rad(1))
    idx_half_constraint = []

    for k in np.arange(1, n_):
        # check collinear
        w_smooth = weight_smooth
        if k < n_-2:

            # slack, if err too large, ignore the deviation.
            err_sq = (X[k+1] - pt_ref[k])**2 + (X[k+1+c_off] - pt_ref[k+w_off])**2
            if err_sq > err_range_sq:
                w_deviation = 0.0
            else:
                w_deviation = 1-w_smooth

            # adjust smooth weight
            p1 = np.array([pt_ref[k-1], pt_ref[k-1+w_off]])
            p2 = np.array([pt_ref[k], pt_ref[k+w_off]])
            p3 = np.array([pt_ref[k+1], pt_ref[k+1+w_off]])
            v1 = p2-p1
            v2 = p3-p2
            proj_v12 = v1.dot(v2)
            # 平滑代价微调
            # if np.abs(proj_v12) < eps_proj:
            #     w_smooth *= 0.1
            colin_v12 = np.cross(v1, v2).item()
            if np.abs(colin_v12) < eps_proj:
                w_smooth *= 1
                w_deviation *= 2
                print("collinear: w_s {}, w_dev {}".format(w_smooth, w_deviation))

        # calc smooth
        P[k-1,k-1] += 1 * w_smooth
        P[k-1,k]   += -2 * w_smooth
        P[k-1,k+1] += 1 * w_smooth

        P[k,k-1] += -2 * w_smooth
        P[k,k]   += 4 * w_smooth
        P[k,k+1] += -2 * w_smooth

        P[k+1,k-1] += 1 * w_smooth
        P[k+1,k]   += -2 * w_smooth
        P[k+1,k+1] += 1 * w_smooth

        P[k-1+c_off,k-1+c_off] += 1 * w_smooth
        P[k-1+c_off,k+c_off]   += -2 * w_smooth
        P[k-1+c_off,k+1+c_off] += 1 * w_smooth

        P[k+c_off,k-1+c_off] += -2 * w_smooth
        P[k+c_off,k+c_off]   += 4 * w_smooth
        P[k+c_off,k+1+c_off] += -2 * w_smooth

        P[k+1+c_off,k-1+c_off] += 1 * w_smooth
        P[k+1+c_off,k+c_off]   += -2 * w_smooth
        P[k+1+c_off,k+1+c_off] += 1 * w_smooth

        # calc deviation
        if k<n_-2:

            # k in [1,7] -> get m1-3
            #   m1**2*p_k**2   + 2*m1*m2*p_k*p_kp1   + 2*m1*m3*p_k*p_kp2 - 2*m1*p_kref*p_k
            # + m2**2*p_kp1**2 + 2*m2*m3*p_kp1*p_kp2 - 2*m2*p_kref*p_kp1
            # + m3**2*p_kp2**2 - 2*m3*p_kref*p_kp2   + p_kref**2
            m1 = MB[k,k]
            m2 = MB[k,k+1]
            m3 = MB[k,k+2]
            P[k,k]   += m1**2 * w_deviation
            P[k,k+1] += m1*m2 * w_deviation
            P[k,k+2] += m1*m3 * w_deviation
            P[k+1,k]   += m1*m2 * w_deviation
            P[k+1,k+1] += m2**2 * w_deviation
            P[k+1,k+2] += m2*m3 * w_deviation
            P[k+2,k]   += m1*m3 * w_deviation
            P[k+2,k+1] += m2*m3 * w_deviation
            P[k+2,k+2] += m3**2 * w_deviation

            P[k+c_off,k+c_off]   += m1**2 * w_deviation
            P[k+c_off,k+c_off+1] += m1*m2 * w_deviation
            P[k+c_off,k+c_off+2] += m1*m3 * w_deviation
            P[k+1+c_off,k+c_off]   += m1*m2 * w_deviation
            P[k+1+c_off,k+c_off+1] += m2**2 * w_deviation
            P[k+1+c_off,k+c_off+2] += m2*m3 * w_deviation
            P[k+2+c_off,k+c_off]   += m1*m3 * w_deviation
            P[k+2+c_off,k+c_off+1] += m2*m3 * w_deviation
            P[k+2+c_off,k+c_off+2] += m3**2 * w_deviation

            Q[k]   += -1*m1*pt_ref[k] * w_deviation
            Q[k+1] += -1*m2*pt_ref[k] * w_deviation
            Q[k+2] += -1*m3*pt_ref[k] * w_deviation

            Q[k+c_off]   += -1*m1*pt_ref[k+w_off] * w_deviation
            Q[k+c_off+1] += -1*m2*pt_ref[k+w_off] * w_deviation
            Q[k+c_off+2] += -1*m3*pt_ref[k+w_off] * w_deviation

            # constraint half plane
            pt_k = np.array([pt_ref[k], pt_ref[k+w_off]])
            # look forward
            pt_kp1 = np.array([pt_ref[k+1], pt_ref[k+w_off+1]])
            pt_kp2 = pt_kp1 ## XXX
            if n_-3 != k:
                pt_kp2 = np.array([pt_ref[k+2], pt_ref[k+w_off+2]])

            p1 = pt_kp1 - pt_k
            p2 = pt_kp2 - pt_k
            angle_0 = way_angles[k]
            angle_1 = np.arctan2(p1[1], p1[0])
            angle_2 = np.arctan2(p2[1], p2[0])

            delta_1 = wrap_pi(angle_1 - angle_0)
            delta_2 = wrap_pi(angle_2 - angle_0)
            eps = np.deg2rad(0.5)
            flg_constraint_forward = False
            flg_constraint_backward = False
            if abs(delta_1) > eps:
                if delta_1 > 0 and delta_2 > 0 :
                    angle_n = angle_0 + np.pi*0.5
                    flg_constraint_forward = True
                elif delta_1 < 0 and delta_2 < 0 :
                    angle_n = angle_0 - np.pi*0.5
                    flg_constraint_forward = True
            elif abs(delta_2) > eps:    # delta1 == 0, delta2 != 0
                if delta_2 > 0:
                    angle_n = angle_0 + np.pi*0.5
                    flg_constraint_forward = True
                else:
                    angle_n = angle_0 - np.pi*0.5
                    flg_constraint_forward = True
            # look backward
            pt_km1 = np.array([pt_ref[k-1], pt_ref[k+w_off-1]])
            pt_km2 = pt_km1 ## XXX
            if 1 != k:
                pt_km2 = np.array([pt_ref[k-2], pt_ref[k+w_off-2]])

            pm1 = pt_k - pt_km1
            pm2 = pt_k - pt_km2
            angle_m1 = np.arctan2(pm1[1], pm1[0])
            angle_m2 = np.arctan2(pm2[1], pm2[0])

            delta_m1 = wrap_pi(angle_m1 - angle_0)
            delta_m2 = wrap_pi(angle_m2 - angle_0)
            if abs(delta_m1) > eps:
                if delta_m1 > 0 and delta_m2 > 0 :
                    angle_n = angle_0 - np.pi*0.5
                    flg_constraint_backward = True
                elif delta_m1 < 0 and delta_m2 < 0 :
                    angle_n = angle_0 + np.pi*0.5
                    flg_constraint_backward = True
            elif abs(delta_m2) > eps:    # delta1 == 0, delta2 != 0
                if delta_m2 > 0:
                    angle_n = angle_0 - np.pi*0.5
                    flg_constraint_backward = True
                else:
                    angle_n = angle_0 + np.pi*0.5
                    flg_constraint_backward = True

            if flg_constraint_forward:
                nx = np.cos(angle_n)
                ny = np.sin(angle_n)
                ap = np.zeros((1,(n_+1)*dim_))
                # 注掉的代码是约束曲线实际位置的，改用约束控制点
                # ap[0,k] = m1 * nx
                # ap[0,k+1] = m2 * nx
                # ap[0,k+2] = m3 * nx
                # ap[0,k+c_off] = m1*ny
                # ap[0,k+c_off+1] = m2*ny
                # ap[0,k+c_off+2] = m3*ny
                ap[0,k+1] = nx
                ap[0,k+1+c_off] = ny
                dk = -1 * (nx*pt_k[0] + ny*pt_k[1])
                lbp = -dk
                ubp = -dk + 100
                A = np.vstack((A, ap))
                lb = np.hstack((lb, lbp))
                ub = np.hstack((ub, ubp))
                idx_half_constraint.append(k)
                # 前探，往后扩散一下
                ap1 = np.zeros((1,(n_+1)*dim_))
                ap1[0,k+0] = nx
                ap1[0,k+0+c_off] = ny
                dk = -1 * (nx*pt_km1[0] + ny*pt_km1[1])
                lbp = -dk
                ubp = -dk + 100
                A = np.vstack((A, ap1))
                lb = np.hstack((lb, lbp))
                ub = np.hstack((ub, ubp))
                idx_half_constraint.append(k-1)
                print("half plane: {},({},{}), ({}, {})".format(k,pt_ref[k], pt_ref[k+w_off], nx, ny))
                print("constraint: {} < {} < {}".format(lbp, ap@X.T, ubp))
                # Xt = X.copy()
                # Xr = X.copy()
                # Xt[k] += 1
                # Xr[k] -= 1
                # print("test constraint dx: {}, {}".format(ap@Xt.T, ap@Xr.T))
                # Xt[k] -= 1
                # Xr[k] += 1
                # Xt[k+c_off] += 1
                # Xr[k+c_off] -= 1
                # print("test constraint dy: {}, {}".format(ap@Xt.T, ap@Xr.T))
                # idx_half_constraint.append(k)
            if flg_constraint_backward:
                nx = np.cos(angle_n)
                ny = np.sin(angle_n)
                ap = np.zeros((1,(n_+1)*dim_))

                ap[0,k+1] = nx
                ap[0,k+1+c_off] = ny
                dk = -1 * (nx*pt_k[0] + ny*pt_k[1])
                lbp = -dk
                ubp = -dk + 100
                A = np.vstack((A, ap))
                lb = np.hstack((lb, lbp))
                ub = np.hstack((ub, ubp))
                idx_half_constraint.append(k)

                # 往前扩散一下
                ap2 = np.zeros((1,(n_+1)*dim_))
                ap2[0,k+2] = nx
                ap2[0,k+2+c_off] = ny
                dk = -1 * (nx*pt_kp1[0] + ny*pt_kp1[1])
                lbp = -dk
                ubp = -dk + 100
                A = np.vstack((A, ap2))
                lb = np.hstack((lb, lbp))
                ub = np.hstack((ub, ubp))
                idx_half_constraint.append(k+1)
                print("half plane: {},({},{}), ({}, {})".format(k,pt_ref[k], pt_ref[k+w_off], nx, ny))
                print("constraint: {} < {} < {}".format(lbp, ap@X.T, ubp))


    Ps = sparse.csc_matrix(P)
    As = sparse.csc_matrix(A)
    problem = osqp.OSQP()

    problem.setup(Ps, Q, As, lb, ub)
    res = problem.solve()

    ctrl_pts_new_nx2 = res.x.reshape(dim_,n_+1).T
    ctrl_pts_new_nx2[:2,:] = ctrl_pts_nx2[:2,:]
    ctrl_pts_new_nx2[-2:,:] = ctrl_pts_nx2[-2:,:]

    if __name__ == "__main__":
        fig,ax = plt.subplots()
        ax.plot(way_pts_2xn[0,:], way_pts_2xn[1,:],'r*', markersize=9, label='way pts')
        ax.plot(ctrl_pts_nx2[:,0], ctrl_pts_nx2[:,1], 'k.', markersize=4, label='ctrl pts Interpolate')
        ax.plot(ctrl_pts_new_nx2[:,0], ctrl_pts_new_nx2[:,1], 'b1', markersize=6, label='ctrl pts OSQP half plane')
        u_show = np.linspace(0,1,100)
        spl_show_basis = spcol(u_show, knots_, p_)
        show_pts_nx2 = spl_show_basis @ ctrl_pts_nx2
        ax.plot(show_pts_nx2[:,0], show_pts_nx2[:,1], 'g-', linewidth='1', label='spline Interpolate')
        show_pts_new_nx2 = spl_show_basis @ ctrl_pts_new_nx2
        ax.plot(show_pts_new_nx2[:,0], show_pts_new_nx2[:,1], 'm-', linewidth='1', label='spline OSQP half plane')
        # constraint
        flg_lb = True
        for idxx in idx_half_constraint:
            circle1 = plt.Circle((way_pts_2xn[0,idxx], way_pts_2xn[1,idxx]), 0.1, color='g', alpha=0.1)
            if flg_lb:
                circle1 = plt.Circle((way_pts_2xn[0,idxx], way_pts_2xn[1,idxx]), 0.1, color='g', alpha=0.1, label='half plane constraint pts')
                flg_lb = False
            ax.add_patch(circle1)
        plt.axis('equal')
        plt.legend()
        plt.grid()
        # plt.title('Interpolate')
        plt.savefig('waypoints_OSQP_weight_constraint.png', dpi=300)
        plt.show()

    return ctrl_pts_new_nx2.T

def delta(knots, i):
    return knots[i+1]-knots[i]

def spcol(tau, knots, degree):
    """
    :param tau:
    :param knots:
    :param degree:
    :return:
            colmat: m x n matrix
    """
    columns = len(knots) - degree - 1
    colmat = np.nan*np.ones((len(tau), columns))
    for i in range(columns):
        # evaluates the ith spline basis given by knots on points in tau
        colmat[:,i] = np.array([cox_deboor(knots, i, degree, u) for u in tau])
    colmat[-1,-1] = 1
    return  colmat

def cox_deboor(knots, i, k, u):
    if (0 == k):
        return ((u-knots[i]>=0) & (u-knots[i+1]<0)).astype(int)

    denom1 = knots[i+k] - knots[i]
    term1 = 0
    if denom1 > 0:
        term1 = ((u-knots[i])/denom1) * cox_deboor(knots, i, k-1, u)
    denom2 = knots[i+k+1] - knots[i+1]
    term2 = 0
    if denom2 > 0:
        term2 = ((knots[i+k+1]-u)/denom2) * cox_deboor(knots, i+1, k-1, u)

    return term1 + term2



def knots_quasi_uniform(ctrl_pts_num, degree=3):
    n = ctrl_pts_num - 1
    m = n+degree+1
    num_inner = n-1
    knots_inner = np.linspace(0, 1, num_inner, endpoint=True)
    knots_all = np.append(np.append(np.zeros([degree]), knots_inner), np.ones([degree]))
    return knots_all

# test
if __name__ == "__main__":

    test_case = 1

    if test_case == 0:

        wpt_list = np.array([
            [0., 0.11672268, 0.4539905 , 0.87690559, 1.40306285, 1.92224408, 2.4859] \
        ,[0., 0.48618496, 0.89100652, 1.21697847, 1.42527704, 1.59842976, 1.6793]])
        wpt_angle = np.array([1.5707, 1.3352, 1.0996, 0.9464, 0.79325, 0.69368, 0.59411])

        ctrl_pts = solver_cubic_uniform_bspline_2d_v4(wpt_list, wpt_angle, 0.)
    elif test_case == 1:
        wpt_list = np.array([[ 0, 0.5, 1, 1.5, 2, 2,   2,   2, 2]
                            ,[0, 0,   0, 0,   0, 0.5, 1, 1.5, 2]])
        psi = np.pi/2
        wpt_angle = np.array([ 0, 0,   0, 0,   0, psi, psi, psi, psi])

        # ctrl_pts = solver_cubic_uniform_bspline_2d_v2(wpt_list)
        ctrl_pts = solver_cubic_uniform_bspline_2d_v4(wpt_list, wpt_angle)
    elif test_case == 2:
        wpt_list = np.array([[  7.52523248,   8.00700005,   8.41083938,   8.90647312,   9.46460853,
                                9.90573155,  10.27536637,  10.73613238,  11.30161159,  11.83311605,
                                12.24588121,  12.6636705 ],
                            [139.59676301, 139.59987533, 139.59917138, 139.59629827, 139.59673211,
                                139.59949598, 139.59896431, 139.59643679, 139.59589325, 139.5974165,
                                139.59978395, 139.59833656]])
        wpt_angle = np.array([ 0.00646011, -0.00174314, -0.00579678,  0.00077732,  0.00626543, -0.00143834,
                            -0.00548544, -0.00096119,  0.00286591,  0.00573552, -0.00346439, -0.00346439])
        ctrl_pts = solver_cubic_uniform_bspline_2d_v4(wpt_list, wpt_angle)