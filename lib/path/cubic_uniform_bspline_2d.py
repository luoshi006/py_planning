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
    # wpt_list = [
    #     [0., 0.0188, 0.0749, 0.1673, 0.2944, 0.4539, 0.6092, 0.7832, 0.9746, 1.1818, 1.4030, 1.6045, 1.8144, 2.0318, 2.2559,
    #      2.4859] \
    #     ,
    #     [0., 0.1991, 0.3929, 0.5761, 0.7438, 0.8910, 1.0338, 1.1603, 1.2688, 1.3576, 1.4252, 1.5051, 1.5709, 1.6222, 1.6585,
    #      1.6793]]

    wpt_list = np.array([
        [0., 0.11672268, 0.4539905 , 0.87690559, 1.40306285, 1.92224408, 2.4859] \
       ,[0., 0.48618496, 0.89100652, 1.21697847, 1.42527704, 1.59842976, 1.6793]])

    delta_s = 1 # np.linalg.norm( np.array(wpt_list)[:,1] - np.array(wpt_list)[:,0])

    angle0 = 90
    rad0 = np.deg2rad(angle0)
    angle1 = 34.04
    rad1 = np.deg2rad(angle1)
    vel0 = np.array([[np.cos(rad0)],[np.sin(rad0)]]) * delta_s
    vel1 = np.array([[np.cos(rad1)],[np.sin(rad1)]]) * delta_s

    # solver_cubic_uniform_bspline_2d(np.array(wpt_list))
    solver_cubic_uniform_bspline_2d_v3(wpt_list, vel0, vel1, np.zeros((2,1)), np.zeros((2,1)))
    # test_tau = np.linspace(0,1,6)
    # knots = knots_quasi_uniform(4)
    # basis = spcol(test_tau, knots, 3)
    # print(basis)