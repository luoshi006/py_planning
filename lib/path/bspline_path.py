import numpy as np
import time
from scipy import interpolate
from scipy import integrate
from scipy import spatial

if __package__:
    from lib.collision_detect.poly_poly_intersection import poly_poly_intersection
    from lib.utils import get_transform

class BSplinePath2D:
    # default 2D cubic clamped uniform b-spline
    def __init__(self, ctrl_pts_2xn=[], dim=2, degree=3):
        self.dim = dim
        self.p = degree
        self.order = degree+1
        self.arclen = 0
        if len(ctrl_pts_2xn) > 0:
            assert ctrl_pts_2xn.shape[0] == 2
            self.ctrl_pts_2xn = ctrl_pts_2xn    # 2xn
            self.n = ctrl_pts_2xn.shape[1] - 1
            self.m = self.n + self.p + 1
            knots_inner = np.linspace(0, 1, self.n-1, endpoint=True)
            self.knots = np.append(np.append(np.zeros([self.p]), knots_inner), np.ones([self.p]))
            self.spl = interpolate.BSpline(self.knots, ctrl_pts_2xn.T, degree)
            self.spl_der1 = self.spl.derivative()
            self.spl_der2 = self.spl.derivative(2)
            self.arclen = self.arclength()
            self.bbox_leftdown = [np.min(ctrl_pts_2xn[0,:]), np.min(ctrl_pts_2xn[1,:])]
            self.bbox_rightup = [np.max(ctrl_pts_2xn[0,:]), np.max(ctrl_pts_2xn[1,:])]

    def eval_list(self, step_size=0.02):
        assert 0 < self.arclen, "the spline param is null!"
        sample_num = int(max(10, self.arclength() // step_size))    # at least 10 pts
        # xi, yi = interpolate.splev(np.linspace(0,1,sample_num), self.tck)
        xy_nx2 = self.spl(np.linspace(0,1,sample_num))
        return xy_nx2.T

    def eval_arclen(self, step_arclen=0.02, t0=0, t1=1):
        """interpolate the (u, s(u)), and calc u(s_eq)
            :return spline(u)
        """
        sample_factor = 2
        approx_len = self.arclen * (t1-t0)
        sample_num = int(max(10, approx_len // step_arclen))
        u_ref = np.linspace(t0, t1, sample_num*sample_factor)
        s_ref = self.arclength(u_ref)
        s_lin = np.linspace(s_ref[0], s_ref[-1], sample_num)
        s_interp = np.interp(s_lin, s_ref,u_ref)
        xy_nx2 = self.spl(s_interp)
        return xy_nx2.T

    def eval_arc2u(self, step_arclen=0.01, t0=0, t1=1):
        sample_factor = 2
        approx_len = self.arclen * (t1-t0)
        sample_num = int(max(10, approx_len // step_arclen))
        u_ref = np.linspace(t0, t1, sample_num*sample_factor)
        s_ref = self.arclength(u_ref)
        s_lin = np.linspace(s_ref[0], s_ref[-1], sample_num)
        s_interp = np.interp(s_lin, s_ref,u_ref)
        return s_interp
    def eval(self, u):
        u = np.atleast_1d(u)  # force to array
        return self.spl(u)

    def eval_angle(self, u, deg=False):
        u = np.atleast_1d(u)  # force to array
        dx,dy = self.eval_der(u)
        angle_rad = np.arctan2(dy,dx)
        if deg:
            return np.rad2deg(angle_rad)
        else:
            return angle_rad

    def eval_der(self, u, order=1):
        u = np.atleast_1d(u)  # force to array
        if 1 == order:
            dxy = self.spl_der1(u)
            dx = dxy[:, 0]
            dy = dxy[:, 1]
            return dx,dy
        elif 2 == order:
            ddxy = self.spl_der2(u)
            ddx = ddxy[:, 0]
            ddy = ddxy[:, 1]
            return ddx,ddy
        else:
            print("order now only support 0, 1; but now is {}".format(order))

    def arclength(self, t1=1, t0=0):
        # if 0 == t0 and 1 == t1 and self.arclen > 0:
        #     return self.arclen
        def derivate_s(u):
            dx,dy = self.eval_der(u)
            return np.hypot(dx,dy)

        t1 = np.atleast_1d(t1)
        if t1.size > 1 and t0 == 0:
            t0 = np.zeros_like(t1)
        else:
            t0 = np.atleast_1d(t0)
        #TODO: check [t0,t1] size equal
        # res = integrate.romberg(derivate_s, t0,t1, tol=1e-6, vec_func=True)
        res = np.array([integrate.romberg(derivate_s, t0i, t1i, tol=1e-5, vec_func=True) for t0i,t1i in zip(t0,t1)])
        return res.squeeze()

    def curvature(self, t):
        """Return the signed curvature at t for bspline"""
        dx, dy = self.eval_der(t)
        ddx, ddy = self.eval_der(t, 2)
        return (dx*ddy - dy*ddx) / np.power(dx*dx + dy*dy, 1.5)

    def convex_hull(self):
        assert self.p == 3, "only support cubic spline."
        hulls_idxs = []
        pts_4x2s = []
        for i in range(self.n-2):
            V_bs_2x4 = self.ctrl_pts_2xn[:, i:i + 4]
            # check collinear
            if np.linalg.matrix_rank(V_bs_2x4, tol=0.001) == 1:
                coll_idx = np.array([0,3])
                hulls_idxs.append(coll_idx)
            else:
                hull = spatial.ConvexHull(V_bs_2x4.T)
                hulls_idxs.append(hull.vertices)
            pts_4x2s.append(V_bs_2x4.T)
        return hulls_idxs, pts_4x2s

    def MINVO_hull(self):
        assert self.p == 3, "only support cubic uniform spline."
        #TODO: should check knots is clamped uniform
        hulls_idxs = []
        minvo_pts_4x2s = []
        for i in range(self.n-2):
            # calc transform matrix from b-spline to MINVO control points in one segment
            M_seg_4x4 = np.ones((4,4))
            if 0 == i:              # M_pos_bs2mv_seg0
                M_seg_4x4 = np.array(   [1.1023313949144333268037598827505,   0.34205724556666972091534262290224, -0.092730934245582874453361910127569, -0.032032766697130621302846975595457,
                                    -0.049683556253749178166501110354147,   0.65780347324677179710050722860615,   0.53053863760186903419935333658941,   0.21181027098212013015654520131648,
                                    -0.047309044211162346038612724896666,  0.015594436894155586093013710069499,    0.5051827557159349613158383363043,   0.63650059656260427054519368539331,
                                    -0.0053387944495217444854096022766043, -0.015455155707597083292181849856206,  0.057009540927778303009976212933907,   0.18372189915240558222286892942066]).reshape((4,4))
            elif 1 == i:            # M_pos_bs2mv_seg1
                M_seg_4x4 = np.array([  0.27558284872860833170093997068761,  0.085514311391667430228835655725561, -0.023182733561395718613340477531892, -0.0080081916742826553257117438988644,
                                    0.6099042761975865811763242163579,   0.63806904207840509091198555324809,   0.29959938009132258684985572472215,    0.12252106674808682651445224109921,
                                    0.11985166952332682033244282138185,   0.29187180223752445806795208227413,   0.66657381254229419731416328431806,    0.70176522577378930289881964199594,
                                    -0.0053387944495217444854096022766043, -0.015455155707597083292181849856206,  0.057009540927778303009976212933907,    0.18372189915240558222286892942066]).reshape((4,4))
            elif self.n-4 == i:     # M_pos_bs2mv_seg_last2
                M_seg_4x4 = np.array([  0.18372189915240569324517139193631,  0.057009540927778309948870116841135, -0.015455155707597145742226985021261, -0.0053387944495218164764338553140988,
                                    0.70176522577378952494342456702725,   0.66657381254229453038107067186502,   0.29187180223752412500104469472717,    0.11985166952332593215402312125661,
                                    0.1225210667480875342816304396365,   0.29959938009132280889446064975346,   0.63806904207840497988968309073243,    0.60990427619758624810941682881094,
                                    -0.0080081916742826154270717964323012, -0.023182733561395621468825822830695,  0.085514311391667444106623463540018,    0.27558284872860833170093997068761]).reshape((4,4))
            elif self.n-3 == i:     # M_pos_bs2mv_seg_last
                M_seg_4x4 = np.array([  0.18372189915240555446729331379174, 0.057009540927778309948870116841135, -0.015455155707597117986651369392348, -0.0053387944495218164764338553140988,
                                    0.63650059656260415952289122287766,   0.5051827557159349613158383363043,  0.015594436894155294659469745965907,  -0.047309044211162887272337229660479,
                                    0.21181027098212068526805751389475,  0.53053863760186914522165579910506,   0.65780347324677146403359984105919,  -0.049683556253749622255710960416764,
                                    -0.032032766697130461708287185729205, -0.09273093424558248587530329132278,   0.34205724556666977642649385416007,     1.1023313949144333268037598827505]).reshape((4,4))
            else:                   # M_pos_bs2mv_rest
                M_seg_4x4 = np.array([  0.18372189915240555446729331379174,  0.057009540927778309948870116841135, -0.015455155707597117986651369392348, -0.0053387944495218164764338553140988,
                                    0.70176522577378919187651717948029,   0.66657381254229419731416328431806,   0.29187180223752384744528853843804,    0.11985166952332582113172065874096,
                                    0.11985166952332682033244282138185,   0.29187180223752445806795208227413,   0.66657381254229419731416328431806,    0.70176522577378930289881964199594,
                                    -0.0053387944495217444854096022766043, -0.015455155707597083292181849856206,  0.057009540927778303009976212933907,    0.18372189915240558222286892942066]).reshape((4,4))
            V_bs_2x4 = self.ctrl_pts_2xn[:, i:i + 4]
            V_mv_2x4 = V_bs_2x4 @ M_seg_4x4

            # check collinear
            if np.linalg.matrix_rank(V_mv_2x4, tol=0.001) == 1:
                coll_idx = np.array([0,3])
                hulls_idxs.append(coll_idx)
            else:
                hull = spatial.ConvexHull(V_mv_2x4.T)
                hulls_idxs.append(hull.vertices)
            minvo_pts_4x2s.append(V_mv_2x4.T)
        return hulls_idxs, minvo_pts_4x2s

    def collision_check(self, obs_poly_nx2):
        hulls_idxs, hulls_vertex = self.MINVO_hull()
        for i in range(self.n - 2):
            pts = hulls_vertex[i]
            idx_closed = np.append(hulls_idxs[i], hulls_idxs[i][0])
            seg_poly = pts[idx_closed, :]
            if poly_poly_intersection(obs_poly_nx2, seg_poly):
                return True
        return False

    def distance_to(self, pts_1x2, err=0.01):
        #TODO: should be improved according to
            # https://stackoverflow.com/questions/2742610/closest-point-on-a-cubic-bezier-curve
            # https://github.com/qnzhou/nanospline/blob/main/include/nanospline/BSplineBase.h#L29
        xy_2xn = self.eval_arclen(err*2)
        dists = xy_2xn.T - pts_1x2
        norm = np.linalg.norm(dists, ord=2, axis=1, keepdims=True)
        return np.min(norm)

    def convex_hulls_of_curve(self, vehicle_contour_nx2_b):
        t_0 = time.time()
        hulls = []
        hulls_idxs, hulls_vertex = self.MINVO_hull()
        t_1 = time.time()
        t_poly = 0
        t_conv = 0
        if [] == vehicle_contour_nx2_b:
            for i in range(self.n - 2):
                polyi = hulls_vertex[i][hulls_idxs[i],:]
                hulls.append(polyi)
        else:
            for i in range(self.n -2):
                t_2 = time.time()
                # get polygon of ith segments
                polyi = hulls_vertex[i][hulls_idxs[i], :]
                # get segment of polygon, and convolution the segment endpoints and robot shape
                pts_conv_nx2s = []
                for j in range(polyi.shape[0]):
                    end_idx = j+1
                    if j+1 == polyi.shape[0]:
                        end_idx = 0     # get the closed polygon
                    end_pt = polyi[end_idx,:]
                    start_pt = polyi[j,:]
                    vec_diff_1x2 = end_pt - start_pt
                    angle_j2jp1_rad = np.arctan2(vec_diff_1x2[1], vec_diff_1x2[0])
                    # start point
                    trans, rot = get_transform(start_pt[0], start_pt[1], angle_j2jp1_rad)
                    pt_start_2xn = rot @ vehicle_contour_nx2_b.T + trans
                    # pts_conv_nx2 = np.vstack((pts_conv_nx2, pt_start_2xn.T))
                    pts_conv_nx2s.append(pt_start_2xn.T)
                    # end point
                    trans, rot = get_transform(end_pt[0], end_pt[1], angle_j2jp1_rad)
                    pt_end_2xn = rot @ vehicle_contour_nx2_b.T + trans
                    # pts_conv_nx2 = np.vstack((pts_conv_nx2, pt_end_2xn.T))
                    pts_conv_nx2s.append(pt_end_2xn.T)
                pts_conv_nx2 = np.array(pts_conv_nx2s)
                rows = pts_conv_nx2.shape[0]*pts_conv_nx2.shape[1]
                pts_conv_nx2 = pts_conv_nx2.reshape((rows,2))
                t_3 = time.time()
                hull = spatial.ConvexHull(pts_conv_nx2)
                hulls.append(pts_conv_nx2[hull.vertices,:])
                t_4 = time.time()
                t_poly = t_poly + (t_3-t_2)
                t_conv = t_conv + (t_4-t_3)
        # convex hull: minvo: 0.004, seg_poly: 0.001, convex hull: 0.003
        # print("convex hull: minvo: {:.3f}, seg_poly: {:.3f}, convex hull: {:.3f}".format(t_1-t_0, t_poly, t_conv))
        return hulls

# ============ test =========================
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon
    import sys
    import os
    #sys.path.append('..')
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from collision_detect.poly_poly_intersection import poly_poly_intersection
    from utils import get_transform

    wpt_list = np.array([
        [0., 0.11672268, 0.4539905 , 0.87690559, 1.40306285, 1.92224408, 2.4859] \
       ,[0., 0.48618496, 0.89100652, 1.21697847, 1.42527704, 1.59842976, 1.6793]])

    ctrl_pts = np.array([[0.    , 0.    , 0.071 , 0.4517, 0.8462, 1.425 , 1.872 , 2.3709,2.4859],
                         [0.    , 0.1389, 0.5183, 0.8948, 1.2484, 1.4133, 1.65  , 1.6016, 1.6793]])

    spl1 = BSplinePath2D(ctrl_pts)

    # show_pts1 = spl1.eval_list(0.1)
    show_pts1 = spl1.eval_arclen(0.01)
    hulls, ptss = spl1.convex_hull()
    hulls_minvo, pts_minvo = spl1.MINVO_hull()
    r = 0.1
    vehicle_poly_nx2 = np.array([  r,  r
                                , -r,  r
                                , -r, -r
                                ,  r, -r]).reshape((4, 2))
    collision_hulls = spl1.convex_hulls_of_curve(vehicle_poly_nx2)

    fig,ax1 = plt.subplots()
    # fig, (ax0, ax1) = plt.subplots(nrows=2, subplot_kw=dict(aspect='equal'))
    # spline
    # ax0.plot(show_pts0[0,:], show_pts0[1,:], 'k.-', linewidth='0.3', label='path')
    ax1.plot(show_pts1[0,:], show_pts1[1,:], 'k-', linewidth='0.3', label='path')
    # ax1.plot(wpt_list[0,:], wpt_list[1,:], 'r.', markersize=12, label='way pts')
    # ax1.plot(show_pts1[0,:], show_pts1[1,:], 'k.', label='path pts')
    ax1.plot(spl1.ctrl_pts_2xn[0,:], spl1.ctrl_pts_2xn[1,:], 'c.', label='ctrl pts')

    # # tail
    # u_nm3 = spl1.knots[spl1.n-3]
    # smp_u = np.linspace(u_nm3,1,100)
    # smp_pts = spl1.spl(smp_u)
    # ax1.plot(smp_pts[:,0], smp_pts[:,1], 'r', linewidth='0.8', label='tail')

    # hulls
    draw_hull = True
    if draw_hull:
        for i in range(spl1.n-2):
            # convex hull
            pts = ptss[i]
            idx_closed = np.append(hulls[i], hulls[i][0])
            poly = Polygon(pts[idx_closed, :], alpha=0.2, fc='r')
            if 0 == i:
                poly.set_label('convex hull')
            ax1.add_patch(poly)
            # ax0.add_patch(poly)
            # minvo
            pts_m = pts_minvo[i]
            idx_m_closed = np.append(hulls_minvo[i], hulls_minvo[i][0])
            poly_m = Polygon(pts_m[idx_m_closed,:], alpha=0.4, fc='g')
            if 0 == i:
                poly_m.set_label('MINVO hull')
            ax1.add_patch(poly_m)
            # ax1.add_patch(poly_m)
            # collision hull
            col_poly_nx2 = collision_hulls[i]
            col_poly_close_nx2 = np.vstack((col_poly_nx2, col_poly_nx2[0,:]))
            poly_col = Polygon(col_poly_close_nx2, alpha=0.4, fc = 'b')
            if 0 == i:
                poly_col.set_label('collision hull')
            ax1.add_patch(poly_col)

    plt.axis('equal')
    plt.legend()
    # plt.savefig('bspline_path_collision_hull.png', dpi=600)
    plt.show()
