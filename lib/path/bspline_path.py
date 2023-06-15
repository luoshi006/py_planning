import numpy as np
from scipy import interpolate
from scipy import integrate
from scipy import spatial

from ..point_in_poly import point_in_polygon
from ..poly_poly_intersection import poly_poly_intersection

class BSplinePath2D:
    # default 2D cubic clamped uniform b-spline
    def __init__(self, ctrl_pts=[], dim=2, degree=3):
        self.dim = dim
        self.p = degree
        self.order = degree+1
        self.arclen = 0
        if len(ctrl_pts) > 0:
            assert ctrl_pts.shape[0] == 2
            self.ctrl_pts = ctrl_pts    # 2xn
            self.n = ctrl_pts.shape[1]-1
            self.m = self.n + self.p + 1
            knots_inner = np.linspace(0, 1, self.n-1, endpoint=True)
            self.knots = np.append(np.append(np.zeros([self.p]), knots_inner), np.ones([self.p]))
            self.spl = interpolate.BSpline(self.knots, ctrl_pts.T, degree)
            self.spl_der1 = self.spl.derivative()
            self.spl_der2 = self.spl.derivative(2)
            self.arclength()

    def eval_list(self, step_size=0.02):
        #TODO: arc parameter the spline
        assert 0 < self.arclen, "the spline param is null!"
        sample_num = int(max(10, self.arclength() // step_size))    # at least 10 pts
        # xi, yi = interpolate.splev(np.linspace(0,1,sample_num), self.tck)
        xy = self.spl(np.linspace(0,1,sample_num))
        return xy.T

    def eval(self, u):
        if u<0 or u>1:
            print("u should be in [0,1]")
        return self.spl(u)

    def eval_der(self, u, order=1):
        if u<0 or u>1:
            print("u should be in [0,1]")
        if 1 == order:
            return self.spl_der1(u)
        elif 2 == order:
            return self.spl_der2(u)
        else:
            print("order now only support 0, 1; but now is {}".format(order))

    def arclength(self, t0=0, t1=1):
        if 0 == t0 and 1 == t1 and self.arclen > 0:
            return self.arclen
        def derivate_s(u):
            u = np.atleast_1d(u)   # bug! the u sometimes is [0.5]
            dxy = self.spl_der1(u)
            dx = dxy[:,0]
            dy = dxy[:,1]
            return np.hypot(dx,dy)
        res = integrate.romberg(derivate_s, t0,t1, tol=1e-6, vec_func=True)
        if 0 == t0 and 1 == t1:
            self.arclen = res
        return res

    def curvature(self, t):
        """Return the signed curvature at t for bspline"""
        t = np.atleast_1d(t)    # force to array
        dxy = self.spl_der1(t)
        ddxy = self.spl_der2(t)
        dx = dxy[:,0]
        dy = dxy[:,1]
        ddx = ddxy[:,0]
        ddy = ddxy[:,1]
        return (dx*ddy - dy*ddx) / np.power(dx*dx + dy*dy, 1.5)

    def convex_hull(self):
        assert self.p == 3, "only support cubic spline."
        hulls_idxs = []
        pts_4x2s = []
        for i in range(self.n-2):
            # check collinear
            V_bs_2x4 = self.ctrl_pts[:,i:i+4]
            if np.linalg.matrix_rank(V_bs_2x4, tol=0.001) == 1:
                coll_idx = np.array([0,3])
                hulls_idxs.append(coll_idx)
            else:
                hull = spatial.ConvexHull(V_bs_2x4.T)
                hulls_idxs.append(hull.vertices)
            pts_4x2s.append(V_bs_2x4)
        return hulls_idxs

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
            V_bs_2x4 = self.ctrl_pts[:,i:i+4]
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



# ============ test =========================
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon
    wpt_list = [
        [0., 0.0188, 0.0749, 0.1673, 0.2944, 0.4539, 0.6092, 0.7832, 0.9746, 1.1818, 1.4030, 1.6045, 1.8144, 2.0318, 2.2559, 2.4859] \
       ,[0., 0.1991, 0.3929, 0.5761, 0.7438, 0.8910, 1.0338, 1.1603, 1.2688, 1.3576, 1.4252, 1.5051, 1.5709, 1.6222, 1.6585, 1.6793]]

    spl1 = BSplinePath2D(np.array(wpt_list))
    show_pts = spl1.eval_list()
    hulls = spl1.convex_hull()
    hulls_minvo, pts_minvo = spl1.MINVO_hull()

    fig, ax = plt.subplots()
    # spline
    ax.plot(show_pts[0,:], show_pts[1,:],'k', linewidth='0.3', label='path')
    # hulls
    for i in range(spl1.n-2):
        # convex hull
        pts = spl1.ctrl_pts[:,i:i+4].T
        idx_closed = np.append(hulls[i].vertices, hulls[i].vertices[0])
        poly = Polygon(pts[idx_closed, :], alpha=0.2, fc='r')
        if 0 == i:
            poly.set_label('convex hull')
        ax.add_patch(poly)
        # minvo
        pts_m = pts_minvo[i]
        idx_m_closed = np.append(hulls_minvo[i].vertices, hulls_minvo[i].vertices[0])
        poly_m = Polygon(pts_m[idx_m_closed,:], alpha=0.4, fc='g')
        if 0 == i:
            poly_m.set_label('MINVO hull')
        ax.add_patch(poly_m)


    plt.axis('equal')
    plt.legend()
    plt.show()

