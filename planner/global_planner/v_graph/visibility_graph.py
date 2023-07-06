import numpy as np
import math
import csv
import matplotlib.pyplot as plt
import matplotlib.path as mplPath

import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from path_search import edge_to_search_graph, dijkstra_search


# https://github.com/enriquea52/Visibility-Graph.git

class Point:
    def __init__(self, x=0, y=0):
        self.x = x
        self.y = y
        self.alph = 0

    def alpha(self, x):
        self.alph = x

    def dist(self, p):
        return math.sqrt((self.x - p.x)**2 + (self.y - p.y)**2)

    def numpy(self):
        return np.array([self.x, self.y])

    def dist_line(self, l):
        return np.linalg.norm(np.cross(l.p2.numpy() - l.p1.numpy(), l.p1.numpy() - self.numpy())) / np.linalg.norm(l.p2.numpy() - l.p1.numpy())

    def __str__(self):
        return "({}, {}, {})".format(self.x, self.y,self.alph)

    def dot(self, p):
        return self.x * p.x + self.y*p.y

    def length(self):
        return math.sqrt(self.x**2 + self.y**2)

    def vector(self, p):
        return Point(p.x - self.x, p.y - self.y)

    def unit(self):
        mag = self.length()
        return Point(self.x/mag, self.y/mag)

    def scale(self, sc):
        return Point(self.x * sc, self.y * sc)

    def add(self, p):
        return Point(self.x + p.x, self.y + p.y)

    def __sub__(self, p):
        return Point(self.x - p.x, self.y - p.y)

    def __truediv__(self, s):
        return Point(self.x / s, self.y / s)

    def __floordiv__(self, s):
        return Point(int(self.x / s), int(self.y / s))

    def __mul__(self, s):
        return Point(self.x * s, self.y * s)

    def __rmul__(self, s):
        return self.__mul__(s)

    def dist_segment(self, s):
        line_vec = s.p1.vector(s.p2)
        pnt_vec = s.p1.vector(self)
        line_len = line_vec.length()
        line_unitvec = line_vec.unit()
        pnt_vec_scaled = pnt_vec.scale(1.0/line_len)
        t = line_unitvec.dot(pnt_vec_scaled)
        if t < 0.0:
            t = 0.0
        elif t > 1.0:
            t = 1.0
        nearest = line_vec.scale(t)
        dist = nearest.dist(pnt_vec)
        nearest = nearest.add(s.p1)
        return dist

    def __str__(self):
        return "({}, {}, {})".format(self.x, self.y,self.alph)

def ccw(A, B, C):
    return (C.y - A.y) * (B.x - A.x) >= (B.y - A.y) * (C.x - A.x)

def on_segment(p,q,r):
    if (q.x <= max(p.x, r.x)) and (q.x >= min(p.x, r.x)):
        if (q.y <= max(p.y, r.y)) and (q.y >= min(p.y, r.y)):
            return True
    return False

def wrap2pi(angle):
    # https://stackoverflow.com/a/32266181
    # wrap angle to (-pi, pi]
    return (( -angle + np.pi) % (2.0 * np.pi ) - np.pi) * -1.0

def det(a, b):
    return a[0] * b[1] - a[1] * b[0]

class Segment:
    def __init__(self, p1=Point(), p2=Point()):
        self.p1 = p1
        self.p2 = p2
        self.distance = 0

    @classmethod
    def point_angle_length(cls, p1=Point(), angle=0, length=1):
        x2 = p1.x + math.cos(angle) * length
        y2 = p1.y + math.sin(angle) * length
        return cls(p1, Point(x2, y2))

    # Return true if line segments AB and CD intersect
    def intersect(self, s):
        o1 = ccw(self.p1, s.p1, s.p2)
        o2 = ccw(self.p2, s.p1, s.p2)
        o3 = ccw(self.p1, self.p2, s.p1)
        o4 = ccw(self.p1, self.p2, s.p2)

        if o1 != o2 and o3 != o4:
            return True, self.intersection_point(s)
        return False,None

        # if ccw(self.p1, s.p1, s.p2) != ccw(self.p2, s.p1, s.p2) and ccw(self.p1, self.p2, s.p1) != ccw(self.p1, self.p2, s.p2):
        #     return True, self.intersection_point(s)
        # else:
        #     return False, None

    def intersection_point(self, line):
        xdiff = (self.p1.x - self.p2.x, line.p1.x - line.p2.x)
        ydiff = (self.p1.y - self.p2.y, line.p1.y - line.p2.y)

        div = det(xdiff, ydiff)
        if div == 0:
            #print("Something went wrong!")
            return None

        d = (det((self.p1.x, self.p1.y), (self.p2.x, self.p2.y)), det((line.p1.x, line.p1.y), (line.p2.x, line.p2.y)))
        x = det(d, xdiff) / div
        y = det(d, ydiff) / div
        return Point(x, y)

    def dis(self, x):
        self.distance = x

    def magnitude(self):
        return math.sqrt(((self.p2.x-self.p1.x)**2)+(self.p2.y-self.p1.y)**2)

    def angle(self):
        return abs(math.atan2((self.p2.y-self.p1.y), (self.p2.x-self.p1.x)))

    def __str__(self):
        return "[{}, {}]".format(self.p1, self.p2)

# The present class is the implementation of the Rotational Plane Sweep Algorithm used
# for building visibility graphs.
# this class includes all the function definitions required to achieve the application.
class implementation(object):

    def __init__(self, vertexes, edges):
        self.vertexes = vertexes
        self.obstacles_edges = edges

    def get_vertexes_from_dict(self, v_dict): # This function retrieves vertexes from a dictionary of lists and converts it to a list of vertexes.
        vertexes = []
        for i in v_dict:
            for i in v_dict[i]:
                vertexes.append(i)
        return vertexes

    def angle(self,y,x): # Make an angle from 0 to 2pi for sorting purposes
        angle = np.arctan2(y,x)
        if angle < 0:
            angle = (angle + 2*np.pi)
        return angle

    def copy_vertex_list(self,list): # Function for copying a list of vertexes safely
        new_list = []
        for vertex in list:
            new_list.append(Point(vertex.x,vertex.y))
        return new_list

    def S_inicialization(self,half_line, current_vertex):
        # This function initializes the S list by creating an horizontal halfline with 0 degrees with respect the global x axis.
        # Every edge in the environment intersecting the half line will be added to the S list.
        # The S list will be sorted depending on which edge was intersected first.
        S = []
        for edge in self.obstacles_edges:
            is_interset = half_line.intersect(edge)
            temp_point= half_line.intersection_point(edge)
            if (is_interset[0] and round(current_vertex.dist(temp_point),0) != 0):
                edge.distance = current_vertex.dist(temp_point)
                S.append(edge)
        S = sorted(S, key=lambda x: x.distance)
        return S

    def is_visible(self,v,vi,s, sweep_line):
        # This function returns True if a vertex vi is visible from a vertex v,
        # Otherwise it returns false.
        # It follows a checklist for corroborating if the vertex vi is visible from different criteria

        # If the S list is empty the vertex vi is visible from v
        if len(s) == 0:
            return True
        # If both v and vi lay on the same edge in S, vi is visible from v
        for i in s:
            if round(v.dist_segment(i),3) == 0. and round(vi.dist_segment(i),3) == 0.:
                return True
        # If vi and v are on the same obstacle and if the midpoint between them is inside the obstacle
        # vi is not visible from v
        if self.inside_poligon(v,vi,s):
            return False
        # If the first edge in S intersect the sweepline going from v to vi, vi is not visible from v
        for edge in s:
            is_interset = sweep_line.intersect(edge)
            if is_interset[0] and not(round(v.dist_segment(edge),3) == 0.):
                return False
            else:
                return True

    def inside_poligon(self, v, vi, s):
        # First check if both vertexes belong to the same obstacle
        id1 = None
        id2 = None
        for i in range(0,len(self.vertexes)):
            for j in self.vertexes[i]:
                if (v.x,v.y) == (j.x,j.y):
                    id1 = i
                if (vi.x,vi.y) == (j.x,j.y):
                    id2 = i
        # If both vertexes belong to the same obstacle, and the midpoint between them is inside an obstacle, vi is not visible from v
        if id1 == id2:
            poly_path = mplPath.Path(np.array([[vertex.x,vertex.y] for vertex in self.vertexes[id1]]))
            midpoint = ((v.x+vi.x)/2, (v.y+vi.y)/2)
            return poly_path.contains_point(midpoint)
        else:
            return False

    def remove_repeated(self, visible): # Function used to remove repeated visibility edges from the final visibility edge list
        i = 0
        j = 1
        while i<len(visible) - 1:
            while j<len(visible):
                if (visible[i].p1.x == visible[j].p2.x and visible[i].p1.y == visible[j].p2.y and visible[i].p2.x == visible[j].p1.x and visible[i].p2.y == visible[j].p1.y) :
                    visible.remove(visible[j])
                    break
                j+=1
            i+=1
            j = i+1

        return [ x for x in visible if not(x.p1.x == x.p2.x and x.p1.y == x.p2.y)]

    def rotational_sweep(self): # Rotational Plane Sweep Algorithm Implementation

        vertexes = self.get_vertexes_from_dict(self.vertexes)
        sorted_vertexes = self.copy_vertex_list(vertexes)
        visibility = []
        Len_inf = 10000

        for k in range(0,len(vertexes)):
            v = vertexes[k] # Vertex to check visibility from

            # Sort vertexes according to the angle
            for point in sorted_vertexes:
                point.alpha(self.angle(point.y-v.y,point.x-v.x))

            sorted_vertexes = sorted(sorted_vertexes, key=lambda x: x.alph)

            half_line = Segment(v,Point(v.x+Len_inf,v.y))

            # S list inizialization
            S = self.S_inicialization(half_line, vertexes[k])

            for vi in sorted_vertexes: # Start to check visibility of vi with respect v
                for edge in self.obstacles_edges: # S list update
                    if round(vi.dist_segment(edge),2) == 0. and edge not in S:
                        S.append(edge)
                    elif (round(vi.dist_segment(edge),2) == 0.  and edge in S) or (round(v.dist_segment(edge),2) == 0. and edge in S):
                        S.remove(edge)
                # create a sweep line from vertex v to vi with an angle offset of 0.001 and a magnitude of Len_inf
                vi_SL = Point(v.x+(Len_inf)*np.cos(vi.alph + 0.001),v.y+(Len_inf)*np.sin(vi.alph + 0.001))
                sweep_line = Segment(v,vi_SL)
                sweep_line1 = Segment(v,vi)
                #///////////////////////////////////////////////////////////////////////////////////////////

                # Calculate the distance of the sweepline to every edge in S
                for s_edge in S:
                    temp_point= sweep_line.intersection_point(s_edge)
                    s_edge.distance = v.dist(temp_point)

                    # check collinear
                    vec_edge = s_edge.p1.numpy() - s_edge.p2.numpy()
                    vec_sweep = sweep_line1.p1.numpy() - sweep_line1.p2.numpy()
                    chk_col = np.cross(vec_edge, vec_sweep)
                    if 0 == chk_col.item():
                        s_edge.distance = 10000
                ##############################################################

                # Sort the S list with respect which obstacle edge is closer to v
                S = sorted(S, key=lambda x: x.distance)

                # Check for visibility
                if self.is_visible(v,vi,S, sweep_line1):
                    visibility.append(Segment(v,vi))

        return self.remove_repeated(visibility) # Return the visibility edges excluding repeated ones

    def get_visibility_edge_for_pt(self, pt):
        vertexes = self.get_vertexes_from_dict(self.vertexes)
        sorted_vertexes = self.copy_vertex_list(vertexes)
        visibility = []
        Len_look = 10000

        v = pt
        for point in sorted_vertexes:
            point.alpha(self.angle(point.y-v.y, point.x-v.x))

        sorted_vertexes = sorted(sorted_vertexes, key=lambda x: x.alph)
        half_line = Segment(v,Point(v.x+Len_look, v.y))
        S = self.S_inicialization(half_line, v)

        for vi in sorted_vertexes:
            for edge in self.obstacles_edges:
                if round(vi.dist_segment(edge),2) == 0. and edge not in S:
                    S.append(edge)
                elif (round(vi.dist_segment(edge),2) == 0.  and edge in S) or (round(v.dist_segment(edge),2) == 0. and edge in S):
                    S.remove(edge)
            vi_SL = Point(v.x+(Len_look)*np.cos(vi.alph + 0.001),v.y+(Len_look)*np.sin(vi.alph + 0.001))
            sweep_line = Segment(v,vi_SL)
            sweep_line1 = Segment(v,vi)
            for s_edge in S:
                temp_point= sweep_line.intersection_point(s_edge)
                s_edge.distance = v.dist(temp_point)

                # check collinear
                vec_edge = s_edge.p1.numpy() - s_edge.p2.numpy()
                vec_sweep = sweep_line1.p1.numpy() - sweep_line1.p2.numpy()
                chk_col = np.cross(vec_edge, vec_sweep)
                if 0 == chk_col.item():
                    s_edge.distance = 10000
            S = sorted(S, key=lambda x: x.distance)
            if self.is_visible(v,vi,S, sweep_line1):
                visibility.append(Segment(v,vi))

        return visibility
class VisibilityGraph:
    def __init__(self, contours_dict=[], edge=[]) -> None:
        """
            contours_dict: {id, points}
            edge: [segment], contours of obstacle
        """
        self.contours_dict = contours_dict
        self.edge = edge
        self.v_edge = []
        self.v_edge_goal = []
        self.v_edge_start = []
        if [] == contours_dict or [] == edge:
            pass
        else:
            rspa = implementation(contours_dict, edge)
            self.v_edge = rspa.rotational_sweep()

    def save(self, filename='tmp.csv'):
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f, quoting=csv.QUOTE_NONE, quotechar='', escapechar='/')
            # write header
            writer.writerow(['# The file is automatically generated. And should not be modified.'])
            writer.writerow(['# v-graph obstacle polygon. format: id  x  y.'])

            # write obstacle polygon
            for i in range(len(self.contours_dict)):
                cti =  self.contours_dict[i]
                for j in range(len(cti)):
                    pt = cti[j]
                    writer.writerow([i, f'{pt.x:.3g}', f'{pt.y:.3g}'])

            # write edge Separator
            writer.writerow(['# v-graph visibility edge. format: start.x  start.y  end.x  end.y'])
            for i in range(len(self.v_edge)):
                seg_i = self.v_edge[i]
                writer.writerow([f'{seg_i.p1.x:.3g}', f'{seg_i.p1.y:.3g}', f'{seg_i.p2.x:.3g}', f'{seg_i.p2.y:.3g}'])

    def load(self, filename):
        with open(filename, newline='', encoding='utf-8') as f:
            reader = csv.reader(f)
            stage = 0
            contours_dict = {}
            contours_edge = []
            v_edge = []
            for row in reader:
                if row[0].startswith("#"):
                    stage = stage + 1
                    continue
                if 1 == stage:
                    # header
                    pass
                elif 2 == stage:
                    # parse obstacle polygon
                    id = int(row[0])
                    if not bool(contours_dict.get(id)):
                        contours_dict[id] = []
                    contours_dict[id].append(Point(int(row[1]), int(row[2])))
                elif 3 == stage:
                    # parse visibility edge
                    p1 = Point(int(row[0]),int(row[1]))
                    p2 = Point(int(row[2]),int(row[3]))
                    v_edge.append(Segment(p1, p2))
            for i in range(len(contours_dict)):
                poly = contours_dict[i]
                for j in range(len(poly)):
                    next_idx = j+1
                    if j==len(poly)-1 : next_idx = 0
                    pt1 = poly[j]
                    pt2 = poly[next_idx]
                    contours_edge.append(Segment(pt1, pt2))
            self.contours_dict = contours_dict
            self.v_edge = v_edge
            self.edge = contours_edge

    def update_pt(self):
        v_alg = implementation(self.contours_dict, self.edge)
        self.v_edge_start = v_alg.get_visibility_edge_for_pt(self.start_pt)
        self.v_edge_goal  = v_alg.get_visibility_edge_for_pt(self.goal_pt)

    def set_start_pt(self, x, y):
        self.start_pt = Point(x,y)

    def set_goal_pt(self, x, y):
        self.goal_pt = Point(x,y);

    def get_contour_pts_nx2s(self):
        # list of nx2 array
        res = []
        for i in range(len(self.contours_dict)):
            poly = self.contours_dict[i]
            pts_nx2s = [pt.numpy() for pt in poly]
            res.append(np.array(pts_nx2s))
        return res

    def get_vedge_nx4(self):
        res = []
        for i in range(len(self.v_edge)):
            seg = self.v_edge[i]
            res.append(np.array([seg.p1.x, seg.p1.y, seg.p2.x, seg.p2.y]))
        for i in range(len(self.v_edge_start)):
            seg = self.v_edge_start[i]
            res.append(np.array([seg.p1.x, seg.p1.y, seg.p2.x, seg.p2.y]))
        for i in range(len(self.v_edge_goal)):
            seg = self.v_edge_goal[i]
            res.append(np.array([seg.p1.x, seg.p1.y, seg.p2.x, seg.p2.y]))
        return np.array(res)

    def get_shortest_path(self):
        res = []
        if [] == self.start_pt or [] == self.goal_pt:
            print("start and goal point should set.")
            return res
        edges = self.get_vedge_nx4()
        search_graph = edge_to_search_graph(edges)
        return dijkstra_search(search_graph, self.start_pt.numpy(), self.goal_pt.numpy())


    def draw(self, ax=[], transpose=False):
        if [] == ax:
            fig, ax = plt.subplots()
        contour_pts_nx2s = self.get_contour_pts_nx2s()
        v_edge_nx4 = self.get_vedge_nx4()
        idx = [0]
        idy = [1]
        if transpose:
            idx = [1]
            idy = [0]
        start = self.start_pt.numpy()
        goal = self.goal_pt.numpy()
        ax.plot(start[idx], start[idy], 'gs', label='start')
        ax.plot(goal[idx], goal[idy], 'rp', label='goal')
        for i in range(len(contour_pts_nx2s)):
            poly = contour_pts_nx2s[i]
            poly_closed = np.vstack((poly, poly[0]))
            if 0==i:
                ax.plot(poly[:,idx], poly[:,idy], 'r.', label='vertex')
                ax.plot(poly_closed[:,idx], poly_closed[:,idy], 'b-', label='contour')
            else:
                ax.plot(poly[:,idx], poly[:,idy], 'r.')
                ax.plot(poly_closed[:,idx], poly_closed[:,idy], 'b-')
        idx = [0,2]
        idy = [1,3]
        if transpose:
            idx = [1,3]
            idy = [0,2]
        for i in range(v_edge_nx4.shape[0]):
            seg = v_edge_nx4[i,:]
            if 0==i: ax.plot(seg[idx], seg[idy], 'g-', alpha=0.6, linewidth='0.3', label="visibility edge")
            else: ax.plot(seg[idx], seg[idy], 'g-', alpha=0.6, linewidth='0.3')
        return ax



# ============ test =========================
if __name__ == "__main__":
    edge = Segment(Point(310,215), Point(305,211))
    sweep_line = Segment(Point(135,75), Point(310,215))

    vec1 = np.array([310,215]) - np.array([305,211])
    vec2 = np.array([135,75]) - np.array([310,215])
    xx = np.cross(vec1, vec2)
    print(xx)