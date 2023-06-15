# https://github.com/jingxixu/collision_checker.git
from .point_in_poly import point_in_polygon
import numpy as np

def get_segments(poly):
    '''
    TODO
    :param poly:
    :return:
    '''
    segs = []
    for i in range(len(poly)-1):
        segs.append([poly[i], poly[i+1]])
    if (np.linalg.norm(np.array(poly[-1]) - np.array(poly[0])) > 0.001):
        # is not closed polygon
        segs.append([poly[-1], poly[0]])
    return segs

def seg_seg_intersection(seg1, seg2):
    '''
    :param seg1: [(x1, y1), (x2, y2)]
    :param seg2:
    :return: True: collision detect
    '''

    # the interval of x value where the intersection must lie in
    I_x = [max(min(seg1[0][0], seg1[1][0]), min(seg2[0][0], seg2[1][0])),
            min(max(seg1[0][0], seg1[1][0]), max(seg2[0][0],seg2[1][0]))]
    # print(I_x)
    if I_x[0] > I_x[1]:
        return False
    # the interval of y value where the intersection must lie in
    I_y = [max(min(seg1[0][1], seg1[1][1]), min(seg2[0][1], seg2[1][1])),
            min(max(seg1[0][1], seg1[1][1]), max(seg2[0][1],seg2[1][1]))]
    # print(I_y)
    if I_y[0] > I_y[1]:
        return False

    # none of the two segments are vertical to x-axis
    if seg1[0][0] != seg1[1][0] and seg2[0][0] != seg2[1][0]:
        k1 = (seg1[0][1]-seg1[1][1]) / (seg1[0][0]-seg1[1][0])
        k2 = (seg2[0][1]-seg2[1][1]) / (seg2[0][0]-seg2[1][0])
        b1 = seg1[0][1]-k1*seg1[0][0]
        b2 = seg2[0][1]-k2*seg2[0][0]

        if k1 == k2:
            if b1 == b2:
                return True
            else:
                return False
        # not sure why x_intersection is rounded here
        x_intersection = (b2 - b1) / (k1 - k2)
        if x_intersection < I_x[0] or x_intersection > I_x[1]:
            return False
        y_intersection = k1 * x_intersection + b1
        # return (x_intersection, y_intersection)
        return True

    # seg1 is vertical to x-axis
    if seg1[0][0] == seg1[1][0] and seg2[0][0] != seg2[1][0]:
        k2 = (seg2[0][1]-seg2[1][1]) / (seg2[0][0]-seg2[1][0])
        b2 = seg2[0][1]-k2*seg2[0][0]
        x_intersection = float(seg1[0][0])
        if x_intersection < I_x[0] or x_intersection > I_x[1]:
            return False
        y_intersection = k2 * x_intersection + b2
        if y_intersection < I_y[0] or y_intersection > I_y[1]:
            return False
        # return (x_intersection, y_intersection)
        return True

    # seg2 is vertical to x-axis
    if seg2[0][0] == seg2[1][0] and seg1[0][0] != seg1[1][0]:
        k1 = (seg1[0][1]-seg1[1][1]) / (seg1[0][0]-seg1[1][0])
        b1 = seg1[0][1]-k1*seg1[0][0]
        x_intersection = float(seg2[0][0])
        if x_intersection < I_x[0] or x_intersection > I_x[1]:
            return False
        y_intersection = k1 * x_intersection + b1
        if y_intersection < I_y[0] or y_intersection > I_y[1]:
            return False
        # return (x_intersection, y_intersection)
        return True

    # both segments are vertical to the x-axis
    if seg1[0][0] == seg1[1][0] and seg2[0][0] == seg2[1][0]:
        if I_x[0] == I_x[1] and I_y[0] <= I_y[1]:
            return True
        return False
    return False

def seg_polygon_intersection(seg, poly):
    '''
    :param seg: [(x1, y1), (x2, y2)]
    :param poly: a list of vertices (len > 2)
    :return: true: collistion detected
    '''
    poly_segs = get_segments(poly)
    for poly_seg in poly_segs:
        if seg_seg_intersection(poly_seg, seg):
            return True
    return False


def poly_poly_intersection(obs, obj):
    '''
    :param obs: a list of vertics (len > 2), e.g. [(x1, y1), (x2, y2), (x3, y3)]
    :param obj: a list of vertics (len > 2), e.g. [(x1, y1), (x2, y2), (x3, y3)]
    :return: true: collision
    '''
    obj_segs = get_segments(obj)
    for obj_seg in obj_segs:
        if seg_polygon_intersection(obj_seg, obs):
            return True
    # check endpoints if the obj is contained in the obs, or inverse
    obj_pt = obj[0]
    if point_in_polygon(obj_pt, obs):
        return True
    obs_pt = obs[0]
    if point_in_polygon(obs_pt, obj):
        return True

    # nothing collistion detect
    return  False

def polys_poly_intersection(obstacles, object):
    '''
    :param obstacles: a list of vertices of obstacles, e.g. [[(x0,y0), (x1,y1), (x2,y2)]]
    :param object: a list of vertics of object, e.g. [(x0,y0), (x1,y1), (x2,y2)]
    :return:
            true: collistion detected
            false: none
    '''
    for obs in obstacles:
        if poly_poly_intersection(obs, object):
            return True
    return False