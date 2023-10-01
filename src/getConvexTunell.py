#!/usr/bin/env python

import rospy
from nav_msgs.msg import OccupancyGrid
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
import numpy as np
import casadi as cs
import tf
import math
from geometry_msgs.msg import Point32
from geometry_msgs.msg import PolygonStamped

class PathPlannerNode:
    def __init__(self): # ~~~~~~~~~~~~~~~~~~~~~~~~ Constructor ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        rospy.init_node('path_planner_node')

        self.R = 100 # m

        # Subscribers
        self.map_sub = rospy.Subscriber('/map', OccupancyGrid, self.map_callback)
        self.odom_sub = rospy.Subscriber('/odom_base', Odometry, self.odom_callback)
        self.goal_sub = rospy.Subscriber('/move_base_simple/goal', PoseStamped, self.goal_callback)
        self.path_sub = rospy.Subscriber('/nav_path', Path, self.path_callback)
        # Publishers
        self.path_pub = rospy.Publisher('/new_path', Path, queue_size=10)
        self.polygon_pub = rospy.Publisher('/polygon', PolygonStamped, queue_size=10)
        
        # Initialize the initial position (odom)
        self.x0 = np.zeros(3)
        
        #  Initialize the goal position
        self.goal_pos = np.zeros(2)
        self.goal_set = False
        
        # Initialize the map
        self.map = OccupancyGrid()
        self.map_set = False
        
        # Initialize the optimal paths
        self.n_opt = []
        self.x_opt = []
        self.y_opt = []
        self.dt_opt = []

        self.mapped_points = []

        self.hull = []
        self.hulls = []

        self.astar_path = Path()
        
        # Define the constraints
        Vlim = 3 # m/s
        g = 9.81 # m/s^2
        Alim = g/4 # m/s^2
        Slim = 30/180*np.pi # rad
        L = 1.326*2 # m


    def map_callback(self, map_msg): # ~~~~~~~~~~~~~~~~~~~~~~~~ Callbacks ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        self.map = map_msg
        self.map_set = True

    def odom_callback(self, odom_msg):
        self.x0[0] = odom_msg.pose.pose.position.x
        self.x0[1] = odom_msg.pose.pose.position.y
        q = odom_msg.pose.pose.orientation
        # Transform the quaternion to Euler angles
        e = tf.transformations.euler_from_quaternion([q.x, q.y, q.z, q.w])
        self.x0[2] = e[2]

    def goal_callback(self, goal_msg):
        self.goal_pos[0] = goal_msg.pose.position.x
        self.goal_pos[1] = goal_msg.pose.position.y
        self.goal_set = True

    def path_callback(self, path_msg):

        if path_msg.poses == []:
            return
        
        self.astar_path = Path()
        self.astar_path = path_msg

    def getTunells(self): # ~~~~~~~~~~~~~~~~~~~~~~~~ Path generation ~~~~~~~~~~~~~~~~~

        if self.astar_path.poses == []:
            return
        
        centerPoint = self.astar_path.poses[-1
                                            ]
        self.getConcaveTunell(centerPoint)

    def getConcaveTunell(self, centerPoint): # ~~~~~~~~~~~~~~~~~~~~~~~~ Path generation ~~~~~~~~~~~~~~~~~

        origin_x = self.map.info.origin.position.x
        origin_y = self.map.info.origin.position.y
        resolution = self.map.info.resolution
        width = self.map.info.width
        height = self.map.info.height

        x0 = centerPoint.pose.position.x
        y0 = centerPoint.pose.position.y

        x_ind = int((x0 - origin_x)/resolution)
        y_ind = int((y0 - origin_y)/resolution)

        mapped_point = np.zeros(2)

        delta_ind = int(self.R/resolution)

        for i in range(2*delta_ind):
            for j in range(2*delta_ind):

                x_ind_map = int(x_ind + i - delta_ind)
                y_ind_map = int(y_ind + j - delta_ind)

                if x_ind_map >= 0 and x_ind_map < width and y_ind_map >= 0 and y_ind_map < height:
                        
                    if self.map.data[x_ind_map + y_ind_map*width] == 100:

                        x_point = origin_x + x_ind_map*resolution
                        y_point = origin_y + y_ind_map*resolution

                        px = x_point - x0
                        py = y_point - y0

                        pnorm = np.sqrt(px**2 + py**2)

                        if (pnorm == 0.0): return

                        mapped_point[0] = px + px * 2*(self.R - pnorm) / pnorm
                        mapped_point[1] = py + py * 2*(self.R - pnorm) / pnorm

                        self.mapped_points.append(mapped_point)

        # Get convex hull
        self.quickHull(self.mapped_points)
        self.hulls.append(self.hull)   
        self.hull = []


    def quickHull(self,S):
        S = sorted(S, key=lambda x: x[0])
        self.hull.append(S[0])
        self.hull.append(S[-1])
        S1 = []
        S2 = []

        for x in S:
            if (self.sideOfLinePointIsOn([S[0], S[-1]], x) > 0.00):
                    S2.append(x)
                    
            if(self.sideOfLinePointIsOn([S[0], S[-1]], x) < 0.00):
                    S1.append(x)
        
        self.findHull(S1, S[0], S[-1])
        self.findHull(S2, S[-1], S[0])

    def findHull(self,Sk, P, Q):
        if(len(Sk) == 0): return
        furthestPoint = Sk[0]
        maxDist = 0
        for x in Sk:
            
            dist = self.findDistPointLine(self,x, [P, Q])
            if(dist > maxDist): 
                maxDist = dist
                furthestPoint = x
        
        Sk.remove(furthestPoint)

        self.hull.insert(1, furthestPoint)
        S1 = []
        S2 = []
        for p in Sk: 
            if(self.isInsideTriangle(P, furthestPoint, Q, p)): Sk.remove(p)
            if(self.sideOfLinePointIsOn([P, furthestPoint], p) > 0.00):
                    S2.append(x)
            if(self.sideOfLinePointIsOn([furthestPoint, Q], x) < 0.00):
                    S1.append(x)
        self.findHull(S1, P, furthestPoint)
        self.findHull(S2, furthestPoint, Q)
        

    def slopeCalc(self,P, Q):
        return (Q[1] - P[1])/(Q[0]-P[0])


    # y =ax+b, y=cx+d
    # x = (d-b)/(a-c)
    def findNearestPoint(self,point, line):
        line = sorted(line, key=lambda x: x[0])
        # reg line
        slope = self.slopeCalc(line[0], line[1])

        yInt = line[0][1] - (slope * line[0][0])

        #neg reciprocal
        slopeRecip = (-1)/slope
        yIntRecip = point[1] - (slopeRecip * line[0][0])

        nearestPoint = [0,0]
        nearestPoint[0] = (yIntRecip -yInt)/(slope-slopeRecip)
        nearestPoint[1] = slope * nearestPoint[0] + yInt
        
        # if nearestPoint x val is less than min x val on segment PQ
        if(nearestPoint[0] < line[0][0]):
            return line[0]

        # if nearestPoint x val is greater than max x val on segment PQ
        if(nearestPoint[0] > line[-1][0]):
            return line[-1]
        
    
        return nearestPoint
        
    def findPointDistance(self,P1, P2):
        
        return math.hypot(P2[0] - P1[0], P2[1] - P1[1])


    def findDistPointLine(self,point, line):
        return self.findPointDistance(point, self.findNearestPoint(point, line));


    def sideOfLinePointIsOn(self,line, x):
        vectAB = ((line[1][0] - line[0][0]), line[1][1] - line[0][1])
        vectAX = ((x[0] - line[0][0]), x[1] - line[0][1])
        zCoord = (vectAB[0] * vectAX[1]) - (vectAB[1] * vectAX[0])
        return zCoord

    def isInsideTriangle(self,A, B, C, p):
        if( 
        (self.sideOfLinePointIsOn((A,B), p) > 0) and
        (self.sideOfLinePointIsOn((B,C), p) > 0) and
        (self.sideOfLinePointIsOn((C,A), p) > 0)):
            return True
        
    def publishPath(self):

        if self.hulls == []:
            return
        print(self.hulls)

    def run(self): # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Run ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        rate = rospy.Rate(10)  # 10 Hz

        while not rospy.is_shutdown():
            # Generate and publish the path message here
            
            #if self.goal_set == False:
            #    return
            
            self.getTunells()
            self.publishPath()
                        
            # Publish the polygon hull
            hull_msg = PolygonStamped()
            hull_msg.header.stamp = rospy.Time.now()
            hull_msg.header.frame_id = "map"
            # Fill in the hull message with self.hull
            if self.hulls == []:
                hull_msg.polygon.points = [Point32(0, 0, 0)]
            else:
                x = []
                y = []
                z = []
                for i in range(len(self.hulls[0])):
                    x.append(self.hulls[0][i][0])
                    y.append(self.hulls[0][i][1])
                    z.append(0)
                hull_msg.polygon.points = Point32(x,y,z)

            # Fill in the path message with the desired data

            self.polygon_pub.publish(hull_msg)
            

            rate.sleep()

if __name__ == '__main__': # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Main ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    path_planner_node = PathPlannerNode()
    path_planner_node.run()
    
    #except rospy.ROSInterruptException:
    #    pass
