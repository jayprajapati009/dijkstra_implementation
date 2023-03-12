"""
# Title:  Project-2 : Dijkstra Algorithm Implementation
# Course: ENPM661 - Planning for Autonomous Robots
"""

from queue import PriorityQueue

import sys
import time
import cv2
import numpy as np


__author__ = "Jay Prajapati"
__copyright__ = "Copyright 2023, Project-2"
__credits__ = ["Jay Prajapati"]
__license__ = "MIT"
__version__ = "1.0.1"
__email__ = "jayp@umd.edu"


class Node:
    """class representating the data structure for the nodes
    """

    def __init__(self, coords, cost, preced=None):
        self.coords = coords
        self.x_cord = coords[0]
        self.y_cord = coords[1]
        self.cost = cost
        self.preced = preced


def setOrigin(y_cord):
    """Sets the origin to the left bottom corner

    Args:
        y_cord (int): actual y coordinate

    Returns:
        int: updated y coordinate
    """
    return 249 - y_cord


def createCanvas():
    """Create the 2D map (Canvas) for the simulation

    Returns:
        np.ndarray: Binary Image containing the info of all the available pixels
    """
    # Create an empty frame
    map = np.zeros((300, 600), dtype=np.uint8)

    # Draw the rectangles
    cv2.rectangle(map, (95, 0), (155, 105), (255, 255, 255), -1)
    cv2.rectangle(map, (95, 145), (155, 245), (255, 255, 255), -1)
    cv2.rectangle(map, (0, 0), (600, 250), (255, 255, 255), 5)

    # Draw the hexagons
    hexagon_corners = np.array([
        [235,  87.5],
        [235, 162.5],
        [300,   205],
        [370, 162.5],
        [370,  87.5],
        [300,    45]
    ], np.int32)
    hexagon_corners = hexagon_corners.reshape((-1, 1, 2))
    cv2.fillPoly(map, [hexagon_corners], (255, 255, 255))

    # Draw the triangles
    triangle_corners = np.array([
        [455, 20],
        [455, 230],
        [515, 125]
    ], np.int32)
    triangle_corners = triangle_corners.reshape((-1, 1, 2))
    cv2.fillPoly(map, [triangle_corners], (255, 255, 255))

    return map


def createAnimationCanvas():
    """Creates a visual 2D Map for the animation video for this simulation

    Returns:
        np.ndarray: Image frame for the 2D map
    """
    # Create an empty frame
    anim_canvas = np.zeros((250, 600, 3), dtype=np.uint8)

    # Draw the rectangles
    cv2.rectangle(anim_canvas, (100, 0), (150, 100), (0, 165, 255), -1)
    cv2.rectangle(anim_canvas, (95, 0), (155, 105), (255, 255, 255), 5)
    cv2.rectangle(anim_canvas, (100, 150), (150, 250), (0, 165, 255), -1)
    cv2.rectangle(anim_canvas, (95, 145), (155, 245), (255, 255, 255), 5)
    cv2.rectangle(anim_canvas, (0, 0), (600, 250), (255, 255, 255), 5)

    # Draw the hexagons
    hexagon_corners = np.array([
        [235,  87.5],
        [235, 162.5],
        [300,   200],
        [365, 162.5],
        [365,  87.5],
        [300,    50]
    ], np.int32)
    hexagon_corners = hexagon_corners.reshape((-1, 1, 2))
    hexagon_padding = np.array([
        [235,  87.5],
        [235, 162.5],
        [300,   205],
        [370, 162.5],
        [370,  87.5],
        [300,    45]
    ], np.int32)
    hexagon_padding = hexagon_padding.reshape((-1, 1, 2))
    cv2.fillPoly(anim_canvas, [hexagon_padding], (0, 255, 0))
    cv2.polylines(anim_canvas, [hexagon_corners],
                  True, (255, 255, 255), thickness=5)

    # Draw the triangles
    traingle_corners = np.array([
        [460, 25],
        [460, 225],
        [510, 125]
    ], np.int32)
    traingle_corners = traingle_corners.reshape((-1, 1, 2))
    triangle_padding = np.array([
        [455,  20],
        [455, 230],
        [515, 125]
    ], np.int32)
    triangle_padding = triangle_padding.reshape((-1, 1, 2))
    cv2.fillPoly(anim_canvas, [traingle_corners], (0, 165, 255))
    cv2.polylines(anim_canvas, [triangle_padding],
                  True, (255, 255, 255), thickness=10)

    return anim_canvas


def checkSolvable(x, y):
    """Checks whether the input coordinates are valid or not

    Args:
        x (int): x coordinate
        y (int): x coordinate

    Returns:
        bool: boolean value to check whether the case is solvable or not 
    """
    # Initialize the flag
    solvable = True

    # Check for borders
    if ((x >= 0 and x < 5) or (x > 594 and x < 600) or (y >= 0 and y < 5) or (y > 244 and y < 250)):
        solvable = False

    # Check for rectangles
    if (x >= 100 and x <= 150 and y >= 150 and y < 250):
        solvable = False
    if (x >= 100 and x <= 150 and y >= 0 and y <= 100):
        solvable = False

    # Check for the hexagon
    if (x >= 460):
        if (-100 * x + 50 * y > -44750) and (2 * x + y < 1145):
            solvable = False

    # Check for the triangle
    if (x >= (300 - 75 * np.cos(np.deg2rad(30))) and x <= (300 + 75 * np.cos(np.deg2rad(30)))):
        if (37.5 * x + 64.95 * y > 14498) and (-37.5 * x + 64.95 * y < 1742):
            if (37.5 * x + 64.95 * y < 24240) and (37.5 * x - 64.95 * y < 8002.5):
                solvable = False

    return solvable


def moveUp(point):
    """ Moves the robot UP

    Args:
        point (list): coordinates of the pixel

    Returns:
        list: updated coordinates of the pixel
    """
    point[1] += 1
    return point


def moveDown(point):
    """ Moves the robot DOWN

    Args:
        point (list): coordinates of the pixel

    Returns:
        list: updated coordinates of the pixel
    """
    point[1] -= 1
    return point


def moveRight(point):
    """ Moves the robot Right

    Args:
        point (list): coordinates of the pixel

    Returns:
        list: updated coordinates of the pixel
    """
    point[0] += 1
    return point


def moveLeft(point):
    """ Moves the robot Left

    Args:
        point (list): coordinates of the pixel

    Returns:
        list: updated coordinates of the pixel
    """
    point[0] -= 1
    return point


def moveUpRight(point):
    """ Moves the robot UP-Right

    Args:
        point (list): coordinates of the pixel

    Returns:
        list: updated coordinates of the pixel
    """
    point[1] += 1
    point[0] += 1
    return point


def moveUpLeft(point):
    """ Moves the robot UP-Left

    Args:
        point (list): coordinates of the pixel

    Returns:
        list: updated coordinates of the pixel
    """
    point[1] += 1
    point[0] -= 1
    return point


def moveDownRight(point):
    """ Moves the robot DOWN-Right

    Args:
        point (list): coordinates of the pixel

    Returns:
        list: updated coordinates of the pixel
    """
    point[1] -= 1
    point[0] += 1
    return point


def moveDownLeft(point):
    """ Moves the robot DOWN-Left

    Args:
        point (list): coordinates of the pixel

    Returns:
        list: updated coordinates of the pixel
    """
    point[1] -= 1
    point[0] -= 1
    return point


def generateChildren(node):
    """Generates all the possible children nodes

    Args:
        node (<class 'NODE'>): object of NODE class representating a node (Pixel)

    Returns:
        list: List of all the possible children nodes
    """
    # Get the x and y coordinates of the node
    x_cord, y_cord = node.x_cord, node.y_cord

    # Creating list for the succeeding nodes
    succ_nodes_all = []

    # Generating and appending the succeeding nodes using all the movement functions
    up = moveUp([x_cord, y_cord])
    if up is not None:
        succ_nodes_all.append(up.copy())

    down = moveDown([x_cord, y_cord])
    if down is not None:
        succ_nodes_all.append(down.copy())

    left = moveLeft([x_cord, y_cord])
    if left is not None:
        succ_nodes_all.append(left.copy())

    right = moveRight([x_cord, y_cord])
    if right is not None:
        succ_nodes_all.append(right.copy())

    upLeft = moveUpLeft([x_cord, y_cord])
    if upLeft is not None:
        succ_nodes_all.append(upLeft.copy())

    upRight = moveUpRight([x_cord, y_cord])
    if upRight is not None:
        succ_nodes_all.append(upRight.copy())

    downLeft = moveDownLeft([x_cord, y_cord])
    if downLeft is not None:
        succ_nodes_all.append(downLeft.copy())

    downRight = moveDownRight([x_cord, y_cord])
    if downRight is not None:
        succ_nodes_all.append(downRight.copy())

    # Check the posibility of exitence of the succeeding node
    succ_nodes = []
    for idx, coords in enumerate(succ_nodes_all):
        if coords[0] >= 0 and coords[0] < canvas_width and coords[1] >= 0 and coords[1] < canvas_height:
            if canvas[setOrigin(coords[1])][[coords[0]]] == 0:
                # set the cost for the step if the node is valid
                if idx > 3:
                    cost = 1.4
                else:
                    cost = 1
                succ_nodes.append([coords, cost])

    return succ_nodes


def dijkstra():
    """Generates the node graph and reaches the end goal using Dijkstra algorithm

    Returns:
        dict, np.ndarray, list : node_graph, anim_canvas, animation_array
    """
    # create a list for stores all the frame for the animation video
    animation_array = []

    # Initializing the priority queue
    queue = PriorityQueue()

    # creating a set for storing all the visited nodes
    visited_nodes = set([])

    # creating dictionaty for the node graph
    node_graph = {}
    # creating dictionaty for the node step distance and cost calculation
    step_dist = {}

    # set distance as infinity for all the nodes
    for y in range(canvas_height):
        for x in range(canvas_width):
            step_dist[str([y, x])] = infinity

    # set the parameters for the initial coordinate
    # and add it to the visited nodes list
    step_dist[str(initial_cord)] = 0
    visited_nodes.add(str(initial_cord))
    initial_node = Node(initial_cord, 0)
    node_graph[str(initial_node.coords)] = initial_node

    # add the initial node to the queue
    queue.put([initial_node.cost, initial_node.coords])

    # create a map for animation visualization
    anim_canvas = createAnimationCanvas()

    # Initialize the while loop
    i = 0
    while queue.empty() == False:
        # get the element in queue
        queued_node = queue.get()

        # add it to the node graph
        current_node = node_graph[str(queued_node[1])]

        # break the loop if reached to the goal coordinate
        if queued_node[1][0] == final_cord[0] and queued_node[1][1] == final_cord[1]:
            # update the parent to the goal coordinate
            node_graph[str(final_cord)] = Node(
                final_cord, queued_node[0], current_node)
            break

        # Generate the succeeding nodes and process them
        for succ_node, cost in generateChildren(current_node):
            # check if the child is already visited or not
            if str(succ_node) in visited_nodes:
                # update the cost to the current node if it is smaller than the previous one
                current_cost = cost + step_dist[str(current_node.coords)]
                if current_cost < step_dist[str(succ_node)]:
                    step_dist[str(succ_node)] = current_cost
                    node_graph[str(succ_node)].preced = current_node

            else:
                # if not in visited nodes
                visited_nodes.add(str(succ_node))

                # mark the pixel
                anim_canvas[setOrigin(succ_node[1]),
                            succ_node[0], :] = np.array([255, 0, 0])

                # add the frame to the animation frame array
                if i % 300 == 0:
                    animation_array.append(anim_canvas.copy())
                    cv2.imshow("Animation", anim_canvas)
                    cv2.waitKey(1)

                # update the current cost
                cost_ccn = cost + step_dist[str(current_node.coords)]

                # update the step distance
                step_dist[str(succ_node)] = cost_ccn

                # get the next node
                next_node_in_queue = Node(succ_node, cost_ccn,
                                          node_graph[str(current_node.coords)])

                # put the next node on queue
                queue.put([cost_ccn, next_node_in_queue.coords])

                # put the next node on node graph dictionary
                node_graph[str(next_node_in_queue.coords)] = next_node_in_queue

        # increment operator
        i += 1

    return node_graph, anim_canvas, animation_array


def backTrack(node_graph):
    """Generates the node graph

    Args:
        node_graph (dictionary): node graph info for all the nodes

    Returns:
        list: path from goal point to the inital point
    """
    # get the parent of the final node
    final_node = node_graph[str(final_cord)]
    final_cord_parent = final_node.preced
    path = []

    while final_node:
        if not final_cord_parent:
            break
        # find the ancesstor of all the nodes until the initial node
        path.append([setOrigin(final_cord_parent.coords[1]),
                    final_cord_parent.coords[0]])
        final_cord_parent = final_cord_parent.preced
    return path


def main():
    """main function
    """
    global canvas_height, canvas_width, initial_cord, final_cord, infinity, canvas

    canvas_height, canvas_width = 250, 600
    infinity = sys.maxsize


if __name__ == "__main__":
    main()
