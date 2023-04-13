"""
Functions to create different grids of points
"""

import numpy as np
import antenna_toolbox as ant
from scipy.spatial import cKDTree as KDTree
from scipy.optimize import minimize_scalar


def rectangular_grid(n: int, m: int, d_x: float, d_y: float, z: float=0):
    """
    Creates a rectangular grid of points. The grid begins at x,y = 0,0.

    :param n: number of points along the x-axis (rows)
    :param m: number of points along the y-axis (columns)
    :param d_x: spacing of elements along the x-axis
    :param d_y: spacing of elements along the y-axis
    :param z: z-coordinate of the rectangular grid... default z=0
    :return: Numpy array [x, y, z] of coordinates on each point on the grid
    """

    # create numpy array for the grid
    grid = np.zeros((n * m, 3))

    # for loops to create each element
    point_number = 0  # number point in the grid
    z_pos = z         # z-coordinate of the grid
    for jj in range(0, m):  # y-loop
        for ii in range(0, n):  # x-loop
            # compute positions
            x_pos = ii * d_x
            y_pos = jj * d_y

            # assign positions to the numpy array with coordinates
            grid[point_number, 0] = x_pos
            grid[point_number, 1] = y_pos
            grid[point_number, 2] = z_pos

            # increment point number
            point_number += 1

    # return grid coordinates
    return grid


def triangular_grid_in_rectangle(spacing: float, x_lim: float, y_lim: float, start_point: tuple = (0, 0)):

    """
    Generates triangular grid, based on equilateral triangles, within a rectangle.

    :param spacing: spacing from point to point (side length on equilateral triangle)
    :param x_lim: maximum x-coordinate for points (exclusive)
    :param y_lim: maximum y-coordinate for points (exclusive)
    :param start_point: starting point to place bottom left corner of rectangle containing grid at (default origin)
    :return: nparray of grid points [X, Y]
    """

    # triangular grid information
    h_dist = spacing  # spacing between points horizontally (x)
    v_dist = spacing * ant.math_funcs.cosd(30)  # spacing between points vertically (y)

    # grid generation variables
    grid = None  # variable for grid points... [X, Y]
    y_current = start_point[1]  # current y-coordinate
    xx = None  # currently generated X points
    yy = None  # currently generated Y points
    displacement = False  # odd (False) or even (True) row?

    # while loop for grid generation
    while y_current < y_lim:

        # generate grid points... both odd and even rows share the same general procedure
        if not displacement:  # odd row
            xx = np.arange(start_point[0], x_lim, h_dist)  # create x-coordinates, spaced by h_dist
            yy = np.ones(len(xx)) * y_current  # create y-points, based on current y-coordinate
            xx = np.reshape(xx, (len(xx), 1))  # reshape x point np array to have 1 column and n rows
            yy = np.reshape(yy, (len(yy), 1))  # same as above
            displacement = True  # change flag for odd/even rows
        else:  # even row
            xx = np.arange(start_point[0] + h_dist / 2, x_lim, h_dist)
            yy = np.ones(len(xx)) * y_current
            xx = np.reshape(xx, (len(xx), 1))
            yy = np.reshape(yy, (len(yy), 1))
            displacement = False

        # add points to grid
        if grid is None:  # initialize grid if this is the first set of points generated
            grid = np.concatenate((xx, yy), axis=1)
        else:  # append points
            temp = np.concatenate((xx, yy), axis=1)
            grid = np.concatenate((grid, temp), axis=0)

        # increment current y-coordinate
        y_current += v_dist

    return grid


def wrap_points_around_circle(x, y):
    """
    Wraps a set of x-y points around a circle.

    :param x: x-points
    :param y: y-points
    :return: new x points and y points, wrapped around a circle
    """
    theta = 2 * np.pi / np.max(x) * (x - np.min(x))     # create theta
    x_new = y * np.cos(theta)                           # compute x coordinates
    y_new = y * np.sin(theta)                           # compute y coordinates
    return x_new, y_new


def wrap_tri_grid_around_circle_iterative(design_spacing, inner_radius, outer_radius):
    """
    Wraps a set of triagnular gridded points, with uniform spacing, around a circle of inner radius. As the points go
    further out in radius, the uniform triangular grid spacing is not maintained (ie some points are further away from
    their neighbors than designed with the triangular grid). An iterative method is used to adjust the actual spacing
    between points in the grid is that of the design spacing.


    :param design_spacing: desired avg spacing between points
    :param inner_radius: inner radius of ring
    :param outer_radius: outer radius of ring
    :return: x points, y points, new spacing to get the desired average spacing, and percent error
    """


    # goal function minimization... want to minimize the difference between avg hole spacing and designed hole spacing
    def goal_function(s_new):

        # make triangle grid, wrap around circle, reshape into grid
        temp_grid = triangular_grid_in_rectangle(spacing=s_new,
                                                 x_lim=2 * np.pi * inner_radius - 1.7 * s_new,
                                                 y_lim=outer_radius,
                                                 start_point=(0, inner_radius))
        x_new, y_new = wrap_points_around_circle(temp_grid[:, 0], temp_grid[:, 1])
        x_new = np.reshape(x_new, (len(x_new), 1))
        y_new = np.reshape(y_new, (len(y_new), 1))
        temp_grid = np.concatenate((x_new, y_new), axis=1)

        # find avg spacing between each hole and its 2nd nearest neighbor (first is itself)
        k = KDTree(temp_grid)
        (dists, idx) = k.query(temp_grid, 2)
        avg_spacing = np.mean(dists[:, 1])

        # return percent difference between the avg spacing and design spacing
        return np.abs(avg_spacing - design_spacing) / design_spacing * 100


    # minimize difference between avg spacing and designed spacing
    res = minimize_scalar(fun=goal_function,
                          bracket=(design_spacing*0.1, design_spacing, design_spacing*1.9))
    spacing_new = res.x  # retrieve optimal spacing

    # create new grid
    corrected_grid = triangular_grid_in_rectangle(spacing=spacing_new,
                                                  x_lim=2 * np.pi * inner_radius - 1.7 * spacing_new,
                                                  y_lim = outer_radius,
                                                  start_point=(0, inner_radius))
    x_new, y_new = wrap_points_around_circle(corrected_grid[:, 0], corrected_grid[:, 1])
    x_new = np.reshape(x_new, (len(x_new), 1))
    y_new = np.reshape(y_new, (len(y_new), 1))

    # return new x points, y points, spacing, and percent error
    return x_new, y_new, spacing_new, res.fun


def triangular_grid_hex_expansion(point_distance, expansion_level=4, starting_point=(0, 0)):
    """
    Lovingly stolen from: https://github.com/Panteleimon/Python-Triangular-Lattice (1/15/2021) <3

    :param point_distance: distance between grid points
    :param expansion_level: number of points from the center to the outside of the hexagon on a diagonal line,
    excluding the center point
    :param starting_point: middle point of grid
    :return: np array of triangular grid points [X, Y]
    """

    set_of_points_on_the_plane = set()

    for current_level in range(expansion_level + 1):

        temporary_hexagon_coordinates = {}

        equilateral_triangle_side = current_level * point_distance
        equilateral_triangle__half_side = equilateral_triangle_side / 2
        equilateral_triangle_height = (np.sqrt(3) * equilateral_triangle_side) / 2
        if current_level != 0:
            point_distance_as_triangle_side_percentage = point_distance / equilateral_triangle_side

        temporary_hexagon_coordinates['right'] = (
        starting_point[0] + point_distance * current_level, starting_point[1])  # right
        temporary_hexagon_coordinates['left'] = (
        starting_point[0] - point_distance * current_level, starting_point[1])  # left
        temporary_hexagon_coordinates['top_right'] = (starting_point[0] + equilateral_triangle__half_side,
                                                      starting_point[1] + equilateral_triangle_height)  # top_right
        temporary_hexagon_coordinates['top_left'] = (starting_point[0] - equilateral_triangle__half_side,
                                                     starting_point[1] + equilateral_triangle_height)  # top_left
        temporary_hexagon_coordinates['bottom_right'] = (starting_point[0] + equilateral_triangle__half_side,
                                                         starting_point[
                                                             1] - equilateral_triangle_height)  # bottom_right
        temporary_hexagon_coordinates['bottom_left'] = (starting_point[0] - equilateral_triangle__half_side,
                                                        starting_point[1] - equilateral_triangle_height)  # bottom_left

        if current_level > 1:
            for intermediate_points in range(1, current_level):
                set_of_points_on_the_plane.add((temporary_hexagon_coordinates['left'][
                                                    0] + intermediate_points * point_distance_as_triangle_side_percentage * abs(
                    temporary_hexagon_coordinates['top_left'][0] - temporary_hexagon_coordinates['left'][0]),
                                                temporary_hexagon_coordinates['left'][
                                                    1] + intermediate_points * point_distance_as_triangle_side_percentage * abs(
                                                    temporary_hexagon_coordinates['top_left'][1] -
                                                    temporary_hexagon_coordinates['left'][1])))  # from left to top left
                # print(intermediate_points)
                # print((temporary_hexagon_coordinates['left'][0] + intermediate_points * point_distance_as_triangle_side_percentage * abs(temporary_hexagon_coordinates['top_left'][0] - temporary_hexagon_coordinates['left'][0]) , temporary_hexagon_coordinates['left'][1] + intermediate_points * point_distance_as_triangle_side_percentage * abs(temporary_hexagon_coordinates['top_left'][1] - temporary_hexagon_coordinates['left'][1])  ))
                set_of_points_on_the_plane.add((temporary_hexagon_coordinates['left'][
                                                    0] + intermediate_points * point_distance_as_triangle_side_percentage * abs(
                    temporary_hexagon_coordinates['bottom_left'][0] - temporary_hexagon_coordinates['left'][0]),
                                                temporary_hexagon_coordinates['left'][
                                                    1] - intermediate_points * point_distance_as_triangle_side_percentage * abs(
                                                    temporary_hexagon_coordinates['bottom_left'][1] -
                                                    temporary_hexagon_coordinates['left'][
                                                        1])))  # from left to bottom left

                set_of_points_on_the_plane.add((temporary_hexagon_coordinates['top_right'][
                                                    0] - intermediate_points * point_distance_as_triangle_side_percentage * abs(
                    temporary_hexagon_coordinates['top_left'][0] - temporary_hexagon_coordinates['top_right'][0]),
                                                temporary_hexagon_coordinates['top_right'][
                                                    1]))  # from top right to top left
                set_of_points_on_the_plane.add((temporary_hexagon_coordinates['top_right'][
                                                    0] + intermediate_points * point_distance_as_triangle_side_percentage * abs(
                    temporary_hexagon_coordinates['right'][0] - temporary_hexagon_coordinates['top_right'][0]),
                                                temporary_hexagon_coordinates['top_left'][
                                                    1] - intermediate_points * point_distance_as_triangle_side_percentage * abs(
                                                    temporary_hexagon_coordinates['right'][1] -
                                                    temporary_hexagon_coordinates['top_right'][
                                                        1])))  # from top right to right

                set_of_points_on_the_plane.add((temporary_hexagon_coordinates['bottom_right'][
                                                    0] - intermediate_points * point_distance_as_triangle_side_percentage * abs(
                    temporary_hexagon_coordinates['bottom_left'][0] - temporary_hexagon_coordinates['bottom_right'][0]),
                                                temporary_hexagon_coordinates['bottom_right'][
                                                    1]))  # apo bottom right pros aristera
                set_of_points_on_the_plane.add((temporary_hexagon_coordinates['bottom_right'][
                                                    0] + intermediate_points * point_distance_as_triangle_side_percentage * abs(
                    temporary_hexagon_coordinates['right'][0] - temporary_hexagon_coordinates['bottom_right'][0]),
                                                temporary_hexagon_coordinates['bottom_right'][
                                                    1] + intermediate_points * point_distance_as_triangle_side_percentage * abs(
                                                    temporary_hexagon_coordinates['right'][1] -
                                                    temporary_hexagon_coordinates['bottom_right'][
                                                        1])))  # from bottom right to right

        # dictionary to set
        set_of_points_on_the_plane.update(temporary_hexagon_coordinates.values())

    return np.array(list(set_of_points_on_the_plane))
