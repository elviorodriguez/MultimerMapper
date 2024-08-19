#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 26 17:11:26 2024

@author: elvio
"""

import numpy as np

# ########################################################################
# ####################### Oscillation using lines ########################
# ########################################################################

def oscillate_line(start_point: np.array,
                   end_point: np.array,
                   
                   # Oscillation parameters
                   amplitude: float | int = 0.02,
                   frequency: int = 8,
                   
                   # Resolution parameter
                   additional_points = 200):
    
    # Generate additional points along the edge
    intermediate_x = np.linspace(start_point[0], end_point[0], additional_points + 2)
    intermediate_y = np.linspace(start_point[1], end_point[1], additional_points + 2)
    
    # Generate perpendicular oscillation
    line_length = np.hypot(end_point[0] - start_point[0], end_point[1] - start_point[1])
    oscillation = amplitude * np.sin(frequency * np.linspace(0, 4 * np.pi, additional_points + 2))
    
    # Calculate perpendicular direction
    dx = (end_point[1] - start_point[1]) / line_length
    dy = (start_point[0] - end_point[0]) / line_length
    
    # Apply oscillation perpendicular to the line
    oscillated_x = intermediate_x + oscillation * dx
    oscillated_y = intermediate_y + oscillation * dy
    
    return oscillated_x, oscillated_y


# ---------------- Testing and debugging ----------------

# import matplotlib.pyplot as plt

# # Starting and end points
# additional_points = 200
# start_point = np.array([0, 0])
# end_point = np.array([10, 10])

# # Generate additional points along the edge
# intermediate_x = np.linspace(start_point[0], end_point[0], additional_points + 2)
# intermediate_y = np.linspace(start_point[1], end_point[1], additional_points + 2)

# oscillated_x, oscillated_y = oscillate_line(start_point, end_point)

# # Plot the original array
# plt.plot(intermediate_x, intermediate_y,
#          label='Original Array',
#          linestyle='--', color='gray')

# # Plot the oscillated array
# plt.plot(oscillated_x, oscillated_y,
#          label='Oscillated Array', color='blue')

# plt.legend()
# plt.show()

# ########################################################################
# ###################### Oscillation using circles #######################
# ########################################################################


def oscillate_circle(reference_point: np.array,
                     radius: int | float,
                     amplitude = 0.02,
                     frequency = 20,
                     resolution = 200):
    
    # Get "resolution" values of pi
    theta = np.linspace(0, 2 * np.pi, resolution)
    
    # Generate radial oscillation
    circle_oscillation = amplitude * np.sin(frequency * theta)
    
    # Apply oscillation radially
    circle_oscillated_x = reference_point[0] + (radius + circle_oscillation) * np.cos(theta)
    circle_oscillated_y = reference_point[1] + (radius + circle_oscillation) * np.sin(theta)
    
    return circle_oscillated_x, circle_oscillated_y

# ---------------- Testing and debugging ----------------

# circle_oscillated_x, circle_oscillated_y = oscillate_circle(start_point = start_point,
#                      radius = radius)

# # Plot the oscillated circle
# plt.plot(circle_oscillated_x, circle_oscillated_y,
#          label='Oscillated Circle',
#          color='blue')

# plt.legend()
# plt.show()



# # Generate 100 radian points
# theta = np.linspace(0, 2 * np.pi, additional_points)
# radius = 5
# amplitude = 0.2
# frequency = 20

# # Adjust the position of the circle
# circle_x = start_point[0] + radius * np.cos(theta)
# circle_y = start_point[1] + radius * np.sin(theta)

# # Plot the original circle
# plt.plot(circle_x, circle_y,
#          label='Original Circle',
#          linestyle='--', color='gray')

# # Generate radial oscillation
# circle_oscillation = amplitude * np.sin(frequency * theta)

# # Apply oscillation radially
# circle_oscillated_x = start_point[0] + (radius + circle_oscillation) * np.cos(theta)
# circle_oscillated_y = start_point[1] + (radius + circle_oscillation) * np.sin(theta)

# # Plot the oscillated circle
# plt.plot(circle_oscillated_x, circle_oscillated_y,
#          label='Oscillated Circle',
#          color='blue')

# plt.legend()
# plt.show()

# ########################################################################
# ###################### Parabolas and oscillation #######################
# ########################################################################


def generate_parabolic_points(start, end, valency, resolution=200):
    # Determine direction based on valency (odd: positive, even: negative)
    direction = 1 if valency % 2 != 0 else -1
    
    # Calculate the maximum shift based on valency
    max_shift = (valency + 1) // 2 * direction * 0.1  # Adjust 0.1 to change the curve intensity
    
    # Generate t values from 0 to 1
    t = np.linspace(0, 1, resolution)
    
    # Calculate the midpoint
    mid = (start + end) / 2
    
    # Calculate the perpendicular vector
    perp = np.array([-(end[1] - start[1]), end[0] - start[0]])
    perp = perp / np.linalg.norm(perp)
    
    # Generate points along the parabola
    points = (1-t)[:, np.newaxis] * start + t[:, np.newaxis] * end + \
             (4 * max_shift * t * (1-t))[:, np.newaxis] * perp
    
    return points



def generate_oscillating_parabolic_points(start_point, end_point, valency, 
                                          amplitude=0.02, frequency=8, resolution=200):
    # Determine direction based on valency (odd: positive, even: negative)
    direction = 1 if valency % 2 != 0 else -1
    
    # Calculate the maximum shift based on valency
    max_shift = (valency + 1) // 2 * direction * 0.1  # Adjust 0.1 to change the curve intensity
    
    # Generate t values from 0 to 1
    t = np.linspace(0, 1, resolution)
    
    # Calculate the midpoint
    mid = (start_point + end_point) / 2
    
    # Calculate the perpendicular vector
    perp = np.array([-(end_point[1] - start_point[1]), end_point[0] - start_point[0]])
    perp = perp / np.linalg.norm(perp)
    
    # Generate points along the parabola
    points = (1-t)[:, np.newaxis] * start_point + t[:, np.newaxis] * end_point + \
             (4 * max_shift * t * (1-t))[:, np.newaxis] * perp
    
    # Calculate the length of the parabolic curve
    lengths = np.cumsum(np.sqrt(np.sum(np.diff(points, axis=0)**2, axis=1)))
    total_length = lengths[-1]
    
    # Generate oscillation
    oscillation = amplitude * np.sin(frequency * np.linspace(0, 4 * np.pi, resolution))
    
    # Calculate tangent and normal vectors along the curve
    tangents = np.diff(points, axis=0)
    tangents = np.vstack([tangents, tangents[-1]])  # Repeat last tangent for consistency
    tangents = tangents / np.linalg.norm(tangents, axis=1)[:, np.newaxis]
    normals = np.array([-tangents[:, 1], tangents[:, 0]]).T
    
    # Apply oscillation perpendicular to the curve
    oscillated_points = points + (oscillation[:, np.newaxis] * normals)
    
    return oscillated_points[:, 0], oscillated_points[:, 1]