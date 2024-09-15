
import pandas as pd
import numpy as np
import igraph
import plotly.graph_objects as go
from plotly.offline import plot
from Bio.PDB import Chain
from Bio.SeqUtils import seq1

from utils.pdb_utils import calculate_distance
from src.detect_domains import plot_backbone

#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

# -----------------------------------------------------------------------------
# Functions to apply rotation and translations to cloud points and CMs --------
# -----------------------------------------------------------------------------

def normalize_vector(v):
    """Normalize a vector."""
    return v / np.linalg.norm(v)

def center_of_mass(points):
   
    x_list = []
    y_list = []
    z_list = []
    
    for point in points:
        
        x_list.append(point[0])
        y_list.append(point[1])
        z_list.append(point[2])
        
    return np.array([np.mean(x_list), np.mean(y_list), np.mean(z_list)])

def rotate_points(points, v1_cloud_direction, v2_to_match, reference_point=None):
    """
    Rotate a cloud of points around an axis defined by the vectors v1 and v2.
    
    Parameters:
    - points: Numpy array of shape (N, 3) representing the cloud of points.
    - v1: Initial vector defining the rotation axis.
    - v2: Target vector defining the rotation axis.
    - reference_point: Reference point for the rotation. If None, rotation is around the origin.
    :return: Rotated points

    The function calculates the rotation matrix needed to rotate vector v1 to align
    with vector v2. It then applies this rotation matrix to the cloud of points,
    resulting in a rotated set of points.

    If a reference point is provided, the rotation is performed around this point.
    Otherwise, rotation is around the origin.

    Note: The vectors v1 and v2 are assumed to be non-collinear.
    """
    
    # Helper
    def rotation_matrix_from_vectors(v1, v2):
        """
        Compute the rotation matrix that rotates vector v1 to align with vector v2.
        :param v1: Initial vector
        :param v2: Target vector
        :return: Rotation matrix
        """
        v1 = normalize_vector(v1)
        v2 = normalize_vector(v2)

        cross_product = np.cross(v1, v2)
        dot_product = np.dot(v1, v2)
        
        skew_matrix = np.array([[0, -cross_product[2], cross_product[1]],
                                [cross_product[2], 0, -cross_product[0]],
                                [-cross_product[1], cross_product[0], 0]])

        rotation_matrix = np.eye(3) + skew_matrix + np.dot(skew_matrix, skew_matrix) * (1 - dot_product) / np.linalg.norm(cross_product)**2
        
        return rotation_matrix
    
    rotation_matrix = rotation_matrix_from_vectors(v1_cloud_direction, v2_to_match)
    
    # With reference point (CM != (0,0,0))
    if reference_point is not None:
        
        # vector to translate the points to have as reference the origin
        translation_vector = - reference_point
        
        # Check if single 3D point/vector is passed
        if type(points) == np.ndarray and np.ndim(points) == 1 and len(points) == 3:
            rotated_points = np.dot(rotation_matrix, (points + translation_vector).T).T - translation_vector
        
        # Multiple 3D points/vectors
        else:
            # Initialize results array
            rotated_points = []
            
            for point in points:
                # Translate points to origin, rotate, and translate back
                rotated_point = np.dot(rotation_matrix, (point + translation_vector).T).T - translation_vector
                rotated_points.append(np.array(rotated_point))
    
    # No reference point
    else:
        # Check if single 3D point/vector is passed
        if type(points) == np.ndarray and np.ndim(points) == 1 and len(points) == 3:
            rotated_points = np.dot(rotation_matrix, points.T).T
        
        # Multiple 3D points/vectors 
        else: 
            # Initialize results array
            rotated_points = []
            
            for point in points:
                # Rotate around the origin
                rotated_point = np.dot(rotation_matrix, point.T).T
                rotated_points.append(rotated_point)
        
    return np.array(rotated_points, dtype = "float32")

def translate_points(points, v_direction, distance, is_CM = False):
    """
    Translate a cloud of points in the direction of the given vector by the specified distance.
    
    Parameters:
    - points: Numpy array of shape (N, 3) representing the cloud of points.
    - v_direction: Numpy array of shape (3,) representing the translation direction.
    - distance: Distance to translate each point along the vector (Angstroms).
    - is_CM: set to True to translate CM (center of mass) points

    Returns:
    Numpy array of shape (N, 3) representing the translated points.
    """
    normalized_vector = normalize_vector(v_direction)
    translation_vector = distance * normalized_vector
    if is_CM:
        # Translate the individual point
        translated_points = points + translation_vector
    else:
        # Translate one point at a time
        translated_points = [point + translation_vector for point in points]
    return translated_points

def precess_points(points, angle, reference_point=None):
    """
    Precesses a cloud of points around the axis defined between the reference point
    and the center of mass of the points. If no reference point is provided,
    the origin is taken as a reference point.

    Parameters:
    - points: Numpy array of arrays of shape (1, 3) representing the cloud of points.
              Each index in the outer numpy array represents a point.
    - angle: Rotation angle in degrees.
    - reference_point: Reference point for the rotation. If None, the reference point
              will be set at the origin (0, 0, 0).

    Returns:
    - Precessed points in the same format as input.

    The function rotates the cloud of points around the axis defined by the vector
    reference_point -> center_of_mass(points) by a specified angle.
    If a reference point is not provided, the origin of the precession vector is
    located at the point (0, 0, 0).
    """
    # Convert angle to radians
    angle = np.radians(angle)

    # If no reference point is provided, use the origin
    if reference_point is None:
        reference_point = np.array([0.0, 0.0, 0.0])

    # Convert points to a 2D NumPy array
    points_array = np.array([np.array(point) for point in points])

    # Calculate the center of mass of the points
    center_of_mass = np.mean(points_array, axis=0)

    # Calculate the rotation axis
    rotation_axis = center_of_mass - reference_point

    # Normalize the rotation axis
    rotation_axis /= np.linalg.norm(rotation_axis)

    # Create a rotation matrix
    rotation_matrix = np.array([[np.cos(angle) + rotation_axis[0]**2 * (1 - np.cos(angle)),
                                 rotation_axis[0] * rotation_axis[1] * (1 - np.cos(angle)) - rotation_axis[2] * np.sin(angle),
                                 rotation_axis[0] * rotation_axis[2] * (1 - np.cos(angle)) + rotation_axis[1] * np.sin(angle)],
                                [rotation_axis[1] * rotation_axis[0] * (1 - np.cos(angle)) + rotation_axis[2] * np.sin(angle),
                                 np.cos(angle) + rotation_axis[1]**2 * (1 - np.cos(angle)),
                                 rotation_axis[1] * rotation_axis[2] * (1 - np.cos(angle)) - rotation_axis[0] * np.sin(angle)],
                                [rotation_axis[2] * rotation_axis[0] * (1 - np.cos(angle)) - rotation_axis[1] * np.sin(angle),
                                 rotation_axis[2] * rotation_axis[1] * (1 - np.cos(angle)) + rotation_axis[0] * np.sin(angle),
                                 np.cos(angle) + rotation_axis[2]**2 * (1 - np.cos(angle))]])
    
    # # Debug
    # print("points_array")
    # print(points_array)
    # print("reference_point")
    # print(reference_point)
    # print("rotation_matrix")
    # print(rotation_matrix)
    
    # Apply the rotation to the points
    rotated_points = np.dot(points_array - reference_point, rotation_matrix.T) + reference_point

    return [np.array(point, dtype = "float32") for point in rotated_points]


def precess_until_minimal_distance(points_1, other_points_2, reference_point=None, angle_steps = 5):
    """
    Precesses a cloud of points around the axis defined between the reference point
    and the center of mass of the points. If no reference point is provided,
    the origin is taken as a reference point. The precession is performed step
    by step, until the average distance between contact points is minimal.

    Parameters:
    - points: Numpy array of arrays of shape (1, 3) representing the cloud of points.
              Each index in the outer numpy array represents a point.
    - other_points: contact points over the surface of another partner protein.
              The algorithm tries to minimize residue-reside distance between contacts.
    - reference_point: Reference point for the rotation. If None, the reference point
              will be set at the origin (0, 0, 0).

    Returns:
    - Precessed points in the same format as input.

    The function rotates the cloud of points around the axis defined by the vector
    reference_point -> center_of_mass(points).
    If a reference point is not provided, the origin of the precession vector is
    located at the point (0, 0, 0). The precession is performed step
    by step, until the average distance between contact points is minimal.
    """
    # # Progress
    # print("Precessing cloud of points...")
    
    def calculate_total_distance(points_1, other_points_2):
        list_of_distances = [calculate_distance(points_1[i], other_points_2[i]) for i in range(len_points_1)]
        sum_of_distances = sum(list_of_distances)
        return sum_of_distances
    
    # Check if lengths are equal
    len_points_1 = len(points_1)
    len_points_2 = len(other_points_2)
    
    # Manage the case when lengths are different
    if len_points_1 != len_points_2:
        raise ValueError("The case in which points_1 and other_points_2 have different lengths, is not implemented yet.")    
    
    angles_list = [0]
    points_cloud_by_angle = [points_1]
    sum_of_distances_by_angle = [calculate_total_distance(points_1, other_points_2)]
    
    # Compute the distance by precessing the points angle_steps at a time    
    for angle in range(angle_steps, 360, angle_steps):
        
        # Precess the cloud of points
        precessed_cloud = precess_points(points = points_1,
                                         angle = angle,
                                         reference_point = reference_point)        
        # Compute the new distance for the precessed cloud
        new_sum_of_distances = calculate_total_distance(precessed_cloud, other_points_2)
        
        # Store the data
        angles_list.append(angle)
        points_cloud_by_angle.append(precessed_cloud)
        sum_of_distances_by_angle.append(new_sum_of_distances)
        
    # Find the lowest distance cloud and 
    minimum_index = sum_of_distances_by_angle.index(min(sum_of_distances_by_angle))
    
    print(f"   - Best angle is {angles_list[minimum_index]}")
    print( "   - Starting distance:", sum_of_distances_by_angle[0])
    print(f"   - Minimum distance: {sum_of_distances_by_angle[minimum_index]}")
    
    # Return the precessed cloud of points with the lowest distance to their contacts
    return points_cloud_by_angle[minimum_index]



# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------

#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

# -----------------------------------------------------------------------------
# Plot the results of rotations and translations ------------------------------
# -----------------------------------------------------------------------------


# Uses plotly
def plot_vectors(vectors, sub_dict=False, contacts = None, show_plot = True,
                 custom_colors = None, Psurf_size = 5, Psurf_line_width = 5,
                 title = "vectors"):
    '''
    Plots a list/np.array/dict of 3D vectors (1x3 np.arrays) using Plotly.
    If a dict is given, the names of each vector will be used as labels.
    If different clouds of points are used in a dict, with each cloud of points
    as a different key, set sub_dict as True.

    Parameters
    ----------
    vectors : list/dict
        A collection of 3D vectors.
        
        Example of dict structure:
        
        dict_points_cloud_ab = {
            
            # Protein A
            "A": {"points_cloud": points_cloud_Ab_rot_trn,      # list of surface points
                  "R": residue_names_a,                         # list of residue names
                  "CM": CM_a_trn},                              # array with xyz
            
            # Protein B
            "B": {"points_cloud": points_cloud_Ba,
                  "R": residue_names_b,
                  "CM": CM_b}
            }

    sub_dict : bool, optional
        If True, divides groups of vectors (or cloud points).
        
    interactions : 
        NOT IMPLEMENTED
    
    show_plot : bool
        If False, the plot will only be returned. If True, it will also be
        displayed in your browser.

    Returns
    -------
    plotly.graph_objs._figure.Figure
        3D plot of the vectorized contact residues of the proteins defined in 
        vectors.

    '''
    
    colors = ["red", "green", "blue", "orange", "violet", "black", "brown",
              "bisque", "blanchedalmond", "blueviolet", "burlywood", "cadetblue"]
    
    if custom_colors != None:
        colors = custom_colors
        
    # Create a 3D scatter plot
    fig = go.Figure()

    if type(vectors) == dict:
        if sub_dict:
            # Set to store unique labels and avoid repeated labels
            unique_labels = set()

            # Plot each vector with label and color
            for i, (label, sub_vectors) in enumerate(vectors.items()):

                # Plot one vector at a time
                for n, vector in enumerate(sub_vectors["points_cloud"]):
                    fig.add_trace(go.Scatter3d(
                        x=[sub_vectors["CM"][0], vector[0]],
                        y=[sub_vectors["CM"][1], vector[1]],
                        z=[sub_vectors["CM"][2], vector[2]],
                        mode='lines+markers',
                        marker=dict(
                            size = Psurf_size,
                            color = colors[i % len(colors)]
                        ),
                        line=dict(
                            color = colors[i % len(colors)],
                            width = Psurf_line_width
                        ),
                        name = label,
                        showlegend = label not in unique_labels,
                        # hovertext=[f'Point {i+1}' for i in range(len(sub_vectors["points_cloud"]))]
                        hovertext=sub_vectors["R"][n]
                    ))
                    unique_labels.add(label)
            
            # Plot residue-residue contacts
            if contacts != None:
                
                # Convert array points to lists
                points_A_list = [tuple(point) for point in contacts[0]]
                points_B_list = [tuple(point) for point in contacts[1]]
                
                # Unpack points for Scatter3d trace
                x_A, y_A, z_A = zip(*points_A_list)
                x_B, y_B, z_B = zip(*points_B_list)
                
                # Add one contact at a time
                for contact_index in range(len(contacts[0])):
                    # Create the Scatter3d trace for lines
                    fig.add_trace(go.Scatter3d(
                        x = (x_A[contact_index],) + (x_B[contact_index],) + (None,),  # Add None to create a gap between points_A and points_B
                        y = (y_A[contact_index],) + (y_B[contact_index],) + (None,),
                        z = (z_A[contact_index],) + (z_B[contact_index],) + (None,),
                        mode='lines',
                        line=dict(color='gray', width=1, dash='dash'), # 
                        showlegend = False,
                        name = contacts[2][contact_index]
                        ))

        else:
            # Plot each vector with label and color
            for i, (label, vector) in enumerate(vectors.items()):
                fig.add_trace(go.Scatter3d(
                    x=[0, vector[0]],
                    y=[0, vector[1]],
                    z=[0, vector[2]],
                    mode='lines+markers',
                    marker=dict(
                        size = Psurf_size,
                        color = colors[i % len(colors)]
                    ),
                    line=dict(
                        color = colors[i % len(colors)],
                        width = Psurf_line_width
                    ),
                    name=label
                ))

    elif type(vectors) == list or type(vectors) == np.ndarray:
        # Plot each vector
        for i, vector in enumerate(vectors):
            fig.add_trace(go.Scatter3d(
                x=[0, vector[0]],
                y=[0, vector[1]],
                z=[0, vector[2]],
                mode='lines+markers',
                marker=dict(
                    size = Psurf_size,
                    color=colors[i % len(colors)]
                ),
                line=dict(
                    color=colors[i % len(colors)],
                    width = Psurf_line_width
                )
            ))

    else:
        raise ValueError("vectors data structure not supported")

    # Set layout
    fig.update_layout(scene=dict(
        xaxis_title='X',
        yaxis_title='Y',
        zaxis_title='Z',
         
    ), title = title)
    
    # Display the plot?
    if show_plot == True: plot(fig)
    
    return fig

# # Plot the vectors
# vectors_dict = {"Vab": {"points_cloud": [Vab],
#                         "CM": CM_a},
#                 "Vba": {"points_cloud": [Vba],
#                         "CM": CM_b},
#                 "Vba_minus": {"points_cloud": [Vba_minus],
#                               "CM": CM_b},
#                 "Vab_rot": {"points_cloud": [Vab_rot],
#                             "CM": CM_a},
#                 "Vab_rot_trn": {"points_cloud": [Vab_rot_trn],
#                                 "CM": CM_a_trn}
#                 }
# plot_vectors(vectors_dict, sub_dict = True)

# # Plot points cloud
# plot_vectors(points_cloud_Ab)


# # Plot 2 points of clouds with 1 rotated, aligned and translated. Then save as HTML
# dict_points_cloud_ab = {
#     "A": {"points_cloud": points_cloud_Ab_rot_trn,
#           "R": residue_names_a,
#           "CM": CM_a_trn}, 
#     "B": {"points_cloud": points_cloud_Ba,
#           "R": residue_names_b,
#           "CM": CM_b},
#     "A_2":
#         {"points_cloud": points_cloud_Ab_rot_trn_prec,
#          "R": residue_names_a,
#          "CM": CM_a_trn}
#     }
# fig = plot_vectors(dict_points_cloud_ab,
#                    sub_dict = True,
#                    contacts = contacts,
#                    show_plot= True)

# modify axis limits
# fig.update_layout(scene=dict(
#     xaxis=dict(range=[-100, +100]),  # Specify your desired x-axis limits
#     yaxis=dict(range=[-100, +100]),  # Specify your desired y-axis limits
#     zaxis=dict(range=[-100, +100]),  # Specify your desired z-axis limits
#     xaxis_title='X',
#     yaxis_title='Y',
#     zaxis_title='Z'
# ))
# plot(fig)

# fig.write_html('./example_contact_plot.html')

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------

#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

# -----------------------------------------------------------------------------
# Convert everything to object oriented programming (OOP) ---------------------
# -----------------------------------------------------------------------------

def vector_magnitude(v):
    return np.linalg.norm(v)

def are_vectors_collinear(v1, v2, atol = 0.0001):
    '''v1 and v2 are x,y,z values as np.arrays. They need to be centered in origin.'''
    # Calculate the cross product
    cross_product = np.cross(v1, v2)

    # Check if the cross product is the zero vector
    return np.allclose(cross_product, [0, 0, 0], atol = atol)


def precess_points2(points, angle, self_surface_residues_CM, reference_point=None):
    """
    Precesses a cloud of points around the axis defined between the reference point
    and the center of mass of the points. If no reference point is provided,
    the origin is taken as a reference point.

    Parameters:
    - points: Numpy array of arrays of shape (1, 3) representing the cloud of points.
              Each index in the outer numpy array represents a point.
    - angle: Rotation angle in degrees.
    - reference_point: Reference point for the rotation. If None, the reference point
              will be set at the origin (0, 0, 0).

    Returns:
    - Precessed points in the same format as input.

    The function rotates the cloud of points around the axis defined by the vector
    reference_point -> center_of_mass(points) by a specified angle.
    If a reference point is not provided, the origin of the precession vector is
    located at the point (0, 0, 0).
    """
    # Convert angle to radians
    angle = np.radians(angle)

    # If no reference point is provided, use the origin
    if reference_point is None:
        reference_point = np.array([0.0, 0.0, 0.0])

    # Convert points to a 2D NumPy array
    points_array = np.array([np.array(point) for point in points])

    # Calculate the rotation axis
    rotation_axis = self_surface_residues_CM - reference_point

    # Normalize the rotation axis
    rotation_axis /= np.linalg.norm(rotation_axis)

    # Create a rotation matrix
    rotation_matrix = np.array([[np.cos(angle) + rotation_axis[0]**2 * (1 - np.cos(angle)),
                                 rotation_axis[0] * rotation_axis[1] * (1 - np.cos(angle)) - rotation_axis[2] * np.sin(angle),
                                 rotation_axis[0] * rotation_axis[2] * (1 - np.cos(angle)) + rotation_axis[1] * np.sin(angle)],
                                [rotation_axis[1] * rotation_axis[0] * (1 - np.cos(angle)) + rotation_axis[2] * np.sin(angle),
                                 np.cos(angle) + rotation_axis[1]**2 * (1 - np.cos(angle)),
                                 rotation_axis[1] * rotation_axis[2] * (1 - np.cos(angle)) - rotation_axis[0] * np.sin(angle)],
                                [rotation_axis[2] * rotation_axis[0] * (1 - np.cos(angle)) - rotation_axis[1] * np.sin(angle),
                                 rotation_axis[2] * rotation_axis[1] * (1 - np.cos(angle)) + rotation_axis[0] * np.sin(angle),
                                 np.cos(angle) + rotation_axis[2]**2 * (1 - np.cos(angle))]])

    # Apply the rotation to the points
    rotated_points = np.dot(points_array - reference_point, rotation_matrix.T) + reference_point

    return [np.array(point, dtype = "float32") for point in rotated_points]



def precess_until_minimal_distance2(points, self_contact_points_1, partner_contact_points_2, reference_point=None, angle_steps = 5):
    """
    Precesses a cloud of points around the axis defined between the reference point
    and the center of mass of the points. If no reference point is provided,
    the origin is taken as a reference point. The precession is performed step
    by step, until the average distance between contact points is minimal.

    Parameters:
    - points: Numpy array of arrays of shape (1, 3) representing the cloud of points.
              Each index in the outer numpy array represents a point.
    - other_points: contact points over the surface of another partner protein.
              The algorithm tries to minimize residue-reside distance between contacts.
    - reference_point: Reference point for the rotation. If None, the reference point
              will be set at the origin (0, 0, 0).

    Returns:
    - Precessed points in the same format as input.

    The function rotates the cloud of points around the axis defined by the vector
    reference_point -> center_of_mass(points).
    If a reference point is not provided, the origin of the precession vector is
    located at the point (0, 0, 0). The precession is performed step
    by step, until the average distance between contact points is minimal.
    """
    # # Progress
    # print("")
    # print("   - Precessing cloud of points...")
    
    def calculate_total_distance(self_contact_points_1, partner_contact_points_2):
        list_of_distances = [calculate_distance(self_contact_points_1[i], partner_contact_points_2[i]) for i in range(len_points_1)]
        sum_of_distances = sum(list_of_distances)
        return sum_of_distances
    
    # Check if lengths are equal
    len_points_1 = len(self_contact_points_1)
    len_points_2 = len(partner_contact_points_2)
    
    # Manage the case when lengths are different
    if len_points_1 != len_points_2:
        raise ValueError("The case in which self_contact_points_1 and partner_contact_points_2 have different lengths, is not implemented yet.")    
    
    angles_list = [0]
    points_cloud_by_angle = [self_contact_points_1]
    sum_of_distances_by_angle = [calculate_total_distance(self_contact_points_1, partner_contact_points_2)]
    
    # Compute the distance by precessing the points angle_steps at a time    
    for angle in range(angle_steps, 360, angle_steps):
        
        # Precess the cloud of points
        precessed_cloud = precess_points(points = self_contact_points_1,
                                         angle = angle,
                                         reference_point = reference_point)
        # Compute the new distance for the precessed cloud
        new_sum_of_distances = calculate_total_distance(precessed_cloud, partner_contact_points_2)
        
        # Store the data
        angles_list.append(angle)
        points_cloud_by_angle.append(precessed_cloud)
        sum_of_distances_by_angle.append(new_sum_of_distances)
        
    # Find the lowest distance cloud and 
    minimum_index = sum_of_distances_by_angle.index(min(sum_of_distances_by_angle))
    
    print(f"   - Best angle is {angles_list[minimum_index]}")
    print( "   - Starting distance:", sum_of_distances_by_angle[0])
    print(f"   - Minimum distance: {sum_of_distances_by_angle[minimum_index]}")
    
    # Return the precessed cloud of points with the lowest distance to their contacts
    return precess_points2(points = points,
                           self_surface_residues_CM= center_of_mass(self_contact_points_1),
                           angle = angles_list[minimum_index],
                           reference_point = reference_point)

def scale_vector(v, scale_factor):
    return scale_factor * v

def find_vector_with_length(v1, desired_length):
    
    # Calculate the current length of v1
    current_length = np.linalg.norm(v1)

    # Calculate the scaling factor
    scale_factor = desired_length / current_length

    # Scale the vector to the desired length
    v2 = scale_vector(v1, scale_factor)

    return v2

def angle_between_vectors(v1, v2):
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)

    # Ensure denominators are not zero
    if norm_v1 == 0 or norm_v2 == 0:
        raise ValueError("Input vectors must have non-zero length.")

    cosine_theta = dot_product / (norm_v1 * norm_v2)
    angle_radians = np.arccos(np.clip(cosine_theta, -1.0, 1.0))
    
    # Convert angle to degrees
    angle_degrees = np.degrees(angle_radians)
    
    return angle_degrees

# Function to insert None row between each row
def insert_none_row(df):
    try:
        none_row = pd.DataFrame([[None] * len(df.columns)], columns=df.columns)
        result_df = pd.concat([pd.concat([none_row, row.to_frame().T], ignore_index=True) for _, row in df.iterrows()], ignore_index=True)
        result_df = pd.concat([result_df, none_row, none_row], ignore_index=True)
    except:
        return df
    return result_df

# Function to insert None row between each row and copy color values
def insert_none_row_with_color(df):
    result_df = insert_none_row(df)
    
    prev_color = None
    for idx, row in result_df.iterrows():
        if not row.isnull().all():
            prev_color = row['color']
        elif prev_color is not None:
            result_df.loc[idx, 'color'] = prev_color
    
    for idx, row in result_df.iterrows():
        if not row.isnull().all():
            fw_color = row['color']
            for i in range(idx):
                result_df.at[i, 'color'] = fw_color
            break

    return result_df


def draw_ellipses(points, reference_point, subset_indices, ellipses_resolution = 30, is_debug = False):
    '''
    '''
    
    # Empty lists to store ellipses coordinates
    ellipses_x = []
    ellipses_y = []
    ellipses_z = []
    
    if is_debug:
        
        import matplotlib.pyplot as plt
        
        # Plotting for visualization
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Original points
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], label='Original Points')

        # Subset points
        ax.scatter(points[subset_indices, 0], points[subset_indices, 1], points[subset_indices, 2],
                   color='r', s=50, label='Subset Points')

        # Reference point
        ax.scatter(reference_point[0], reference_point[1], reference_point[2], marker='x', label='Reference Point')
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.legend()
    
    for idx in subset_indices:
        subset_point = points[idx]

        # Calculate vector from reference_point to subset_point
        vector_to_subset = subset_point - reference_point

        # Normalize the vector
        normalized_vector = vector_to_subset / np.linalg.norm(vector_to_subset)

        # Define the ellipse parameters
        a = np.linalg.norm(vector_to_subset)
        b = a / 10 # Minor axis length

        # Parametric equation of an ellipse
        theta = np.linspace(0, 2 * np.pi, num = ellipses_resolution)
        x = subset_point[0] + a * np.cos(theta) * normalized_vector[0] - b * np.sin(theta) * normalized_vector[1] + vector_to_subset[0]
        y = subset_point[1] + a * np.cos(theta) * normalized_vector[1] + b * np.sin(theta) * normalized_vector[0] + vector_to_subset[1]
        z = subset_point[2] + a * np.cos(theta) * normalized_vector[2]                                            + vector_to_subset[2]
        
        # Plot the ellipse
        if is_debug: ax.plot(x, y, z, color='b')
        
        ellipses_x += list(x) + [None]
        ellipses_y += list(y) + [None]
        ellipses_z += list(z) + [None]
        
    if is_debug:
        plt.show()
    
    return ellipses_x, ellipses_y, ellipses_z

# # Example usage
# np.random.seed(42)
# points = np.random.rand(100, 3)
# reference_point = np.array([0.4, 3, 0.5])
# subset_indices = [0, 5, 10, 15, 20]

# # Draw ellipses
# ellipses_x, ellipses_y, ellipses_z = draw_ellipses(points, reference_point, subset_indices, ellipses_resolution = 30, is_debug = False)

###############################################################################
###############################################################################
###############################################################################

# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ #

###############################################################################
################################ Class Protein ################################
###############################################################################

# Class name definition
class Protein(object):
    
    # Nº to tag proteins and a list with the IDs added
    protein_tag = 0
    protein_list = []
    protein_list_IDs = []
    
    # Color palette for network representation
    default_color_palette = {
      "Red":            ["#ffebee", "#ffcdd2", "#ef9a9a", "#e57373", "#ef5350", "#f44336", "#e53935", "#d32f2f", "#c62828", "#ff8a80", "#ff5252", "#d50000", "#f44336", "#ff1744", "#b71c1c"],
      "Green":          ["#e8f5e9", "#c8e6c9", "#a5d6a7", "#81c784", "#66bb6a", "#4caf50", "#43a047", "#388e3c", "#2e7d32", "#b9f6ca", "#69f0ae", "#00e676", "#4caf50", "#00c853", "#1b5e20"],
      "Blue":           ["#e3f2fd", "#bbdefb", "#90caf9", "#64b5f6", "#42a5f5", "#2196f3", "#1e88e5", "#1976d2", "#1565c0", "#82b1ff", "#448aff", "#2979ff", "#2962ff", "#2196f3", "#0d47a1"],
      "Yellow":         ["#fffde7", "#fff9c4", "#fff59d", "#fff176", "#ffee58", "#ffeb3b", "#fdd835", "#fbc02d", "#f9a825", "#ffff8d", "#ffff00", "#ffea00", "#ffd600", "#ffeb3b", "#f57f17"],
      "Lime":           ["#f9fbe7", "#f0f4c3", "#e6ee9c", "#dce775", "#d4e157", "#cddc39", "#c0ca33", "#afb42b", "#9e9d24", "#f4ff81", "#eeff41", "#c6ff00", "#aeea00", "#cddc39", "#827717"],
      "Orange":         ["#fff3e0", "#ffe0b2", "#ffcc80", "#ffb74d", "#ffa726", "#ff9800", "#fb8c00", "#f57c00", "#ef6c00", "#ffd180", "#ffab40", "#ff9100", "#ff6d00", "#ff9800", "#e65100"],
      "Purple":         ["#f3e5f5", "#e1bee7", "#ce93d8", "#ba68c8", "#ab47bc", "#9c27b0", "#8e24aa", "#7b1fa2", "#6a1b9a", "#ea80fc", "#e040fb", "#d500f9", "#aa00ff", "#9c27b0", "#4a148c"],
      "Light_Blue":     ["#e1f5fe", "#b3e5fc", "#81d4fa", "#4fc3f7", "#29b6f6", "#03a9f4", "#039be5", "#0288d1", "#0277bd", "#80d8ff", "#40c4ff", "#00b0ff", "#0091ea", "#03a9f4", "#01579b"],
      "Teal":           ["#e0f2f1", "#b2dfdb", "#80cbc4", "#4db6ac", "#26a69a", "#009688", "#00897b", "#00796b", "#00695c", "#a7ffeb", "#64ffda", "#1de9b6", "#00bfa5", "#009688", "#004d40"],
      "Light_Green":    ["#f1f8e9", "#dcedc8", "#c5e1a5", "#aed581", "#9ccc65", "#8bc34a", "#7cb342", "#689f38", "#558b2f", "#ccff90", "#b2ff59", "#76ff03", "#64dd17", "#8bc34a", "#33691e"],
      "Amber":          ["#fff8e1", "#ffecb3", "#ffe082", "#ffd54f", "#ffca28", "#ffc107", "#ffb300", "#ffa000", "#ff8f00", "#ffe57f", "#ffd740", "#ffc400", "#ffab00", "#ffc107", "#ff6f00"],
      "Deep_Orange":    ["#fbe9e7", "#ffccbc", "#ffab91", "#ff8a65", "#ff7043", "#ff5722", "#f4511e", "#e64a19", "#d84315", "#ff9e80", "#ff6e40", "#ff3d00", "#dd2c00", "#ff5722", "#bf360c"],
      "Pink":           ["#fce4ec", "#f8bbd0", "#f48fb1", "#f06292", "#ec407a", "#e91e63", "#d81b60", "#c2185b", "#ad1457", "#ff80ab", "#ff4081", "#f50057", "#c51162", "#e91e63", "#880e4f"],
      "Deep_Purple":    ["#ede7f6", "#d1c4e9", "#b39ddb", "#9575cd", "#7e57c2", "#673ab7", "#5e35b1", "#512da8", "#4527a0", "#b388ff", "#7c4dff", "#651fff", "#6200ea", "#673ab7", "#311b92"],
      "Cyan":           ["#e0f7fa", "#b2ebf2", "#80deea", "#4dd0e1", "#26c6da", "#00bcd4", "#00acc1", "#0097a7", "#00838f", "#84ffff", "#18ffff", "#00e5ff", "#00b8d4", "#00bcd4", "#006064"],
      "Indigo":         ["#e8eaf6", "#c5cae9", "#9fa8da", "#7986cb", "#5c6bc0", "#3f51b5", "#3949ab", "#303f9f", "#283593", "#8c9eff", "#536dfe", "#3d5afe", "#304ffe", "#3f51b5", "#1a237e"],
    }
    
    # Create an object of Protein class
    def __init__(self, ID, PDB_chain, sliced_PAE_and_pLDDTs, symbol = None, name = None):
        
        print(f"Creating object of class Protein: {ID}")
        
        # Check if PDB_chain is a biopython class of type Bio.PDB.Chain.Chain
        # Just in case something goes wrong with class checking, replace here with:
        # if str(type(PDB_chain)) !=  "<class 'Bio.PDB.Chain.Chain'>":
        if not isinstance(PDB_chain, Chain.Chain):            
            PDB_chain_type = type(PDB_chain)
            raise ValueError(f"PDB_chain is not of type Bio.PDB.Chain.Chain. Instead is of type {PDB_chain_type}.")
        
        # Check if the protein was already created
        if ID in Protein.protein_list_IDs:
            raise ValueError(f"Protein {ID} was already created. Protein multicopy not implemented yet.")
        
        # Get seq, res_names, xyz coordinates, res_pLDDT and CM
        seq = "".join([seq1(res.get_resname()) for res in PDB_chain.get_residues()])
        res_xyz = [res.center_of_mass() for res in PDB_chain.get_residues()]
        res_names = [AA + str(i + 1) for i, AA in enumerate(seq)]
        res_pLDDT = [res["CA"].get_bfactor() for res in PDB_chain.get_residues()]
        CM = center_of_mass(res_xyz)
        
        # Translate the protein centroids and PDB_chain to the origin (0,0,0)
        res_xyz = res_xyz - CM
        for atom in PDB_chain.get_atoms(): atom.transform(np.identity(3), np.array(-CM))
        
        # Extract domains from sliced_PAE_and_pLDDTs dict
        domains = sliced_PAE_and_pLDDTs[ID]['no_loops_domain_clusters'][1]
        
        self.ID         = ID                    # str (Required)
        self.seq        = seq                   # str (Required)
        self.symbol     = symbol                # str (Optional)
        self.name       = name                  # str (Optional)
        self.PDB_chain  = PDB_chain             # Bio.PDB.Chain.Chain
        self.domains    = domains               # list
        self.res_xyz    = res_xyz               # List np.arrays with centroid xyz coordinates (Angst) of each residue (Required)   <-------------
        self.res_names  = res_names             # E.g: ["M1", "S2", "P3", ..., "R623"]  (Optional)
        self.res_pLDDT  = res_pLDDT             # Per residue pLDDT (Required)
        self.CM         = np.array([0,0,0])     # Center of Mass (by default, proteins are translated to the origin when created)   <-------------
        
        # Initialize lists for protein partners information (indexes match)
        self.partners                   = []     # Protein instance for each partner    (list of Proteins)
        self.partners_IDs               = []     # ID of each partner                       (list of str )
        self.partners_ipTMs             = []     # ipTMs value for each partner index       (list of ints)
        self.partners_min_PAEs          = []     # min_PAE value for each partner index     (list of ints)
        self.partners_N_models          = []     # Nº of models that surpasses the cutoffs  (list of ints)
        
        # Contacts form 2mers dataset
        self.contacts_2mers_self_res          = []     # Self contact residues                (list of lists)
        self.contacts_2mers_partner_res       = []     # Partner contact residues             (list of lists)
        self.contacts_2mers_distances         = []     # Partner contact distances            (list of lists)
        self.contacts_2mers_PAE_per_res_pair  = []     # PAE for each contact residue pair    (list of lists)
        
        # CM contact surface residues (self)
        self.contacts_2mers_self_res_CM = []               # Self contact residues CM   (list of arrays with xyz coordinates)      <-------------
        
        # # Direction vectors of contacts
        # self.contacts_2mers_V_self_to_partner     = []     # Vector pointing from self.CM to contacts_2mers_self_CM
              
        # Assign a tag to each protein and add it to the list together with its ID
        self.protein_tag = Protein.protein_tag 
        Protein.protein_list.append(self)
        Protein.protein_list_IDs.append(self.ID)
        Protein.protein_tag += 1
    
    # -------------------------------------------------------------------------------------
    # -------------------------------------------------------------------------------------
    # -------------------------------------- Getters --------------------------------------
    # -------------------------------------------------------------------------------------
    # -------------------------------------------------------------------------------------

    def get_ID(self):                   return self.ID
    def get_seq(self):                  return self.seq
    def get_symbol(self):               return self.symbol
    def get_name(self):                 return self.name
    def get_CM(self):                   return self.CM
    def get_protein_tag(self):          return self.protein_tag
    
    def get_res_pLDDT(self, res_list = None):        
        '''Returns per residue pLDDT values as a list. If a res_list of residue
        indexes (zero index based) is passed, you will get only these pLDDT values.'''
        if res_list != None: 
            return[self.res_pLDDT[res] for res in res_list]
        return self.res_pLDDT
    
    def get_res_xyz(self, res_list = None):
        '''Pass a list of residue indexes as list (zero index based) to get their coordinates.'''
        if res_list != None: 
            return[self.res_xyz[res] for res in res_list]
        return self.res_xyz
    
    def get_res_names(self, res_list = None):
        '''Pass a list of residue indexes as list (zero index based) to get their residues names.'''
        if res_list != None:
            return[self.res_names[res] for res in res_list]
        return self.res_names
    
    def get_partners(self, use_IDs = False):
        '''If you want to get the partners IDs instead of Partners object, set use_IDs to True'''
        if use_IDs: return[partner.get_ID() for partner in self.partners]
        return self.partners
    
    def get_partners_ipTMs(self, partner = None, use_IDs = False):
        # Extract ipTM for pair if partner ID was provided
        if partner != None:
            # Find its index and return the residues of the contacts over self
            matching_partner_index = self.get_partners(use_IDs = use_IDs).index(partner)
            return self.partners_ipTMs[matching_partner_index]
        return self.partners_ipTMs
    
    def get_partners_min_PAEs(self, partner = None, use_IDs = False):
        # Extract min_PAE for pair if partner ID was provided
        if partner != None:
            # Find its index and return the residues of the contacts over self
            matching_partner_index = self.get_partners(use_IDs = use_IDs).index(partner)
            return self.partners_min_PAEs[matching_partner_index]
        return self.partners_min_PAEs
    
    def get_partners_N_models(self, partner = None, use_IDs = False):
        # Extract N_models for pair if partner ID was provided
        if partner != None:
            # Find its index and return the residues of the contacts over self
            matching_partner_index = self.get_partners(use_IDs = use_IDs).index(partner)
            return self.partners_N_models[matching_partner_index]
        return self.partners_N_models
    
    def get_partner_index(self, partner, use_ID = False):
        '''Returns the index of the partner in self.partners list'''
        return self.partners.index(partner.get_ID() if use_ID else partner)
    
    def get_partner_contact_surface_CM(self, partner, use_ID = False):
        '''Returns the CM of the partner surface that interact with the protein.'''
        partner_index = self.get_partner_index(partner = partner, use_ID = use_ID)
        return center_of_mass([partner.get_res_xyz()[res] for res in list(set(self.contacts_2mers_partner_res[partner_index]))])

    
    # Get contact residues numbers (for self)
    def get_contacts_2mers_self_res(self, partner = None, use_IDs = False):
        '''Returns the residues indexes of the protein surface that 
        interacts with the partner.'''
        # Extract contacts if partner ID was provided
        if partner != None:
            # Find its index and return the residues of the contacts over self
            matching_partner_index = self.get_partners(use_IDs = use_IDs).index(partner)
            return self.contacts_2mers_self_res[matching_partner_index]
        # return the entire list
        return self.contacts_2mers_self_res
    
    # Get contact residues numbers (for partner)
    def get_contacts_2mers_partner_res(self, partner = None, use_IDs = False):
        '''Returns the xzy positions of the partner surface residues that 
        interacts with the protein. If you are using the ID of the partner to
        select, set use_IDs to True'''
        # Extract contacts if partner was provided
        if partner != None:
            # Find its index and return the residues of the contacts over self
            matching_partner_index = self.get_partners(use_IDs = use_IDs).index(partner)
            return self.contacts_2mers_partner_res[matching_partner_index]
        # return the entire list
        return self.contacts_2mers_partner_res
    
    def get_contacts_2mers_self_res_CM(self, partner = None, use_IDs = False):
        # Extract CM if partner was provided
        if partner != None:
            # Find its index and return the residues of the contacts over self
            matching_partner_index = self.get_partners().index(partner)
            return self.contacts_2mers_self_res_CM[matching_partner_index]
        # return the entire list
        return self.contacts_2mers_self_res_CM
    
    # def get_contacts_2mers_V_self_to_partner(self, partner = None, use_IDs = True):
    #     '''If you are using the ID of the protein to select, set use_IDs to True'''
                
    #     # Extract contacts if partner ID was provided
    #     if partner != None:
    #         # Find its index and return the residues of the contacts over self
    #         matching_partner_index = self.get_partners(use_IDs = use_IDs).index(partner)
    #         self.contacts_2mers_V_self_to_partner[matching_partner_index]
    #     return self.contacts_2mers_V_self_to_partner
    
    # -------------------------------------------------------------------------------------
    # -------------------------------------------------------------------------------------
    # -------------------------------------- Setters --------------------------------------
    # -------------------------------------------------------------------------------------
    # -------------------------------------------------------------------------------------

    def set_ID(self, ID):               self.ID = ID
    def set_seq(self, seq):             self.seq = seq
    def set_symbol(self, symbol):       self.symbol = symbol
    def set_name(self, name):           self.name = name
    def set_res_xyz(self, res_xyz):     self.res_xyz = res_xyz
    def set_res_names(self, res_names): self.res_names = res_names
    def set_CM(self, CM):               self.CM = CM
    
    # # Add multiple partners using contacts_2mers_df
    # def add_partners_manually(self, contacts_2mers_df):

    # Add multiple partners using contacts_2mers_df
    def add_partners_from_contacts_2mers_df(self, contacts_2mers_df, recursive_call = False):
        '''
        To use it, first generate all the instances of proteins involved in the
        interactions.
        '''
        print(f"INITIALIZING: Adding partners for {self.ID} from contacts dataframe.")
        
        # Progress
        if recursive_call: None
        else: 
            print("")
            print(f"Searching partners for {self.ID}...")
        
        # Iterate over the contacts_2mers_df one pair at a time
        for pair, pair_contacts_2mers_df in contacts_2mers_df.groupby(['protein_ID_a', 'protein_ID_b'], group_keys=False):
            # Reset the index for each group
            pair_contacts_2mers_df = pair_contacts_2mers_df.reset_index(drop=True)         
                        
            print("   Analyzing pair:", pair, end= "")
            
            if self.ID in pair:
                print(f" - RESULT: Found partner for {self.ID}.")
                print( "      - Analyzing contacts information...")
                
                # Get index of each protein for the pair
                self_index_for_pair = pair.index(self.ID)                
                partner_index_for_pair = pair.index(self.ID) ^ 1 # XOR operator to switch index from 0 to 1 and vice versa
                partner_ID = pair[partner_index_for_pair]
                
                # Get the name of the protein
                pair[partner_index_for_pair]
                
                # Check if partner was already added
                if pair[partner_index_for_pair] in self.partners_IDs:
                     print(f"      - Partner {pair[partner_index_for_pair]} is already a partner of {self.ID}")
                     print( "      - Jumping to next pair...")
                     continue
                
                # Extract info from the contacts_df
                try: partner_protein = Protein.protein_list[Protein.protein_list_IDs.index(partner_ID)]
                except: raise ValueError(f"The protein {pair[partner_index_for_pair]} is not added as Protein instance of class Protein.")
                partners_IDs = partner_protein.get_ID()
                partner_ipTM = float(pair_contacts_2mers_df["ipTM"][0])
                partner_min_PAE = float(pair_contacts_2mers_df["min_PAE"][0])
                partner_N_models = int(pair_contacts_2mers_df["N_models"][0])
                partners_PAE_per_res_pair = list(pair_contacts_2mers_df["PAE"])
                self_res_contacts = list(pair_contacts_2mers_df["res_a"]) if self_index_for_pair == 0 else list(pair_contacts_2mers_df["res_b"])
                other_res_contacts = list(pair_contacts_2mers_df["res_a"]) if partner_index_for_pair == 0 else list(pair_contacts_2mers_df["res_b"])
                contact_distances = list(pair_contacts_2mers_df["distance"])
                
                # Check correct inputs
                if not isinstance(partner_protein, Protein): raise ValueError(f"Variable partner_protein with value {partner_protein} for {self.ID} is not of type Protein. Instead is of type {type(partner_protein)}")
                if not isinstance(partner_ipTM,      float): raise ValueError(f"Variable partner_ipTM with value {partner_ipTM} for {self.ID} is not of type float. Instead is of type {type(partner_ipTM)}")
                if not isinstance(partner_min_PAE,   float): raise ValueError(f"Variable partner_min_PAE with value {partner_min_PAE} for {self.ID} is not of type float. Instead is of type {type(partner_min_PAE)}")
                if not isinstance(partner_N_models,  int  ): raise ValueError(f"Variable partner_N_models with value {partner_N_models} for {self.ID} is not of type int. Instead is of type {type(partner_N_models)}")
                if not isinstance(self_res_contacts, list ): raise ValueError(f"Variable self_res_contacts with value {self_res_contacts} for {self.ID} is not of type list. Instead is of type {type(self_res_contacts)}")
                if not isinstance(other_res_contacts,list ): raise ValueError(f"Variable other_res_contacts with value {other_res_contacts} for {self.ID} is not of type list. Instead is of type {type(other_res_contacts)}")
                
                # Append their values
                self.partners.append(               partner_protein)
                self.partners_IDs.append(           partners_IDs)
                self.partners_ipTMs.append(         partner_ipTM)
                self.partners_min_PAEs.append(      partner_min_PAE)
                self.partners_N_models.append(      partner_N_models)

                # Contacts form 2mers dataset
                self.contacts_2mers_self_res.append(        self_res_contacts)
                self.contacts_2mers_partner_res.append(     other_res_contacts)
                self.contacts_2mers_distances.append(       contact_distances)
                self.contacts_2mers_PAE_per_res_pair.append(partners_PAE_per_res_pair)
                
                # Compute CM of the surface and append it
                contacts_2mers_self_CM = center_of_mass([self.res_xyz[res] for res in self.contacts_2mers_self_res[-1]])
                self.contacts_2mers_self_res_CM.append(contacts_2mers_self_CM)
                
                print(f"      - Partner information for {pair[partner_index_for_pair]} was added to {self.ID}")
                
                # Do the same with the partner protein
                # sub_pair_contacts_2mers_df = pair_contacts_2mers_df.query("")   # Get only the pairwise info for the pair
                partner_protein.add_partners_from_contacts_2mers_df(contacts_2mers_df = pair_contacts_2mers_df, recursive_call = True)
                
            else:
                print(f" - RESULT: No partners for {self.ID}.")
    
    # def compute_shared_contacts(self):
    #     '''Finds contact residues that are shared with more than one partner
    #     (Co-Occupancy).'''
    #     raise AttributeError("Not implemented yet.")
        
    def get_network_shared_residues(self):
        '''Finds contact residues that are shared with more than one partner
        (Co-Occupancy) and returns it as list of tuples (protein ID, res_name, xyz-position).
        As second return, it gives the info as a dataframe.'''
        
        # Get all the proteins in the network
        all_proteins = self.get_partners_of_partners() + [self]
        
        # To keep track residue pairs at the network level
        contact_pairs_df = pd.DataFrame(columns = [
            "protein_ID_1", "res_name_1", "xyz_1"
            "protein_ID_2", "res_name_2", "xyz_2"])

        # To keep track of already added contacts
        already_computed_pairs = []
                
        # Retrieve contact pairs protein by protein
        for protein in all_proteins:
            
            for P, partner in enumerate(protein.get_partners()):
                
                # Check both directions
                pair_12 = (protein.get_ID(), partner.get_ID())
                pair_21 = (partner.get_ID(), protein.get_ID())
                
                # Go to next partner if pair was already analyzed
                if pair_12 not in already_computed_pairs:
                    already_computed_pairs.extend([pair_12, pair_21])
                else: continue
                
                # Add one contact at a time
                for contact_res_self, contact_res_part in zip(protein.get_contacts_2mers_self_res(partner),
                                                              protein.get_contacts_2mers_partner_res(partner)):
                
                    # Add contact pair to dict
                    contacts12 = pd.DataFrame({
                        # Save them as 0 base
                        "protein_ID_1": [protein.get_ID()],
                        "protein_ID_2": [partner.get_ID()],
                        "res_name_1": [protein.res_names[contact_res_self]],
                        "res_name_2": [partner.res_names[contact_res_part]],
                        "xyz_1": [protein.res_xyz[contact_res_self]],
                        "xyz_2": [partner.res_xyz[contact_res_part]],
                        })
                    
                    # Add contact pair to dict in both directions
                    contacts21 = pd.DataFrame({
                        # Save them as 0 base
                        "protein_ID_1": [partner.get_ID()],
                        "protein_ID_2": [protein.get_ID()],
                        "res_name_1": [partner.res_names[contact_res_part]],
                        "res_name_2": [protein.res_names[contact_res_self]],
                        "xyz_1": [partner.res_xyz[contact_res_part]],
                        "xyz_2": [protein.res_xyz[contact_res_self]],
                        })
                    
                    contact_pairs_df = pd.concat([contact_pairs_df, contacts12, contacts21], ignore_index = True)
        
        # To store network shared residues
        shared_residues    =  []
        shared_residues_df = pd.DataFrame(columns = [
            "protein_ID_1", "res_name_1", "xyz_1"
            "protein_ID_2", "res_name_2", "xyz_2"])
        
        # Explore one residue at a time if it has multiple partners (co-occupancy)
        for residue, residue_df in contact_pairs_df.groupby(['protein_ID_1', 'res_name_1'], group_keys=False):            
            # Reset the index for each group
            residue_df = residue_df.reset_index(drop=True)

            # Get all the proteins that bind the residue
            proteins_that_bind_residue = list(set(list(residue_df["protein_ID_2"])))
            
            # If more than one protein binds the residue
            if len(proteins_that_bind_residue) > 1:
                
                # Add residue data to list and df
                shared_residues.append(residue)
                shared_residues_df = pd.concat([shared_residues_df, residue_df], ignore_index = True)
        
        return shared_residues, shared_residues_df
        

    # Updaters ----------------------------------------------------------------
    
    
    def update_CM(self):
        
        x_list = []
        y_list = []
        z_list = []
        
        for point in self.res_xyz:
            
            x_list.append(point[0])
            y_list.append(point[1])
            z_list.append(point[2])
        
        self.CM = np.array([np.mean(x_list), np.mean(y_list), np.mean(z_list)])
        
    def update_res_names(self):
        self.res_names = [AA + str(i + 1) for i, AA in enumerate(self.seq)]
        
    def update_contacts_res_CM(self):
        
        for P, partner in enumerate(self.get_partners()):
            self.contacts_2mers_self_res_CM[P] = center_of_mass([self.res_xyz[res] for res in list(set(self.contacts_2mers_self_res[P]))])
        

    
    # Rotation, translation and precession of Proteins ------------------------
    

    def rotate(self, partner):
        '''Rotates a protein to align its surface vector to the CM of the
        interaction surface of a partner'''
        
        print(f"   - Rotating {self.ID} with respect to {partner.get_ID()}...")
        
        
        def rotate_points(points, reference_point, subset_indices, target_point):
            # Extract subset of points
            subset_points = np.array([points[i] for i in subset_indices])

            # Calculate center of mass of the subset
            subset_center_of_mass = np.mean(subset_points, axis=0)

            # Calculate the vector from the reference point to the subset center of mass
            vector_to_subset_com = subset_center_of_mass - reference_point

            # Calculate the target vector
            target_vector = target_point - reference_point

            # Calculate the rotation axis using cross product
            rotation_axis = np.cross(vector_to_subset_com, target_vector)
            rotation_axis /= np.linalg.norm(rotation_axis)

            # Calculate the angle of rotation
            angle = np.arccos(np.dot(vector_to_subset_com, target_vector) /
                             (np.linalg.norm(vector_to_subset_com) * np.linalg.norm(target_vector)))

            # Perform rotation using Rodrigues' rotation formula
            rotation_matrix = np.array([[np.cos(angle) + rotation_axis[0]**2 * (1 - np.cos(angle)),
                                         rotation_axis[0] * rotation_axis[1] * (1 - np.cos(angle)) - rotation_axis[2] * np.sin(angle),
                                         rotation_axis[0] * rotation_axis[2] * (1 - np.cos(angle)) + rotation_axis[1] * np.sin(angle)],
                                        [rotation_axis[1] * rotation_axis[0] * (1 - np.cos(angle)) + rotation_axis[2] * np.sin(angle),
                                         np.cos(angle) + rotation_axis[1]**2 * (1 - np.cos(angle)),
                                         rotation_axis[1] * rotation_axis[2] * (1 - np.cos(angle)) - rotation_axis[0] * np.sin(angle)],
                                        [rotation_axis[2] * rotation_axis[0] * (1 - np.cos(angle)) - rotation_axis[1] * np.sin(angle),
                                         rotation_axis[2] * rotation_axis[1] * (1 - np.cos(angle)) + rotation_axis[0] * np.sin(angle),
                                         np.cos(angle) + rotation_axis[2]**2 * (1 - np.cos(angle))]])

            # Apply rotation to all points
            rotated_points = np.dot(points - reference_point, rotation_matrix.T) + reference_point

            return rotated_points, rotation_matrix, rotation_axis
        

    
        rotated_points, rotation_matrix, rotation_axis =\
            rotate_points(points = self.get_res_xyz(),
            reference_point = self.get_CM(),
            subset_indices = self.get_contacts_2mers_self_res(partner),
            target_point = self.get_partner_contact_surface_CM(partner=partner, use_ID=False))
    
        self.res_xyz = rotated_points
        self.rotate_PDB_atoms(self.get_CM(), rotation_matrix)
        self.update_CM()
        self.update_contacts_res_CM()
    
    def rotate_PDB_atoms(self, reference_point, rotation_matrix):

        # Apply rotation to all atoms of PDB chain
        PDB_atoms = [atom.get_coord() for atom in self.PDB_chain.get_atoms()]
        rotated_PDB_atoms = np.dot(PDB_atoms - reference_point, rotation_matrix.T) + reference_point
                        
        for A, atom in enumerate(self.PDB_chain.get_atoms()):
            atom.set_coord(rotated_PDB_atoms[A])
    
        
    def rotate2all(self, use_CM = True):
        '''Rotates a protein to align its surfaces with the mean CMs of all of its partners'''
        
        print(f"   - Rotating {self.ID} with respect to {[partner.get_ID() for partner in self.get_partners()]} CMs...")
        
        def rotate_points(points, reference_point, subset_indices, target_point):
            # Extract subset of points
            subset_points = np.array([points[i] for i in subset_indices])

            # Calculate center of mass of the subset
            subset_center_of_mass = np.mean(subset_points, axis=0)

            # Calculate the vector from the reference point to the subset center of mass
            vector_to_subset_com = subset_center_of_mass - reference_point

            # Calculate the target vector
            target_vector = target_point - reference_point

            # Calculate the rotation axis using cross product
            rotation_axis = np.cross(vector_to_subset_com, target_vector)
            rotation_axis /= np.linalg.norm(rotation_axis)

            # Calculate the angle of rotation
            angle = np.arccos(np.dot(vector_to_subset_com, target_vector) /
                             (np.linalg.norm(vector_to_subset_com) * np.linalg.norm(target_vector)))

            # Perform rotation using Rodrigues' rotation formula
            rotation_matrix = np.array([[np.cos(angle) + rotation_axis[0]**2 * (1 - np.cos(angle)),
                                         rotation_axis[0] * rotation_axis[1] * (1 - np.cos(angle)) - rotation_axis[2] * np.sin(angle),
                                         rotation_axis[0] * rotation_axis[2] * (1 - np.cos(angle)) + rotation_axis[1] * np.sin(angle)],
                                        [rotation_axis[1] * rotation_axis[0] * (1 - np.cos(angle)) + rotation_axis[2] * np.sin(angle),
                                         np.cos(angle) + rotation_axis[1]**2 * (1 - np.cos(angle)),
                                         rotation_axis[1] * rotation_axis[2] * (1 - np.cos(angle)) - rotation_axis[0] * np.sin(angle)],
                                        [rotation_axis[2] * rotation_axis[0] * (1 - np.cos(angle)) - rotation_axis[1] * np.sin(angle),
                                         rotation_axis[2] * rotation_axis[1] * (1 - np.cos(angle)) + rotation_axis[0] * np.sin(angle),
                                         np.cos(angle) + rotation_axis[2]**2 * (1 - np.cos(angle))]])

            # Apply rotation to all points
            rotated_points = np.dot(points - reference_point, rotation_matrix.T) + reference_point

            return rotated_points, rotation_matrix, rotation_axis
        
            
        # Get protein surface residues and partners CMs
        subset_indices = []
        partners_CMs = []
        for P, partner in enumerate(self.get_partners()):
            partners_CMs.append(partner.get_CM())
            for residue_index in self.get_contacts_2mers_self_res(partner):
                if residue_index not in subset_indices:
                    subset_indices.append(residue_index)
                    
        # Compute partners CM centroid
        partners_centroid = center_of_mass(partners_CMs)
        
        rotated_points, rotation_matrix, rotation_axis =\
            rotate_points(points = self.get_res_xyz(),
                          reference_point = self.get_CM(),
                          subset_indices = subset_indices,
                          target_point = partners_centroid)
            
        self.res_xyz = rotated_points
        self.rotate_PDB_atoms(self.get_CM(), rotation_matrix)
        self.update_CM()
        self.update_contacts_res_CM()
        
    def translate_PDB_chain(self, translation_vector):
        for atom in self.PDB_chain.get_atoms():
            atom.transform(np.identity(3), np.array(translation_vector))
            
    
    def translate(self, part, distance = 40, bring_partners = False):
        '''Translates the protein away from a partner until the separation 
        between their contact surface residues (CM) is equal to "distance" 
        (angstroms) in the direction of the surface vector of the partner'''
               
        print(f"Initiating protein translation: {self.ID} --> {part.get_ID()}")
        
        # Check current distance between proteins surfaces 
        self_CM_surf = self.get_contacts_2mers_self_res_CM(partner = part)
        part_CM_surf = part.get_contacts_2mers_self_res_CM(partner = self)
        CM_surf_dist_i = calculate_distance(self_CM_surf, part_CM_surf)       # Absolute distance
        
        print("   - Initial inter-surface distance:", CM_surf_dist_i)
        
        # Reference point
        REFERENCE = part.get_CM()        
        
        # Final point position of the surface CM (relative to partner CM)
        Vba   = self.get_partner_contact_surface_CM(partner = part) - REFERENCE
        Vba_desired_length = vector_magnitude(Vba) + distance
        final_position_vector = find_vector_with_length(Vba, desired_length = Vba_desired_length)
        
        # print(f"DEBUG: DISTANCE BETWEEN partner surface CM and final position (must be {distance}):", calculate_distance(Vba, final_position_vector))
        
        # Current point position of the surface CM (relative to partner CM)
        self_CM_surf_i = self.get_contacts_2mers_self_res_CM(partner = part) - REFERENCE
        
        # Direction vector
        Vdir = final_position_vector - self_CM_surf_i
        
        # Real distance displacement
        real_distance = calculate_distance(final_position_vector, self_CM_surf_i)
        
        print(f"   - Translating {self.ID} {real_distance} angstroms")
        print(f"   - Direction: {Vdir}")
        
        # Translate the protein residues
        translated_residues = translate_points(points = self.res_xyz,
                                               v_direction = Vdir,
                                               distance = real_distance,
                                               is_CM = False)
        
        
        # # Translate the CM of the protein
        # translated_CM = translate_points(points = self.CM,
        #                                  v_direction = Vdir,
        #                                  distance = real_distance,
        #                                  is_CM = True)
                
            
        
        # Update self xyz coordinates and CM
        self.res_xyz = translated_residues
        self.update_CM()
        self.update_contacts_res_CM()
        
        # Check final distance between proteins surfaces 
        final_self_CM_surf = self.get_contacts_2mers_self_res_CM(partner = part)
        final_part_CM_surf = part.get_contacts_2mers_self_res_CM(partner = self)
        final_CM_surf_dist_i = calculate_distance(final_self_CM_surf, final_part_CM_surf)       # Absolute distance
        
        print("   - Final inter-surface distance:", final_CM_surf_dist_i, f"(expected: {distance})")
        
        # If you want bring together during translation the partners
        if bring_partners:
            for other_partner in self.partners:
                print(f"   - Bringing partner {other_partner.get_ID()} together")
                # Rotate every other partner, except the one passed as argument
                if other_partner == part:
                    continue
                else:
                    # Translate the protein residues
                    translated_residues_part = translate_points(points = other_partner.res_xyz,
                                                           v_direction = Vdir,
                                                           distance = real_distance,
                                                           is_CM = False)
                    
                    other_partner.set_res_xyz(translated_residues_part)
                    other_partner.update_CM()
                    other_partner.update_contacts_res_CM()
        
        
    def precess(self, partner, bring_partners = False):
        '''Precesses the protein around the axis defined between the center of
        mass of the contact residues of self and of partner. The precession is
        done until the distance between contact residues is minimized.'''
        
        print(f"Precessing {self.ID} with respect to {partner.get_ID()}...")
        
        # Get the residues coordinates to use as reference to minimize its distance while precessing
        self_contact_points_1    = self.get_contacts_2mers_self_res(partner = partner, use_IDs = False)
        partner_contact_points_2 = self.get_contacts_2mers_partner_res(partner = partner, use_IDs = False)
        
        # Precess the protein until get the minimal distance between points
        precessed_residues = precess_until_minimal_distance2(
            points = self.res_xyz,
            self_contact_points_1 = self.get_res_xyz(self_contact_points_1),
            partner_contact_points_2 = partner.get_res_xyz(partner_contact_points_2),
            reference_point= self.CM,
            angle_steps = 5)
        
        
        # If you want bring partners together with protein during precession
        if bring_partners:
            for other_partner in self.partners:
                print(f"   - Bringing partner {other_partner.get_ID()} together")
                # Rotate every other partner, except the one passed as argument
                if other_partner == partner:
                    continue
                else:
                    # Precess the protein until get the minimal distance between points
                    precessed_residues_part = precess_until_minimal_distance2(
                        points = other_partner.res_xyz,
                        self_contact_points_1 = self.get_res_xyz(self_contact_points_1),
                        partner_contact_points_2 = partner.get_res_xyz(partner_contact_points_2),
                        reference_point= self.CM,
                        angle_steps = 5)
                    
                    other_partner.set_res_xyz(precessed_residues_part)
                    other_partner.update_CM()
                    other_partner.update_contacts_res_CM()
                    
        # Update self xyz coordinates
        self.res_xyz = precessed_residues
        self.update_CM()
        self.update_contacts_res_CM()
    
    def align_surfaces(self, partner, distance = 60, bring_partners = False):
        '''Rotates, translates and precess a protein to align its contact surface 
        with a partner'''
        
        print(f"---------- Aligning surface of {self.ID} to {partner.get_ID()} ----------")
        
        # Perform alignment
        # self.rotate(partner = partner, bring_partners = bring_partners)
        self.translate(part = partner, distance = distance, bring_partners = bring_partners)
        self.rotate(partner = partner)
        self.translate(part = partner, distance = distance, bring_partners = bring_partners)
        self.rotate(partner = partner)
        self.precess(partner = partner, bring_partners = bring_partners)

        
    # 3D positions using igraph
    def get_fully_connected_pairwise_dataframe(self, sub_call = False):
        if not sub_call: print("")
        if not sub_call: print(f"INITIALIZING: Getting pairwise interaction dataframe of fully connected network for {self.ID}:")

        # Get the fully connected partners
        partners_list = self.get_partners_of_partners()
        all_proteins  = [self] + partners_list
        prot_num = len(all_proteins)
        
        print(f"   - Protein network contains {prot_num} proteins: {[prot.get_ID() for prot in all_proteins]}")
        
        # Construct pairwise_2mers_df -----------------------------------------------
        
        # Progress
        print("   - Extracting pairwise interaction data (ipTM, min_PAE and N_models).")
        
        # To store graph data
        pairwise_2mers_df = pd.DataFrame(columns =[
            "protein1", "protein2", "ipTM", "min_PAE", "N_models"])
        
        # To keep track of already added contacts for each pair
        already_computed_pairs = []
        
        for prot_N, protein in enumerate(all_proteins):
            for part_N, partner in enumerate(protein.get_partners()):
                
                # Check if protein pair was already analyzed in both directions
                pair_12 = (protein.get_ID(), partner.get_ID())
                pair_21 = (partner.get_ID(), protein.get_ID())
                if pair_12 not in already_computed_pairs:
                    already_computed_pairs.extend([pair_12, pair_21])
                else: continue
                                
                pair_df =  pd.DataFrame({
                    "protein1": [protein.get_ID()],
                    "protein2": [partner.get_ID()],
                    "ipTM": [protein.get_partners_ipTMs(partner)],
                    "min_PAE": [protein.get_partners_min_PAEs(partner)],
                    "N_models": [protein.get_partners_N_models(partner)]
                    })
                
                pairwise_2mers_df = pd.concat([pairwise_2mers_df, pair_df], ignore_index = True)
        
        if not sub_call: print("   - Resulting pairwise dataframe:")
        if not sub_call: print(pairwise_2mers_df)
        
        return pairwise_2mers_df
    
    def plot_fully_connected_protein_level_2D_graph(self, show_plot = True, return_graph = True, algorithm = "drl",
                                                    save_png = None, sub_call = False):
        
        if not sub_call: print("")
        if not sub_call: print(f"INITIALIZING: Generating 2D graph of fully connected network for {self.ID}:")
        
        # Get pairwise_2mers_df
        pairwise_2mers_df = self.get_fully_connected_pairwise_dataframe(sub_call = True)
                
        # Extract unique nodes from both 'protein1' and 'protein2'
        nodes = list(set(pairwise_2mers_df['protein1']) | set(pairwise_2mers_df['protein2']))
        
        # Create an undirected graph
        graph = igraph.Graph()
        
        # Add vertices (nodes) to the graph
        graph.add_vertices(nodes)
        
        # Add edges to the graph
        edges = list(zip(pairwise_2mers_df['protein1'], pairwise_2mers_df['protein2']))
        graph.add_edges(edges)
        
        # Set the edge weight modifiers
        N_models_W = pairwise_2mers_df.groupby(['protein1', 'protein2'])['N_models'].max().reset_index(name='weight')['weight']
        ipTM_W = pairwise_2mers_df.groupby(['protein1', 'protein2'])['ipTM'].max().reset_index(name='weight')['weight']
        min_PAE_W = pairwise_2mers_df.groupby(['protein1', 'protein2'])['min_PAE'].max().reset_index(name='weight')['weight']
        
        # Set the weights with custom weight function
        graph.es['weight'] = round(N_models_W * ipTM_W * (1/min_PAE_W) * 2, 2)
        
        # Add ipTM, min_PAE and N_models as attributes to the graph
        graph.es['ipTM'] = ipTM_W
        graph.es['min_PAE'] = min_PAE_W
        graph.es['N_models'] = N_models_W

        # Set layout
        layout = graph.layout(algorithm)
    
        # Print information for debugging
        print("   - Nodes:", nodes)
        print("   - Edges:", edges)
        print("   - Weights:", graph.es['weight'])
        
        # Plot the graph
        if show_plot:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots()
    
            igraph.plot(graph, 
                        layout = layout,
                        
                        # Nodes (vertex) characteristics
                        vertex_label = graph.vs["name"],
                        vertex_size = 40,
                        # vertex_color = 'lightblue',
                        
                        # Edges characteristics
                        edge_width = graph.es['weight'],
                        
                        # Plot size
                        bbox = (100, 100),
                        margin = 50,
                        
                        # To allow plot showing in interactive sessions                        
                        target = ax)
        
        # Plot the graph
        if save_png != None:
    
            igraph.plot(graph, 
                        layout = layout,
                        
                        # Nodes (vertex) characteristics
                        vertex_label = graph.vs["name"],
                        vertex_size = 40,
                        # vertex_color = 'lightblue',
                        
                        # Edges characteristics
                        edge_width = graph.es['weight'],
                        # edge_label = graph.es['ipTM'],
                        
                        # Plot size
                        bbox=(400, 400),
                        margin = 50,
                        
                        # PNG output file name path
                        target = save_png)
        
        if return_graph: return graph
        
    
    def set_fully_connected_network_3D_coordinates(self, show_plot = False,
                                                   algorithm = "drl", save_png = None,
                                                   scaling_factor = 100):
    
        # Progress
        print("")
        print(f"INITIALIZING: Setting 3D coordinates of fully connected network for {self.ID} using igraph:")
        print( "   - Translate all proteins to origin first.")
        
        
        # Make sure all proteins CMs are in the origin (0,0,0)
        partners_list = self.get_partners_of_partners()
        all_proteins  = [self] + partners_list
        
        if len(all_proteins) == 1:
            print("   - WARNING: Protein network contains only one protein.")
            print("   - WARNING: There is no relative protein to translate/rotate.")
            print("   - WARNING: Aborting.")
            return
        
        for protein in all_proteins:
            # Translate protein to origin
            old_xyz = protein.get_res_xyz()
            old_CM = protein.get_CM()
            translation_vector = old_CM
            new_xyz = old_xyz - translation_vector
            protein.set_res_xyz(new_xyz)
            protein.translate_PDB_chain(-translation_vector)
            protein.update_CM()
            protein.update_contacts_res_CM() 
            new_CM = protein.get_CM()
            print("      - Protein", protein.get_ID(), "translated from:", old_CM)
            print("      - Protein", protein.get_ID(), "translated to:  ", new_CM)
            
        
        # Create graph
        graph = self.plot_fully_connected_protein_level_2D_graph(show_plot = show_plot, algorithm = algorithm, save_png = save_png, sub_call = True)
        
        # Generate 3D coordinates for plot
        layt = [list(np.array(coord) * scaling_factor) for coord in list(graph.layout(algorithm, dim=3))]
        
        # Get graph nodes
        nodes = graph.vs["name"]        
        
        # Progress
        print("   - Translating proteins to new positions.")
                
        # Translate protein to new positions
        for i, protein_ID in enumerate(nodes):
            
            # Match node name with protein
            protein_match_index = Protein.protein_list_IDs.index(protein_ID)
            protein = Protein.protein_list[protein_match_index]
            
            # Translate protein to new positions
            old_xyz = protein.get_res_xyz()
            old_CM = protein.get_CM()
            translation_vector = layt[i]
            new_xyz = old_xyz + translation_vector
            protein.set_res_xyz(new_xyz)
            protein.translate_PDB_chain(translation_vector)
            protein.update_CM()
            protein.update_contacts_res_CM()
            
            print("      - Protein", protein_ID, "new CM:", translation_vector)
        
        # Progress
        print("   - Rotating proteins to match partners CMs...")
        for protein in all_proteins:
            protein.rotate2all()
            
        print("   - Finished.")
        
        
    
    def get_partners_of_partners(self ,deepness = 6, return_IDs = False, current_deepness = 0, partners_of_partners = None, query = None):
        '''Returns the set of partners of the partners with a depth of deepness.
    
        Parameters:
            deepness: Number of recursions to explore.
            return_IDs: If True, returns the IDs of the proteins. If false, the objects of Class Protein.
        '''
        # keep_track of the query protein
        if current_deepness == 0: query = self
    
        if partners_of_partners is None:
            partners_of_partners = []
    
        if current_deepness == deepness:
            return partners_of_partners
        else:
            for partner in self.partners:
                if (partner not in partners_of_partners) and (partner != query):
                    partners_of_partners.append(partner.get_ID() if return_IDs else partner)
                    partners_of_partners = list(set(partner.get_partners_of_partners(
                        return_IDs = return_IDs,
                        deepness=deepness,
                        current_deepness = current_deepness + 1,
                        partners_of_partners = partners_of_partners,
                        query = query
                    )))
                    
            return partners_of_partners
        
        
    # Protein removal ---------------------------------------------------------
        
    def cleanup_interactions(self):
        """Remove interactions with other Protein objects."""
        for partner in self.partners:
            try:
                index = partner.get_partner_index(self)
            except ValueError:
                continue  # Skip if not found in partners
            partner.partners.pop(index)
            partner.partners_IDs.pop(index)
            partner.partners_ipTMs.pop(index)
            partner.partners_min_PAEs.pop(index)
            partner.partners_N_models.pop(index)
            partner.contacts_2mers_self_res.pop(index)
            partner.contacts_2mers_partner_res.pop(index)
            partner.contacts_2mers_distances.pop(index)
            partner.contacts_2mers_PAE_per_res_pair.pop(index)
            partner.contacts_2mers_self_res_CM.pop(index)
    
    def remove_from_protein_list(self):
        """Remove the Protein instance from the list."""
        Protein.protein_list_IDs.pop(Protein.protein_list_IDs.index(self.ID))
        Protein.protein_list.pop(Protein.protein_list.index(self))
    
    def remove(self):
        """Protein removal with cleanup."""
        self.cleanup_interactions()
        self.remove_from_protein_list()
        print(f"Deleting Protein: {self.ID}")
        del(self)

       
    
    # Operators ---------------------------------------------------------------
    
    # Plus operator between proteins    
    def __add__(self, other):
        '''
        The sum of two or more proteins creates a network with those proteins
        inside it.
        '''
        # Use the method from Network
        return Network.__add__(self, other)
        
    def __lt__(self, other_protein):
        '''
        Returns true if self has less partners that the other protein. Useful
        for sorting hubs (The protein' CM with the highest number of partners
        can be set as the reference frame for plotting and network
        representations).
        '''
        
        # If they have the same number of partners
        if len(self.partners) == len(other_protein.get_partners()):
            # Brake the tie using sequence length
            return len(self.seq) < len(other_protein.get_seq())
        # If they are different, return which
        return len(self.partners) < len(other_protein.get_partners())
    
    
    # Plotting ----------------------------------------------------------------
    
    def plot_alone(self, custom_colors = None, res_size = 5, CM_size = 10,
                   res_color = "tan", res_opacity = 0.6, CM_color = "red",
                   contact_color = "red", contact_line = 5, contact_size = 7,
                   legend_position = dict(x=1.02, y=0.5),
                   show_plot = True, save_html = None, plddt_cutoff = 0,
                   shared_residue_color = "black"):
        '''Plots the vectorized protein in 3D space, coloring its contact surface residues with other partners
        Parameters:
            - custom_colors: list with custom colors for each interface contact residues
            - save_html: path to html file
            - plddt_cutoff: only show residues with pLDDT > plddt_cutoff (default 0).
                Useful for removing long disordered loops.
            - show_plot: if False, only returns the plot.
            - save_html: file path to save plot as HTML format (for easy results sharing).
        Return:
            - plot
        '''
        
        # colors = ["red", "green", "blue", "orange", "violet", "black", "brown",
        #           "bisque", "blanchedalmond", "blueviolet", "burlywood", "cadetblue"]
        
        # if custom_colors != None:
        #     colors = custom_colors
            
        # Create a 3D scatter plot
        fig = go.Figure()
        
        # Plot center of Mass
        fig.add_trace(go.Scatter3d(
            x=[self.CM[0]], y=[self.CM[1]], z=[self.CM[2]],
            mode='markers',
            marker=dict(
                size = CM_size,
                color = CM_color
            ),
            name = self.ID,
            showlegend = True,
            hovertext = self.ID
        ))
        
        # # Get the shared residues for the network (as tuple)
        shared_residues = self.get_network_shared_residues()[0]        
        
        # Plot one self contact residue at a time for each partner
        for partner_i, partner in enumerate(self.partners):
            for R_self, R_partner in zip(self.contacts_2mers_self_res[partner_i], self.contacts_2mers_partner_res[partner_i]):
                
                # Get protein+residue name as a tuple
                prot_plus_res_name_1 = (   self.get_ID(),    self.res_names[R_self]   )
                prot_plus_res_name_2 = (partner.get_ID(), partner.res_names[R_partner])                
                
                # Check if the residue is already in shared_residues
                if (prot_plus_res_name_1 in shared_residues) or (prot_plus_res_name_2 in shared_residues): shared_residue = True
                else: shared_residue = False
                
                fig.add_trace(go.Scatter3d(
                    x=[self.CM[0], self.res_xyz[R_self][0]],
                    y=[self.CM[1], self.res_xyz[R_self][1]],
                    z=[self.CM[2], self.res_xyz[R_self][2]],
                    mode='lines+markers',
                    marker=dict(
                        symbol='circle',
                        size = contact_size,                        
                        color = shared_residue_color if shared_residue else contact_color                        
                    ),
                    line=dict(
                        color = shared_residue_color if shared_residue else contact_color,
                        width = contact_line,
                        dash = 'solid' if partner.get_ID() != self.ID else "dot"
                    ),
                    name = self.res_names[R_self] + "-" + partner.get_res_names()[R_partner],
                    showlegend = False,
                    # self.res_names[R_self] + "-" + partner.get_res_names()[R_partner]
                    hovertext = self.ID + ":" + self.res_names[R_self] + " - " + partner.get_ID() + ":" + partner.get_res_names()[R_partner],
                ))
        
        # Plot one residue at a time
        for R, residue in enumerate(self.res_xyz):
            
            # only add residues that surpass pLDDT cutoff
            if self.res_pLDDT[R] > plddt_cutoff:
                fig.add_trace(go.Scatter3d(
                    x=[residue[0]],
                    y=[residue[1]],
                    z=[residue[2]],
                    mode='markers',
                    marker=dict(
                        size = res_size,
                        color = res_color,
                        opacity = res_opacity
                    ),
                    name = self.res_names[R],
                    showlegend = False,
                    hovertext = self.res_names[R]
                ))        
            
        # Set layout
        fig.update_layout(
            legend = legend_position,
            scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z'
        ))
        
        # Adjust layout margins
        fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))        
        
        # Display the plot?
        if show_plot == True: plot(fig)
        
        # Save the plot?
        if save_html != None: fig.write_html(save_html)
        
        return fig
    
    def plot_with_partners(self, specific_partners = None, custom_res_colors = None, custom_contact_colors = None, 
                           res_size = 5, CM_size = 10, contact_line = 2, contact_size = 5,
                           res_opacity = 0.3, show_plot = True, save_html = None,
                           legend_position = dict(x=1.02, y=0.5), plddt_cutoff = 0,
                           margin = dict(l=0, r=0, b=0, t=0), showgrid = False):
        '''
        
        specific_partners: list
            list of partners to graph with the protein. If None (default), protein
            will be plotted with all of its partners.
        showgrid: shows the 
        '''
        
        default_res_colors = ["blue", "orange", "yellow", "violet", "black", 
                      "brown", 'gray', 'chocolate', "green", 'aquamarine', 
                      'burlywood', 'cadetblue', 'chartreuse', 'chocolate',
                      'cornflowerblue', 'darkgoldenrod', 'darkkhaki',
                      'darkolivegreen', 'khaki', 'blueviolet', "red"]
        
        
        if custom_res_colors != None: res_colors = custom_res_colors
        else: res_colors = default_res_colors
        # if custom_contact_colors != None: contact_colors = custom_contact_colors
        
        fig = self.plot_alone(res_size = res_size, CM_size = CM_size, contact_line = contact_line, contact_size = contact_size,
                              res_color = res_colors[0], CM_color = res_colors[0], contact_color = res_colors[0],
                              res_opacity = res_opacity, show_plot = False)
        
        if specific_partners != None:
            partners_list = specific_partners
            # Check if all are partners
            if any(element not in self.partners for element in partners_list):
                raise ValueError(f"Some proteins in specific_partners list are not partners of {self.ID}")
        else:
            partners_list = self.partners
        
        
        for P, partner in enumerate(partners_list):
            fig2 = partner.plot_alone(res_size = res_size, CM_size = CM_size, contact_line = contact_line, contact_size = contact_size,
                                  res_color = res_colors[1:-1][P], CM_color = res_colors[1:-1][P], contact_color = res_colors[1:-1][P],
                                  res_opacity = res_opacity, show_plot = False)
            
            # Add fig2 traces to fig
            for trace in fig2.data:
                fig.add_trace(trace)
                
            # Add lines between contacts pairs --------------------------------
            
            # Lists of residues indexes for contacts
            contact_res_self = self.get_contacts_2mers_self_res(partner = partner, use_IDs = False)
            contact_res_partner = self.get_contacts_2mers_partner_res(partner = partner, use_IDs = False)
            
            # Coordinates
            contact_res_xyz_self = self.get_res_xyz(contact_res_self)
            contact_res_xyz_partner = partner.get_res_xyz(contact_res_partner)
            
            # Residues names
            contact_res_name_self = self.get_res_names(contact_res_self)
            contact_res_name_partner = partner.get_res_names(contact_res_partner)
            
            # Add one line at a time
            for contact_i in range(len(contact_res_self)):
                fig.add_trace(go.Scatter3d(
                    x = (contact_res_xyz_self[contact_i][0],) + (contact_res_xyz_partner[contact_i][0],) + (None,),  # Add None to create a gap between points_A and points_B
                    y = (contact_res_xyz_self[contact_i][1],) + (contact_res_xyz_partner[contact_i][1],) + (None,),
                    z = (contact_res_xyz_self[contact_i][2],) + (contact_res_xyz_partner[contact_i][2],) + (None,),
                    mode='lines',
                    line=dict(color='gray', width=1), # dash='dash'
                    showlegend = False,
                    name = contact_res_name_self[contact_i] + "-" + contact_res_name_partner[contact_i]
                    ))
        
        fig.update_layout(
            legend = legend_position
            )
        
        # Set layout
        fig.update_layout(
            legend = legend_position,
            scene=dict(
            xaxis=dict(showgrid=showgrid),
            yaxis=dict(showgrid=showgrid),
            zaxis=dict(showgrid=showgrid),
            xaxis_title='X (Å)',
            yaxis_title='Y (Å)',
            zaxis_title='Z (Å)'
        ))
        
        # Adjust layout margins
        fig.update_layout(margin = margin)
        
        
        # Display the plot?
        if show_plot == True: plot(fig)
        
        # Save the plot?
        if save_html != None: fig.write_html(save_html)
        
        return fig
    
    
    def plot_fully_connected(
            self, custom_res_colors = None, custom_contact_colors = None,
            res_size = 5, CM_size = 10, contact_line = 2, contact_size = 5, res_opacity = 0.3,
            show_plot = True, save_html = None, legend_position = dict(x=1.02, y=0.5),
            plddt_cutoff = 0, showgrid = True, margin = dict(l=0, r=0, b=0, t=0),
            show_axis = True, shared_resid_line_color = "black" , shared_residue_color = 'black'):
        '''
        
        Parameters:
            - specific_partners (list): partners to graph with the protein. If
                None (default), protein will be plotted with all of its partners.
            - custom_res_colors (list): list of colors to color each protein
            - res_size (float): size of non-contact residue centroids.
            - res_opacity (float): opacity of non-contact residue centroids.
            - plddt_cutoff (float): show only residues with pLDDT that surpass plddt_cutoff (0 to 100).
            - CM_size (float): size to represent the center of mass (in Å)
            - contact_line (float): size of the line between the CM and the contact residue (in Å)
            - contact_size (float): size of residues centroids that are in contact with other
                proteins (in Å).
            - show_plot (bool): displays the plot as html in the browser.
            - save_html (str): path to output HTML file.
            - showgrid (bool): show the background grid?
            - show_axis (bool): show the background box with axis?
            - margin (dict): margin sizes.
            - legend_position (dict): protein names legend position.
        '''
        
        print("")
        print(f"INITIALIZING: Plotting fully connected network for {self.ID}:")
        
        default_res_colors = ["red", "blue", "orange", "yellow", "violet", "black", 
                      "brown", 'gray', 'chocolate', "green", 'aquamarine', 
                      'burlywood', 'cadetblue', 'chartreuse', 'chocolate',
                      'cornflowerblue', 'darkgoldenrod', 'darkkhaki',
                      'darkolivegreen', 'khaki', 'blueviolet', "red"]
        
        
        if custom_res_colors != None: res_colors = custom_res_colors
        else: res_colors = default_res_colors
        # if custom_contact_colors != None: contact_colors = custom_contact_colors
        
                
        # Get the fully connected partners
        partners_list = self.get_partners_of_partners()
        all_proteins  = [self] + partners_list
        prot_num = len(all_proteins)
        
        print(f"   - Protein network contains {prot_num} proteins: {[prot.get_ID() for prot in all_proteins]}")
                
        # Initialize figure
        fig = go.Figure()
        
                
        # To keep track of already added contacts for each pair
        already_computed_pairs = []
        
        # # Get the shared residues for the network (as tuple)
        shared_residues = self.get_network_shared_residues()[0]
        
        # Work protein by protein
        for P, protein in enumerate(all_proteins):
            
            print(f"   - Adding {protein.get_ID()} coordinates.")
            
            # Add protein names
            
                                    
            # Plot coordinate of single protein
            fig2 = protein.plot_alone(
                res_size = res_size, CM_size = CM_size, contact_line = contact_line, contact_size = contact_size,
                res_color = res_colors[P], CM_color = res_colors[P], contact_color = res_colors[P],
                res_opacity = res_opacity, show_plot = False, plddt_cutoff = plddt_cutoff,  shared_residue_color = shared_residue_color)
        
            # Add individual proteins
            for trace in fig2.data:
                fig.add_trace(trace)
                
             # Add protein names
            fig.add_trace(go.Scatter3d(
                x=[protein.get_CM()[0]],
                y=[protein.get_CM()[1]],
                z=[protein.get_CM()[2]],
                text=[protein.get_ID()],
                mode='text',
                textposition='top center',
                textfont=dict(size = 20, color = "black"),# res_colors[P]),  # Adjust size and color as needed
                showlegend=False
            ))
            
            # Add lines between contacts pairs --------------------------------
            
            # Current protein partners
            partners_P = protein.get_partners()
            
            # Work partner by partner
            for partner_P in partners_P:
                
                # Check both directions
                pair_12 = (protein.get_ID(), partner_P.get_ID())
                pair_21 = (partner_P.get_ID(), protein.get_ID())
                
                if pair_12 not in already_computed_pairs:
                    already_computed_pairs.extend([pair_12, pair_21])
                else: continue
                
                # Lists of residues indexes for contacts
                contact_res_self = protein.get_contacts_2mers_self_res(partner = partner_P)
                contact_res_partner = protein.get_contacts_2mers_partner_res(partner = partner_P )
                
                # Coordinates
                contact_res_xyz_self = protein.get_res_xyz(contact_res_self)
                contact_res_xyz_partner = partner_P.get_res_xyz(contact_res_partner)
                
                # Residues names
                contact_res_name_self = protein.get_res_names(contact_res_self)
                contact_res_name_partner = partner_P.get_res_names(contact_res_partner)
                
                cont_num = len(contact_res_self)
                
                print(f"   - Adding {cont_num} contacts between {protein.get_ID()} and {partner_P.get_ID()}.")
                
                # Add one contact line at a time
                for contact_i in range(len(contact_res_self)):
                    
                    prot_plus_res_name_1 = (protein.get_ID()  , contact_res_name_self[contact_i]   )
                    prot_plus_res_name_2 = (partner_P.get_ID(), contact_res_name_partner[contact_i])
                    
                    # Check if the residue is already in shared_residues
                    if (prot_plus_res_name_1 in shared_residues) or (prot_plus_res_name_2 in shared_residues): shared_residue = True
                    else: shared_residue = False
                    
                    fig.add_trace(go.Scatter3d(
                        x = (contact_res_xyz_self[contact_i][0],) + (contact_res_xyz_partner[contact_i][0],) + (None,),  # Add None to create a gap between points_A and points_B
                        y = (contact_res_xyz_self[contact_i][1],) + (contact_res_xyz_partner[contact_i][1],) + (None,),
                        z = (contact_res_xyz_self[contact_i][2],) + (contact_res_xyz_partner[contact_i][2],) + (None,),
                        mode='lines',
                        line=dict(color = shared_resid_line_color if shared_residue else 'gray', width=1,
                                  dash='solid' if shared_residue else 'dot'),
                        showlegend = False,
                        name = contact_res_name_self[contact_i] + "-" + contact_res_name_partner[contact_i]
                    ))
                    
        
        # Add label for contact and for shared residues
        fig.add_trace(go.Scatter3d(
            x=[None], y=[None], z=[None],
            mode='lines',
            line=dict(color="gray", width=1, dash = "dot"),
            name='Contacts',
            showlegend=True
        )).add_trace(go.Scatter3d(
            x=[None], y=[None], z=[None],
            mode='lines+markers',
            line=dict(color=shared_resid_line_color, width=1),
            marker=dict(symbol='circle', size=8, color=shared_residue_color),
            name='Shared Residues',
            # label='Shared Residues',
            showlegend=True
            ))
        
        
        # Some view preferences
        fig.update_layout(
            legend=legend_position,
            scene=dict(
                # Show grid?
                xaxis=dict(showgrid=showgrid), yaxis=dict(showgrid=showgrid), zaxis=dict(showgrid=showgrid),
                # Show axis?
                xaxis_visible = show_axis, yaxis_visible = show_axis, zaxis_visible = show_axis,
                xaxis_title='X (Å)',
                yaxis_title='Y (Å)',
                zaxis_title='Z (Å)',
                aspectmode="data", 
                # Allow free rotation along all axis
                dragmode="orbit"
            ),
            # Adjust layout margins
            margin=margin
        )

        # Display the plot?
        if show_plot == True: plot(fig)
        
        # Save the plot?
        if save_html != None: fig.write_html(save_html), print(f"   - Plot saved in: {save_html}")
        
        return fig
        
    def plot_fully_connected2(
            self, Nmers = False, custom_res_colors = None, custom_contact_colors = None,
            res_size = 5, CM_size = 10, contact_line_width = 5, contact_res_size = 5,
            non_contact_res_opacity = 0.2, contact_res_opacity = 0.7,
            show_plot = True, save_html = None, legend_position = dict(x=1.02, y=0.5),
            plddt_cutoff = 0, showgrid = True, margin = dict(l=0, r=0, b=0, t=0),
            shared_resid_line_color = "red" , shared_residue_color = 'black', homodimer_same_resid_ellipse_color = "red",
            not_shared_resid_line_color = "gray", is_debug = True, show_axis = True,
            add_backbones = True, visible_backbones = True, name_offset = 15):
        '''
        
        Parameters:
            - specific_partners (list): partners to graph with the protein. If
                None (default), protein will be plotted with all of its partners.
            - custom_res_colors (list): list of colors to color each protein
            - res_size (float): size of non-contact residue centroids.
            - non_contact_res_opacity (float 0 to 1): opacity of non-contact residue centroids.
            - contact_res_opacity (float 0 to 1): 
            - plddt_cutoff (float): show only residues with pLDDT that surpass plddt_cutoff (0 to 100).
            - CM_size (float): size to represent the center of mass (in Å)
            - contact_line_width (float): width of the line between the CM and the contact residue (in Å)
            - contact_res_size (float): size of residues centroids that are in contact with other
                proteins (in Å).
            - show_plot (bool): displays the plot as html in the browser.
            - save_html (str): path to output HTML file.
            - showgrid (bool): show the background grid?
            - show_axis (bool): show the background box with axis?
            - margin (dict): margin sizes.
            - legend_position (dict): protein names legend position.
            - shared_resid_line_color (str): color of the lines connecting residues in which
                at least one is co-occupant with another residue.
            - shared_residue_color (str): color of the residue centroids that have multiple
                proteins in contact with it (co-occupancy).
            - add_backbones (bool): set to False to avoid the addition of protein backbones. 
            - visible_backbones (bool): set to True if you want the backbones to initialize visible.
            - name_offset (int): Angstroms to offset the names trace in the plus Z-axis direction.
                default = 15.
            - is_debug (bool): For developers. Sets ON some debug prints.
        '''
        
        # Progress
        print("")
        print(f"INITIALIZING: Plotting fully connected network for {self.ID}:")
        
        # Set color palette
        if custom_res_colors != None: res_colors = custom_res_colors
        else: res_colors = Protein.default_color_palette
                        
        # Get the fully connected partners
        partners_list = self.get_partners_of_partners()
        all_proteins  = [self] + partners_list
        prot_num = len(all_proteins)
        
        # Progress
        print(f"   - Protein network contains {prot_num} proteins: {[prot.get_ID() for prot in all_proteins]}")
        print( "   - Initializing 3D figure.")            
                
        # Initialize figure
        fig = go.Figure()
                
        # To keep track of already added contacts for each pair
        already_computed_pairs = []
        
        # Get the shared residues for the network (as tuples: (ProteinID, res_name))
        shared_residues = self.get_network_shared_residues()[0]
        
        # Dataframe for plotting proteins as one trace
        proteins_df = pd.DataFrame(columns = [
            "protein",
            "protein_num",      # To assign one color palette to each protein
            "protein_ID",
            "res",              # 0 base res index
            "res_name",
            "CM_x", "CM_y", "CM_z",
            "x", "y", "z",
            "plDDT",
            "is_contact",       # 0: no contact, >=1: contact with partner N, 99: is shared
            "color",
            ])
        
        # Dataframe for plotting contacts as one trace
        contacts_2mers_df = pd.DataFrame(columns = [
            # For protein
            "protein1",
            "protein_ID1",
            "res1",
            "res_name1",
            "x1", "y1", "z1",
            
            # For partner
            "protein2",
            "protein_ID2",
            "res2",
            "res_name2",
            "x2", "y2", "z2",
            
            # Metadata
            "is_shared",         # 0: not shared, 1: shared
            "color",
            "linetype",
            "min_plDDT"
            ])
        
        # Cast to bool type (to avoid warnings)
        contacts_2mers_df["is_shared"] = contacts_2mers_df["is_shared"].astype(bool)
        contacts_2mers_df["linetype"] = contacts_2mers_df["linetype"].astype(bool)
        
        # Extract data for proteins_df
        for P, protein in enumerate(all_proteins):
            
            # Progress
            print(f"   - Extracting {protein.get_ID()} coordinates.")            
            
            # Get protein residues xyz and other data
            prot_df =  pd.DataFrame({
                "protein": protein,
                "protein_num": [P] * len(protein.get_res_xyz()),
                "protein_ID": [protein.get_ID()] * len(protein.get_res_xyz()),
                "res": list(range(len(protein.get_res_xyz()))),
                "res_name": protein.get_res_names(),
                "CM_x": [protein.get_CM()[0]] * len(protein.get_res_xyz()),
                "CM_y": [protein.get_CM()[1]] * len(protein.get_res_xyz()),
                "CM_z": [protein.get_CM()[2]] * len(protein.get_res_xyz()),                
                "x": [xyz[0] for xyz in protein.get_res_xyz()],
                "y": [xyz[1] for xyz in protein.get_res_xyz()],
                "z": [xyz[2] for xyz in protein.get_res_xyz()],
                "plDDT": protein.get_res_pLDDT(),
                
                # Colors and contacts are defined above
                "is_contact": [0] * len(protein.get_res_xyz()),
                "color": [res_colors[list(res_colors.keys())[P]][3]] * len(protein.get_res_xyz()), ######
                })
            
            # add data to proteins_df
            proteins_df = pd.concat([proteins_df, prot_df], ignore_index = True)
            
            # Current protein partners
            partners_P = protein.get_partners()
            
            # Extract contacts partner by partner
            for partner_P in partners_P:
                
                # Check if protein pair was already analyzed in both directions
                pair_12 = (protein.get_ID(), partner_P.get_ID())
                pair_21 = (partner_P.get_ID(), protein.get_ID())
                if pair_12 not in already_computed_pairs:
                    already_computed_pairs.extend([pair_12, pair_21])
                else: continue
                
                # Lists of residues indexes for contacts
                contact_res_self = protein.get_contacts_2mers_self_res(partner = partner_P)
                contact_res_partner = protein.get_contacts_2mers_partner_res(partner = partner_P )
                
                # Coordinates
                contact_res_xyz_self = protein.get_res_xyz(contact_res_self)
                contact_res_xyz_partner = partner_P.get_res_xyz(contact_res_partner)
                
                # Residues names
                contact_res_name_self = protein.get_res_names(contact_res_self)
                contact_res_name_partner = partner_P.get_res_names(contact_res_partner)
                
                cont_num = len(contact_res_self)
                
                print(f"   - Extracting {cont_num} contacts between {protein.get_ID()} and {partner_P.get_ID()}.")
                
                # Add one contact line at a time
                for contact_i in range(len(contact_res_self)):
                    
                    # Get the residue pair identifier (protein_ID, res_name) 
                    prot_plus_res_name_1 = (protein.get_ID()  , contact_res_name_self[contact_i]   )
                    prot_plus_res_name_2 = (partner_P.get_ID(), contact_res_name_partner[contact_i])
                    
                    # Check if the residue is already in shared_residues
                    if (prot_plus_res_name_1 in shared_residues) or (prot_plus_res_name_2 in shared_residues): shared_residue = True
                    else: shared_residue = False
                    
                    # Extract contact data
                    cont_df =  pd.DataFrame({
                        # Protein contact residue data
                        "protein1":     [protein],
                        "protein_ID1": [protein.get_ID()],
                        "res1":         [contact_res_self[contact_i]],
                        "res_name1":    [contact_res_name_self[contact_i]],
                        "x1": [contact_res_xyz_self[contact_i][0]],
                        "y1": [contact_res_xyz_self[contact_i][1]],
                        "z1": [contact_res_xyz_self[contact_i][2]],
                        
                        # Partner contact residue data
                        "protein2":     [partner_P],
                        "protein_ID2":  [partner_P.get_ID()],
                        "res2":         [contact_res_partner[contact_i]],
                        "res_name2":    [contact_res_name_partner[contact_i]],
                        "x2": [contact_res_xyz_partner[contact_i][0]],
                        "y2": [contact_res_xyz_partner[contact_i][1]],
                        "z2": [contact_res_xyz_partner[contact_i][2]],
                        
                        # Contact metadata
                        "is_shared": [shared_residue],         # 0: not shared, 1: shared
                        "color": ["black" if shared_residue else "gray"],
                        "linetype": [shared_residue],
                        "min_plDDT": [min([protein.get_res_pLDDT  ([contact_res_self   [contact_i]]),
                                           partner_P.get_res_pLDDT([contact_res_partner[contact_i]])])]
                        })
                        
                    
                    # add contact data to contacts_2mers_df
                    contacts_2mers_df = pd.concat([contacts_2mers_df, cont_df], ignore_index = True)
                    
        # Redefine dynamic contacts (using Nmers data!)
        if Nmers == True:
            pass
                    
        # Set color scheme for residues
        print("   - Setting color scheme.")
        for protein in proteins_df.groupby("protein"):
            for P, partner in enumerate(protein[0].get_partners()):
                for res in list(protein[1]["res"]):
                    # If the residue is involved in a contact
                    if res in protein[0].get_contacts_2mers_self_res(partner):
                        mask = (proteins_df["protein"] == protein[0]) & (proteins_df["res"] == res)
                        proteins_df.loc[mask, "is_contact"] = P + 1
                        proteins_df.loc[mask, "color"] = res_colors[list(res_colors.keys())[list(protein[1]["protein_num"])[0]]][-(P+1)]
                    # If the residue is involved in multiple contacts
                    if (protein[0].get_ID(), protein[0].get_res_names([res])[0]) in shared_residues:
                        mask = (proteins_df["protein"] == protein[0]) & (proteins_df["res"] == res)
                        proteins_df.loc[mask, "is_contact"] = 99
                        proteins_df.loc[mask, "color"] = shared_residue_color
                    
        # proteins_base_colors_df = proteins_df.loc[(proteins_df["is_contact"] == 0)].filter(["protein","protein_ID", "color"]).drop_duplicates()
                
        # Add protein CM trace ------------------------------------------------
        print("   - Adding CM trace.")
        prot_names_df = proteins_df.filter(["protein_ID", "protein_num", "CM_x", "CM_y", "CM_z"]).drop_duplicates()
        CM_colors = [res_colors[list(res_colors.keys())[num]][-1] for num in prot_names_df["protein_num"]]
        fig.add_trace(go.Scatter3d(            
            x = prot_names_df["CM_x"],
            y = prot_names_df["CM_y"],
            z = prot_names_df["CM_z"],
            mode='markers',
            marker=dict(
                size = CM_size,
                color = CM_colors,
                opacity = contact_res_opacity,
                line=dict(
                    color='gray',
                    width=1
                    ),
            ),
            name = "Center of Masses (CM)",
            showlegend = True,
            hovertext = prot_names_df["protein_ID"]
        ))
        
        # Add residue-residue contacts trace ----------------------------------
        
        # Single contacts
        print("   - Adding single contacts trace.")
        contacts_2mers_df_not_shared = insert_none_row(contacts_2mers_df.query('is_shared == False'))
        single_contacts_names_list = []
        for res_name1, res_name2 in zip(contacts_2mers_df.query('is_shared == False')["res_name1"],
                                        contacts_2mers_df.query('is_shared == False')["res_name2"]):
            single_contacts_names_list.append(res_name1 + "/" + res_name2)
            single_contacts_names_list.append(res_name1 + "/" + res_name2)
            single_contacts_names_list.append(res_name1 + "/" + res_name2)
            single_contacts_names_list.append(res_name1 + "/" + res_name2)
        fig.add_trace(go.Scatter3d(
            x=contacts_2mers_df_not_shared[["x1", "x2"]].values.flatten(),
            y=contacts_2mers_df_not_shared[["y1", "y2"]].values.flatten(),
            z=contacts_2mers_df_not_shared[["z1", "z2"]].values.flatten(),
            mode='lines',
            line=dict(
                color = not_shared_resid_line_color,
                width = 1,
                dash = 'solid'
            ),
            opacity = contact_res_opacity,
            name = "Simple contacts",
            showlegend = True,
            # hovertext = contacts_2mers_df_not_shared["res_name1"] + "-" + contacts_2mers_df_not_shared["res_name2"]
            hovertext = single_contacts_names_list
        ))
        
        # Co-occupant contacts
        print("   - Adding co-occupant contacts trace.")
        contacts_2mers_df_shared = insert_none_row(contacts_2mers_df.query('is_shared == True'))
        shared_contacts_names_list = []
        for res_name1, res_name2 in zip(contacts_2mers_df.query('is_shared == True')["res_name1"],
                                        contacts_2mers_df.query('is_shared == True')["res_name2"]):
            shared_contacts_names_list.append(res_name1 + "/" + res_name2)
            shared_contacts_names_list.append(res_name1 + "/" + res_name2)
            shared_contacts_names_list.append(res_name1 + "/" + res_name2)
            shared_contacts_names_list.append(res_name1 + "/" + res_name2)
        fig.add_trace(go.Scatter3d(
            x=contacts_2mers_df_shared[["x1", "x2"]].values.flatten(),
            y=contacts_2mers_df_shared[["y1", "y2"]].values.flatten(),
            z=contacts_2mers_df_shared[["z1", "z2"]].values.flatten(),
            mode='lines',
            line=dict(
                color = shared_resid_line_color,
                width = 1,
                dash = 'solid'
            ),
            opacity = contact_res_opacity,
            name = "Dynamic contacts",
            showlegend = True,
            # hovertext = contacts_2mers_df_shared["res_name1"] + "-" + contacts_2mers_df_shared["res_name2"]
            hovertext = shared_contacts_names_list
        ))
        
        # Contacts that are the same residue on the same protein (homodimers) --------------
        print("   - Adding self residue contacts for homodimers trace (loops).")
        contacts_2mers_df_homodimers = contacts_2mers_df.query('(protein1 == protein2) & (res1 == res2)')
        ellipses_resolution = 30
        # Hovertext names
        homodimers_loop_contacts_names_list = []
        for res_name1, res_name2 in zip(contacts_2mers_df.query('(protein1 == protein2) & (res1 == res2)')["res_name1"],
                                        contacts_2mers_df.query('(protein1 == protein2) & (res1 == res2)')["res_name2"]):            
            # Add the same name X times
            for i in range(ellipses_resolution + 1): homodimers_loop_contacts_names_list.append(res_name1 + "/" + res_name2)
        # Colors
        homodimers_loop_contacts_colors_list = []
        for C, contact_row in contacts_2mers_df.query('(protein1 == protein2) & (res1 == res2)').iterrows():
            # Add the same name X times
            for i in range(ellipses_resolution + 1):
                if contact_row["is_shared"]: 
                    homodimers_loop_contacts_colors_list.append(shared_resid_line_color)
                else:
                    homodimers_loop_contacts_colors_list.append(not_shared_resid_line_color)
        # Generate ellipses
        ellipses_x = []
        ellipses_y = []
        ellipses_z = []
        for protein in set(contacts_2mers_df_homodimers["protein1"]):
            points = protein.get_res_xyz()
            reference_point = protein.get_CM()
            subset_indices = list(contacts_2mers_df_homodimers["res1"])
            ellip_x, ellip_y, ellip_z = draw_ellipses(points, reference_point, subset_indices, ellipses_resolution = 30, is_debug = False)
            ellipses_x += ellip_x
            ellipses_y += ellip_y
            ellipses_z += ellip_z
        fig.add_trace(go.Scatter3d(
            x=ellipses_x,
            y=ellipses_y,
            z=ellipses_z,
            mode='lines',
            line=dict(
                color = homodimers_loop_contacts_colors_list,
                width = 1,
                dash = 'solid'
            ),
            opacity = contact_res_opacity,
            name = "Self residue contacts (loops)",
            showlegend = True,
            # hovertext = contacts_2mers_df_shared["res_name1"] + "-" + contacts_2mers_df_shared["res_name2"]
            hovertext = homodimers_loop_contacts_names_list
        ))
        
        # Add protein backbones with domains and pLDDT ------------------------
        if add_backbones:
            for protein in all_proteins:
                backbone_fig = plot_backbone(protein_chain = protein.PDB_chain,
                                             domains = protein.domains,
                                             protein_ID = protein.get_ID(),
                                             return_fig = True,
                                             is_for_network = True)
                for trace in backbone_fig["data"]:
                    if not visible_backbones: trace["visible"] = 'legendonly'
                    fig.add_trace(trace)
                
        # Add protein residues NOT involved in contacts -----------------------
        print("   - Adding protein trace: residues not involved in contacts.")
        proteins_df_non_c = proteins_df.query('is_contact == 0')
        fig.add_trace(go.Scatter3d(
            x=proteins_df_non_c["x"],
            y=proteins_df_non_c["y"],
            z=proteins_df_non_c["z"],
            mode='markers',
            marker=dict(
                symbol='circle',
                size = contact_res_size,                        
                color = proteins_df_non_c["color"],
                opacity = non_contact_res_opacity,
                line=dict(
                    color='gray',
                    width=1
                    ),
            ),
            name = "Non-contact residues",
            showlegend = True,
            hovertext = proteins_df_non_c["protein_ID"] + "-" + proteins_df_non_c["res_name"],
            visible = 'legendonly'
        ))
        
        # Add protein residues involved in contacts ---------------------------
        print("   - Adding protein trace: residues involved in contacts.")
        proteins_df_c = proteins_df.query('is_contact > 0')
        
        # Lines
        proteins_df_c2 = insert_none_row_with_color(proteins_df_c)
        colors_list = []
        for color in proteins_df_c2["color"]:
            colors_list.append(color)
            colors_list.append(color)
        protein_IDs_resnames_list = []
        for ID, res_name in zip(proteins_df_c["protein_ID"], proteins_df_c["res_name"]):
            protein_IDs_resnames_list.append(ID + "-" + res_name)
            protein_IDs_resnames_list.append(ID + "-" + res_name)
            protein_IDs_resnames_list.append(ID + "-" + res_name)
            protein_IDs_resnames_list.append(ID + "-" + res_name)
        fig.add_trace(go.Scatter3d(
            x=proteins_df_c2[["CM_x", "x"]].values.flatten(),
            y=proteins_df_c2[["CM_y", "y"]].values.flatten(),
            z=proteins_df_c2[["CM_z", "z"]].values.flatten(),
            mode='lines',
            line=dict(
                color = colors_list,
                width = contact_line_width,
                dash = 'solid',
            ),
            opacity = contact_res_opacity,
            name = "Contact residues lines",
            showlegend = True,
            # hovertext = proteins_df_c2["protein_ID"] + "-" + proteins_df_c2["res_name"]
            hovertext = protein_IDs_resnames_list
        ))
        
        # Markers
        fig.add_trace(go.Scatter3d(
            x=proteins_df_c["x"],
            y=proteins_df_c["y"],
            z=proteins_df_c["z"],
            mode='markers',
            marker=dict(
                symbol='circle',
                size = contact_res_size,                        
                color = proteins_df_c["color"],
                opacity = contact_res_opacity,
                line=dict(
                    color='gray',
                    width=1
                    ),
            ),
            name = "Contact residues centroids",
            showlegend = True,
            hovertext = proteins_df_c["protein_ID"] + "-" + proteins_df_c["res_name"],
        ))
        
        # Add label for shared residues ---------------------------------------
        fig.add_trace(go.Scatter3d(
            x=[None], y=[None], z=[None],
            mode='lines+markers',
            line=dict(color = shared_residue_color, width = 1),
            marker=dict(symbol='circle', size = 5, color = shared_residue_color),
            name='Shared Residues',
            showlegend=True
            ))
        
        # Add protein NAMES trace ---------------------------------------------
        print("   - Adding protein names trace.")
        fig.add_trace(go.Scatter3d(
            x = prot_names_df["CM_x"],
            y = prot_names_df["CM_y"],
            z = prot_names_df["CM_z"] + name_offset,
            text = prot_names_df["protein_ID"],
            mode = 'text',
            textposition = 'top center',
            textfont = dict(size = 30, color = "black"),# res_colors[P]),  # Adjust size and color as needed
            name = "Protein IDs",
            showlegend = True            
        ))

        # Some view preferences -----------------------------------------------
        print("   - Setting layout.")
        fig.update_layout(
            legend=legend_position,
            scene=dict(
                # Show grid?
                xaxis=dict(showgrid=showgrid), yaxis=dict(showgrid=showgrid), zaxis=dict(showgrid=showgrid),
                # Show axis?
                xaxis_visible = show_axis, yaxis_visible = show_axis, zaxis_visible = show_axis,
                xaxis_title='X (Å)',
                yaxis_title='Y (Å)',
                zaxis_title='Z (Å)',
                aspectmode="data", 
                # Allow free rotation along all axis
                dragmode="orbit"
            ),
            # Adjust layout margins
            margin=margin
        )
        
        # make the label to enter in the hovertext square
        # fig.update_scenes(hoverlabel=dict(namelength=-1))

        # Display the plot?
        if show_plot == True: plot(fig)
        
        # Save the plot?
        if save_html != None: fig.write_html(save_html), print(f"   - Plot saved in: {save_html}")
        
        return fig, contacts_2mers_df, proteins_df, proteins_df_c, proteins_df_c2      
    
        
    def __str__(self):
        return f"Protein ID: {self.ID} (tag = {self.protein_tag}) --------------------------------------------\n>{self.ID}\n{self.seq}\n   - Center of Mass (CM): {self.CM}\n   - Partners: {str([partner.get_ID() for partner in self.partners])}"



################################################# EDIT LATER #################################################
################################################# EDIT LATER #################################################
################################################# EDIT LATER #################################################
################################################# EDIT LATER #################################################

class Protein_Nmers(Protein):
    '''
    Extended subclass of Protein to allow working with Nmers datasets.
    '''
    
    def __init__(self, ID, seq, symbol = None, name = None, res_xyz = [],
                 res_names = [], CM = None):
        
        #### Coming from 2mers dataset
        Protein.__init__(self, ID, seq, symbol = None, name = None, res_xyz = [],
                     res_names = [], CM = None, has_nmer = False)        
        
        #### Coming from Nmers dataset (to explore potential dynamic contacts)
        self.contacts_Nmers_proteins = []     # List with lists of Proteins involved in each Nmer model
        self.contacts_Nmers_self     = []     # List with lists of self residues contacts involved in each Nmer model for each partner
        self.contacts_Nmers_partners = []     # List with lists of partner residues contacts involved in each Nmer model for each partner
        self.Nmers_partners_min_PAEs = []
        self.Nmers_partners_mean_PAEs= []     # Mean of the PAE matrix will contain info about dynamic contacts (if they increase => potential addition of contacts, and vice versa)
        self.Nmers_partners_N_models = []
    
    # Add a partner
    def add_partner(self, partner_protein,
                    # 2mers information
                    partner_ipTM, partner_min_PAE, partner_N_models,
                    self_res_contacts, other_res_contacts,
                    # Nmers information
                    Nmers_partners_min_PAEs, Nmers_partners_N_models,
                    contacts_Nmers_proteins,                            
                    contacts_Nmers_self, contacts_Nmers_partners
                    ):
        '''
        To use it, first generate all the instances of proteins and then 
        '''
        Protein.add_partner(self, partner_protein, partner_ipTM, partner_min_PAE, 
                            partner_N_models, self_res_contacts, other_res_contacts)
        
        self.Nmers_partners_min_PAEs.append( )
        self.Nmers_partners_mean_PAEs.append()
        self.Nmers_partners_N_models.append( )
        self.contacts_Nmers_proteins.append( )
        self.contacts_Nmers_self.append(     )
        self.contacts_Nmers_partners.append( )


class Network(object):
    
    def __init__(self):
        raise AttributeError("Network class not implemented yet")
        
        self.proteins = []
        self.proteins_IDs = []
        self.is_solved = False
    
    def add_proteins_from_contacts_2mers_df(self, contacts_2mers_df):
        raise AttributeError("Network class not implemented yet")
    
    def solve(self):
        raise AttributeError("Network class not implemented yet")
        
        self.is_solved = True
        
    def __add__(self, other):
        raise AttributeError("Network class not implemented yet")
        
        # If the addition is another Network
        if isinstance(other, Network):
            print(f"Adding networks: {self.ID} + {other.get_ID()}")
            # Add the other network
            raise AttributeError("Not Implemented")
        
        # If the addition is a Protein
        elif isinstance(other, Protein):
            print(f"Adding protein to the {self.ID} network: {other.get_ID()}")
            # Add the other Protein to the network 
            raise AttributeError("Not Implemented")
            
        else:
            raise ValueError(f"The + operator can not be used between instances of types {type(self)} and {type(other)}. Only Network+Protein, Network+Network and Protein+Protein are allowed.")
            
    def __str__(self):
        return self.proteins_IDs

################################################# EDIT LATER #################################################
################################################# EDIT LATER #################################################
################################################# EDIT LATER #################################################
################################################# EDIT LATER #################################################


def create_proteins_dict_from_contacts_2mers_df(contacts_2mers_df, sliced_PAE_and_pLDDTs, plot_proteins = False, print_proteins = False):

    # Progress
    print("INITIALIZING: creating dictionary with Protein objects from contacts_2mers_df")

    # Extract all protein_IDs from the contacts_2mers_df
    protein_IDs_list = list(set(list(contacts_2mers_df["protein_ID_a"]) +
                                list(contacts_2mers_df["protein_ID_b"])))
    
    # Creation (To implement in class Network)

    proteins_dict = {}
    for protein_ID in protein_IDs_list:
        protein_PDB_chain = sliced_PAE_and_pLDDTs[protein_ID]["PDB_xyz"]
        proteins_dict[protein_ID] = Protein(protein_ID, protein_PDB_chain, sliced_PAE_and_pLDDTs)
        
        if plot_proteins: proteins_dict[protein_ID].plot_alone()
        if print_proteins: print(proteins_dict[protein_ID])
        
    return proteins_dict


def add_partners_to_proteins_dict_from_contacts_2mers_df(proteins_dict, contacts_2mers_df):

    # Add partners one by one for each protein
    for protein in proteins_dict.values():
        protein.add_partners_from_contacts_2mers_df(contacts_2mers_df)
