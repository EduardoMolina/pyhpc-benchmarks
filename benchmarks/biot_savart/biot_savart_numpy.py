import numpy as np
import math

def collapsed_velocities_from_line_vortices(
    points,
    origins,
    terminations,
    strengths,
    ages=None,
    nu=0.0,
):
    """This function takes in a group of points, and the attributes of a group of
    line vortices. At every point, it finds the cumulative induced velocity due to
    all the line vortices.

    Citation: The equations in this function are from "Extended Unsteady
    Vortex-Lattice Method for Insect Flapping Wings" (Nguyen et al., 2016)

    Note: This function uses a modified version of the Bio-Savart law to create a
    smooth induced velocity decay based on a vortex's core radius. The radius is
    determined based on a vortex's age and kinematic viscosity. If the age of the
    vortex is 0.0 seconds, the radius is set to 0.0 meters. The age of a vortex in
    only relevant for vortices that have been shed into the wake.

    Note: This function's performance has been highly optimized for unsteady
    simulations via Numba. While using Numba dramatically increases unsteady
    simulation performance, it does cause a performance drop for the less intense
    steady simulations.

    :param points: 2D array of floats
        This variable is an array of shape (N x 3), where N is the number of points.
        Each row contains the x, y, and z float coordinates of that point's position
        in meters.
    :param origins: 2D array of floats
        This variable is an array of shape (M x 3), where M is the number of line
        vortices. Each row contains the x, y, and z float coordinates of that line
        vortex's origin's position in meters.
    :param terminations: 2D array of floats
        This variable is an array of shape (M x 3), where M is the number of line
        vortices. Each row contains the x, y, and z float coordinates of that line
        vortex's termination's position in meters.
    :param strengths: 1D array of floats
        This variable is an array of shape (, M), where M is the number of line
        vortices. Each position contains the strength of that line vortex in meters
        squared per second.
    :param ages: 1D array of floats, optional
        This variable is an array of shape (, M), where M is the number of line
        vortices. Each position contains the age of that line vortex in seconds. This
        is only relevant for vortices that have been shed into the wake. The default
        value is None. If the age of a specific vortex is 0.0 seconds, then the
        vortex core radius is set to 0.0 meters.
    :param nu: float, optional
        This variable is a float that represents the kinematic viscosity of the fluid
        in meters squared per second. The default value is 0.0 meters squared per
        second.
    :return velocities: 2D array of floats
        This is an array of shape (N x 3), and it holds the cumulative induced
        velocity at each of the N points due to all the line vortices. The units are
        meters per second.
    """
    squire = 10**-4
    lamb = 1.25643
    # Set the value of the local machine error. This will be used to fix removable
    # discontinuities in the induced velocity functions.
    eps = np.finfo(float).eps

    num_vortices = origins.shape[0]
    num_points = points.shape[0]

    # Initialize an empty array, which we will fill with the induced velocities.
    velocities = np.zeros((num_points, 3))

    # If the user didn't specify any ages, set the age of each vortex to 0.0 seconds.
    if ages is None:
        ages = np.zeros(num_vortices)

    for vortex_id in range(num_vortices):
        origin = origins[vortex_id]
        termination = terminations[vortex_id]
        strength = strengths[vortex_id]
        age = ages[vortex_id]

        # Calculate the radius of the vortex's core. If the age is 0.0 seconds,
        # this will evaluate to be 0.0 meters.
        r_c = 2 * math.sqrt(lamb * (nu + squire * abs(strength)) * age)

        # The r_0 vector goes from the line vortex's origin to its termination.
        r_0_x = termination[0] - origin[0]
        r_0_y = termination[1] - origin[1]
        r_0_z = termination[2] - origin[2]

        # Find the r_0 vector's length.
        r_0 = math.sqrt(r_0_x**2 + r_0_y**2 + r_0_z**2)

        c_1 = strength / (4 * math.pi)
        c_2 = r_0**2 * r_c**2

        for point_id in range(num_points):
            point = points[point_id]

            # The r_1 vector goes from the point to the line vortex's origin.
            r_1_x = origin[0] - point[0]
            r_1_y = origin[1] - point[1]
            r_1_z = origin[2] - point[2]

            # The r_2 vector goes from the point to the line vortex's termination.
            r_2_x = termination[0] - point[0]
            r_2_y = termination[1] - point[1]
            r_2_z = termination[2] - point[2]

            # The r_3 vector is the cross product of the r_1 and r_2 vectors.
            r_3_x = r_1_y * r_2_z - r_1_z * r_2_y
            r_3_y = r_1_z * r_2_x - r_1_x * r_2_z
            r_3_z = r_1_x * r_2_y - r_1_y * r_2_x

            # Find the r_1, r_2, and r_3 vectors' lengths.
            r_1 = math.sqrt(r_1_x**2 + r_1_y**2 + r_1_z**2)
            r_2 = math.sqrt(r_2_x**2 + r_2_y**2 + r_2_z**2)
            r_3 = math.sqrt(r_3_x**2 + r_3_y**2 + r_3_z**2)

            c_3 = r_1_x * r_2_x + r_1_y * r_2_y + r_1_z * r_2_z

            # If part of the vortex is so close to the point that they are touching (
            # within machine epsilon), there is a removable discontinuity. In this
            # case, continue to the next point because there is no velocity induced
            # by the current vortex at this point.
            if r_1 < eps or r_2 < eps or r_3**2 < eps:
                continue
            else:
                c_4 = (
                    c_1
                    * (r_1 + r_2)
                    * (r_1 * r_2 - c_3)
                    / (r_1 * r_2 * (r_3**2 + c_2))
                )
                velocities[point_id, 0] += c_4 * r_3_x
                velocities[point_id, 1] += c_4 * r_3_y
                velocities[point_id, 2] += c_4 * r_3_z

    return velocities

def collapsed_velocities_from_ring_vortices(
    points,
    back_right_vortex_vertices,
    front_right_vortex_vertices,
    front_left_vortex_vertices,
    back_left_vortex_vertices,
    strengths,
    ages=None,
    nu=0.0,
):
    """This function takes in a group of points, and the attributes of a group of
    ring vortices. At every point, it finds the cumulative induced velocity due to
    all the ring vortices.

    Note: This function's performance has been highly optimized for unsteady
    simulations via Numba. While using Numba dramatically increases unsteady
    simulation performance, it does cause a performance drop for the less intense
    steady simulations.

    :param points: 2D array of floats
        This variable is an array of shape (N x 3), where N is the number of points.
        Each row contains the x, y, and z float coordinates of that point's position
        in meters.
    :param back_right_vortex_vertices: 2D array of floats
        This variable is an array of shape (M x 3), where M is the number of ring
        vortices. Each row contains the x, y, and z float coordinates of that ring
        vortex's back right vertex's position in meters.
    :param front_right_vortex_vertices: 2D array of floats
        This variable is an array of shape (M x 3), where M is the number of ring
        vortices. Each row contains the x, y, and z float coordinates of that ring
        vortex's front right vertex's position in meters.
    :param front_left_vortex_vertices: 2D array of floats
        This variable is an array of shape (M x 3), where M is the number of ring
        vortices. Each row contains the x, y, and z float coordinates of that ring
        vortex's front left vertex's position in meters.
    :param back_left_vortex_vertices: 2D array of floats
        This variable is an array of shape (M x 3), where M is the number of ring
        vortices. Each row contains the x, y, and z float coordinates of that ring
        vortex's front left vertex's position in meters.
    :param strengths: 1D array of floats
        This variable is an array of shape (, M), where M is the number of ring
        vortices. Each holds the strength of that ring vortex in meters squared per
        second.
    :param ages: 1D array of floats, optional
        This variable is an array of shape (, M), where M is the number of line
        vortices. Each position contains the age of that ring vortex in seconds. This
        is only relevant for vortices that have been shed into the wake. The default
        value is None. If the age of a specific vortex is 0.0 seconds, then the
        vortex core radius is set to 0.0 meters.
    :param nu: float, optional
        This variable is a float that represents the kinematic viscosity of the fluid
        in meters squared per second. The default value is 0.0 meters squared per
        second.
    :return velocities: 2D array of floats
        This is an array of shape (N x 3), and it holds the cumulative induced
        velocity at each of the N points due to all the ring vortices. The units are
        meters per second.
    """
    origins_list = [
        back_right_vortex_vertices,
        front_right_vortex_vertices,
        front_left_vortex_vertices,
        back_left_vortex_vertices,
    ]
    terminations_list = [
        front_right_vortex_vertices,
        front_left_vortex_vertices,
        back_left_vortex_vertices,
        back_right_vortex_vertices,
    ]
    induced_velocities = np.zeros((points.shape[0], 3))

    # Get the velocity induced by each leg of the ring vortex.
    for i in range(4):
        induced_velocities += collapsed_velocities_from_line_vortices(
            points=points,
            origins=origins_list[i],
            terminations=terminations_list[i],
            strengths=strengths,
            ages=ages,
            nu=nu,
        )
    return induced_velocities

def run(points, back_right, front_right, front_left, back_left, strengths, ages, nu, device="cpu"):
    return collapsed_velocities_from_ring_vortices(points, back_right, front_right, front_left, back_left, strengths, ages, nu)