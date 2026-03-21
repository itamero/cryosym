"""
Generate grid of approximately equally spaced rotations.
"""

import numpy as np


def q_to_rot(q):
    """
    Convert quaternion into rotation matrix, assuming q = [x, y, z, w] with w the scalar part.
    This formula is taken from:
    https://en.wikipedia.org/wiki/Conversion_between_quaternions_and_Euler_angles.
    Note, the scalar w is the last element of the input array.
    :param q: quaternion as a list or numpy array of shape (4,)
    :return: 3x3 rotation matrix as numpy array
    """
    q = np.asarray(q)
    q = q / np.linalg.norm(q)
    x, y, z, w = q
    return np.array(
        [
            [1 - 2 * (y**2 + z**2), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x**2 + z**2), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x**2 + y**2)],
        ]
    )


def gen_rotations_grid(resolution):
    """
    Generate approximately equally spaced rotations.
    Based on aspire/abinitio/align/genRotationsGrid.m by Yariv Aizenbud, 31.01.2016

    Parametrization for SO(3)
       x = sin(tau) * sin(theta) * sin(phi)
       y = sin(tau) * sin(theta) * cos(phi)
       z = sin(tau) * cos(theta)
       w = cos(tau)

    :param resolution:      Number of samples per 2*pi
                            * Resolution = 50  generates   4484 rotations
                            * Resolution = 75  generates  15236 rotations
                            * Resolution = 100 generates  39365 rotations
                            * Resolution = 150 generates 129835 rotations
    :return rotations: (number of rotations)x3x3 numpy array of rotations

    """
    counter = 0
    tau1_step = (np.pi / 2) / (resolution / 4)
    for tau1 in np.arange(tau1_step / 2, np.pi / 2 - tau1_step / 2, tau1_step):
        theta1_step = np.pi / (resolution / 2 * np.sin(tau1))
        for theta1 in np.arange(theta1_step / 2, np.pi - theta1_step / 2, theta1_step):
            phi1_step = (2 * np.pi) / (resolution * np.sin(tau1) * np.sin(theta1))
            for _ in np.arange(0, 2 * np.pi - phi1_step, phi1_step):
                counter += 1

    n_of_rotations = counter
    rotations = np.zeros((n_of_rotations, 3, 3))

    counter = 0
    for tau1 in np.arange(tau1_step / 2, np.pi / 2 - tau1_step / 2, tau1_step):
        sintau1 = np.sin(tau1)
        costau1 = np.cos(tau1)
        theta1_step = np.pi / (resolution / 2 * sintau1)
        for theta1 in np.arange(theta1_step / 2, np.pi - theta1_step / 2, theta1_step):
            sintheta1 = np.sin(theta1)
            costheta1 = np.cos(theta1)
            phi1_step = (2 * np.pi) / (resolution * sintau1 * sintheta1)
            for phi1 in np.arange(0, 2 * np.pi - phi1_step, phi1_step):
                rotations[counter] = q_to_rot(
                    [
                        sintau1 * sintheta1 * np.sin(phi1),
                        sintau1 * sintheta1 * np.cos(phi1),
                        sintau1 * costheta1,
                        costau1,
                    ]
                )
                counter += 1

    return rotations


if __name__ == "__main__":
    RESOLUTION = 150
    grid = gen_rotations_grid(RESOLUTION)
    print(f"Generated {grid.shape[0]} rotations for resolution {RESOLUTION}")
