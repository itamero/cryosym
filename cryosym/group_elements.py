"""
This module is for generating the symmetry group elements of different
symmetry types: Cn, Dn, T, O, I.
Also provides utilities related to symmetry groups, such as finding
the normalizer, checking normality of subgroups, and identifying redundant
self common lines.
"""

import numpy as np
from scipy.spatial.transform import Rotation as R

def supported_symmetry_group(sym: str) -> str:
    """
    Check whether a symmetry group is supported. Raise an exception if not.

    :param sym: Symmetry type. One of 'Cn','Dn','T','O','I' where n>1
    :return: sym in uppercase
    """
    sym = sym.upper()
    if sym in ["T", "O", "I"]:
        return sym
    if len(sym) > 1:
        g_order = int(sym[1:])
        if g_order > 1 and sym[0] in ["C", "D"]:
            return sym
    raise TypeError(f"Symmetry type '{sym}' is not supported.")


def group_elements(sym: str) -> np.ndarray:
    """
    This function generates the symmetry group elements of the given symmetry type.

    :param sym: Symmetry type. One of 'Cn','Dn','T','O','I' where n>1
    :return:    Numpy array of size (N,3,3), where N is the group order
    """
    sym = supported_symmetry_group(sym)
    return R.create_group(sym).as_matrix()


def normalizer(sym: str) -> str:
    """
    Return string representing the normalizer of the given symmetry group in SO(3).
    1. For Dn, the normalizer is D2n, except for D2 where the normalizer is O.
    2. For T and O, the normalizer is O.
    3. For I, the normalizer is I.

    :param sym: must be one of 'Dn','T','O','I' where n>1
    :return: string representing the normalizer
    """
    sym = supported_symmetry_group(sym)
    if sym in ["T", "O", "D2"]:
        return "O"
    if sym == "I":
        return "I"
    if sym[0] == "D":
        return "D" + str(2 * int(sym[1:]))
    raise TypeError("Normalizer not available for this symmetry type")


def scl_inds_by_sym(sym: str, info: bool = False) -> list[int]:
    """
    Get indices of group elements to use for self common lines (SCL) based on symmetry type.
    This function identifies pairs of group elements that are transposes of each other
    and excludes one from each pair, as well as the identity element.
    Based on "A common lines approach for ab-initio modeling of molecules with tetrahedral
    and octahedral symmetry" by A. Geva & Y. Shkolnisky.

    :param sym: Symmetry type. One of 'Cn','Dn','T','O','I' where n>1
    :param info: If True, print information about excluded pairs
    :return: List of indices indicating which group elements to use for self common lines
    """
    G = group_elements(sym)

    n = len(G)
    exclude = np.zeros(n, dtype=bool)
    is_identity = np.zeros(n, dtype=bool)

    for i in range(n):
        if np.allclose(G[i], np.eye(3)):
            is_identity[i] = True

    for i in range(n):
        if exclude[i]:
            continue
        for j in range(i + 1, n):
            if np.allclose(G[i], G[j].T):
                exclude[j] = True
                if info:
                    print(f"Transpose pair: Matrix {i} and Matrix {j}")
                    print(f"Matrix {i}:\n{G[i]}")
                    print(f"Matrix {j}:\n{G[j]}\n")
    scl_inds = np.where(~exclude & ~is_identity)[0]
    return scl_inds.tolist()


def is_normal_subgroup(H_name: str, G_name: str, tolerance=1e-10):
    """
    Check if the group H is a normal subgroup of group G.

    For H to be normal in G, we need g*h*g^(-1) ∈ H for all g ∈ G and h ∈ H.

    Parameters:
    H_name: str - name of subgroup, e.g. "D2", "T"
    G_name: str - name of group, e.g. "O", "D8"
    tolerance: float - tolerance for matrix comparison

    Returns:
    tuple: (is_normal, counterexample_info)
    """

    H = group_elements(H_name)
    G = group_elements(G_name)

    def matrix_in_set(matrix, matrix_set, tol=tolerance):
        for m in matrix_set:
            if np.allclose(matrix, m, atol=tol):
                return True
        return False

    # First check H is a subgroup of G (i.e., subset)
    for h_idx, h in enumerate(H):
        if not matrix_in_set(h, G):
            return False, {"h": h, "message": f"h[{h_idx}] is not in G"}
    print(f"{H_name} is a subgroup of {G_name}!")

    # Check each conjugation g*h*g^(-1) is in H
    for g_idx, g in enumerate(G):
        g_inv = g.T

        for h_idx, h in enumerate(H):
            conjugate = g @ h @ g_inv

            if not matrix_in_set(conjugate, H):
                return False, {
                    "g_index": g_idx,
                    "h_index": h_idx,
                    "g": g,
                    "h": h,
                    "conjugate": conjugate,
                    "message": f"g[{g_idx}] * h[{h_idx}] * g[{g_idx}]^(-1) is not in H",
                }
    print(f"{H_name} is a normal subgroup of {G_name}!")
    return True, None

def coset_representatives(sym: str) -> list[np.ndarray]:
    if sym in ["O", "I"]:
        return [np.eye(3)]
    if sym == "D2":
        return [
            np.array(m) for m in [
                [[1,0,0],[0,1,0],[0,0,1]],        # 1
                [[0,0,1],[1,0,0],[0,1,0]],        # ρ
                [[0,1,0],[0,0,1],[1,0,0]],        # ρ²
                [[0,1,0],[1,0,0],[0,0,-1]],       # ε
                [[0,0,-1],[0,1,0],[1,0,0]],       # ρε
                [[1,0,0],[0,0,-1],[0,1,0]],       # ρ²ε
            ]
        ]
    if sym == "T":
        return [
            np.array(m) for m in [
                [[1,0,0], [0,1,0], [0,0,1]],  # 1
                [[0,1,0], [-1,0,0], [0,0,1]], # r
            ]
        ]
    if sym[0] == "D":
        n = int(sym[1:])
        theta = np.pi / n
        c = np.cos(theta)
        s = np.sin(theta)
        return [
            np.array(m) for m in [
                [[1, 0, 0], [0, 1, 0], [0, 0, 1]],  # 1
                [[c, -s, 0], [s, c, 0], [0, 0, 1]], # r (Rotation about z by π/n)
            ]
        ]

    if sym[0] == "C":
        return [
            np.array(m) for m in [
                [[1,0,0], [0,1,0], [0,0,1]],  # 1
                [[1,0,0], [0,-1,0], [0,0,-1]], # r
            ]
        ]
    raise NotImplementedError


def check_j_conjugation_in_group(sym: str, tolerance: float = 1e-10):
    """
    Check for every group element g whether J*g*J is an element of the group,
    where J = diag(1,1,-1).
    
    An element g can be equal to either itself (g = J*g*J) or some other element
    in the group.
    
    :param sym: Symmetry type. One of 'Cn','Dn','T','O','I' where n>1
    :param tolerance: Absolute tolerance for matrix equality comparison
    :return: tuple (is_valid, result)
        - If all elements have matches: (True, pairs_dict)
          where pairs_dict maps each element index to the index of its J-conjugate
        - If any element lacks a match: (False, counterexample_dict)
          where counterexample_dict contains the element index and the computed J*g*J
    """
    sym = supported_symmetry_group(sym)
    G = group_elements(sym)
    n = len(G)
    
    # Define J = diag(1,1,-1)
    J = np.diag([1, 1, -1])
    
    def matrix_in_set(matrix, matrix_set, tol=tolerance):
        """Check if a matrix is in the set (within tolerance)"""
        for idx, m in enumerate(matrix_set):
            if np.allclose(matrix, m, atol=tol):
                return idx
        return None
    
    pairs = {}
    
    # Check each group element g
    for g_idx, g in enumerate(G):
        # Compute J*g*J
        j_g_j = J @ g @ J
        
        # Check if J*g*J is in the group
        match_idx = matrix_in_set(j_g_j, G, tol=tolerance)
        
        if match_idx is None:
            # No match found
            return False, {
                "g_index": g_idx,
                "g": g,
                "J_g_J": j_g_j,
                "message": f"J*g[{g_idx}]*J is not in the group"
            }
        
        pairs[g_idx] = match_idx
    
    return True, pairs

def group_elements_by_axis(sym: str, tolerance: float = 1e-6) -> dict:
    """
    Get all group elements grouped by their rotation axes.
    
    Enforces a canonical axis direction:
    1. Axes are normalized.
    2. Axes are flipped so the first non-zero component is positive.
    3. Angles are adjusted to be in [0, 2*pi) to compensate for axis flips.
       (e.g., -90 degrees becomes 270 degrees).

    :param sym: Symmetry type. One of 'Cn','Dn','T','O','I' where n>1
    :param tolerance: Tolerance for comparing axes and angles
    :return: Dictionary keys are axes (tuples), values are lists of element info.
    """
    sym = supported_symmetry_group(sym) # Assumes this helper exists from your original code
    G = group_elements(sym)             # Assumes this helper exists
    n = len(G)
    
    rotations = R.from_matrix(G)
    rotvecs = rotations.as_rotvec()
    
    grouped = {}

    def get_canonical_axis(v, tol=tolerance):
        """
        Return a standardized axis vector.
        Strategy: First non-zero component must be positive.
        """
        norm = np.linalg.norm(v)
        if norm < tol:
            return np.array([0., 0., 0.]), 1.0 # Identity, sign doesn't matter
        
        v_norm = v / norm
        
        # Find first non-zero component
        sign = 1.0
        for i in range(3):
            if abs(v_norm[i]) > tol:
                if v_norm[i] < 0:
                    sign = -1.0
                break
        
        return v_norm * sign, sign

    for idx in range(n):
        rotvec = rotvecs[idx]
        raw_angle = np.linalg.norm(rotvec)
        
        # --- Handle Identity ---
        if raw_angle < tolerance:
            key = "identity"
            final_axis = np.array([0., 0., 0.])
            final_angle = 0.0
        else:
            # --- Standardize Axis and Angle ---
            raw_axis = rotvec / raw_angle
            canonical_axis, sign = get_canonical_axis(raw_axis)
            
            # Key for dictionary (tuple is hashable)
            key = tuple(np.round(canonical_axis, 5))
            
            final_axis = canonical_axis
            
            # Calculate angle in [0, 2pi)
            # If sign is positive: axis matched canonical, angle is unchanged
            # If sign is negative: axis was flipped, angle becomes 2pi - angle
            if sign > 0:
                final_angle = raw_angle
            else:
                final_angle = (2 * np.pi) - raw_angle

        # --- Store Data ---
        if key not in grouped:
            grouped[key] = []
            
        grouped[key].append({
            'element_index': idx,
            'axis': final_axis,
            'angle_rad': final_angle,
            'angle_deg': np.degrees(final_angle)
        })

    # Sort elements within each axis group by angle for cleaner viewing
    for key in grouped:
        grouped[key].sort(key=lambda x: x['angle_rad'])
        
    return grouped

def print_group_elements_by_axis(sym: str):
    result = group_elements_by_axis(sym)
    print(f"\n{'='*60}")
    print(f"Symmetry Group: {sym}")
    print(f"Unique rotation axes found: {len(result)}")
    print(f"{'='*60}\n")
    
    # FIXED: Sort using tuples to avoid mixing types
    sorted_keys = sorted(
        result.keys(), 
        key=lambda x: (0, (0,0,0)) if x == "identity" else (1, x)
    )

    for axis_key in sorted_keys:
        elements = result[axis_key]
        
        if axis_key == "identity":
            print(f"Axis: Identity")
        else:
            ax = np.array(axis_key)
            
            # Rescale so the smallest non-zero component (in absolute value) equals 1
            abs_ax = np.abs(ax)
            nonzero_mask = abs_ax > 1e-8
            if np.any(nonzero_mask):
                min_nonzero = np.min(abs_ax[nonzero_mask])
                ax_scaled = ax / min_nonzero
            else:
                ax_scaled = ax  # Identity or all-zero axis
            
            print(f"Axis: [{ax_scaled[0]:.4f}, {ax_scaled[1]:.4f}, {ax_scaled[2]:.4f}]")
            
        print(f"  {'Index':<6} | {'Angle (Deg)':<12} | {'Angle (Rad)':<12}")
        print(f"  {'-'*6} + {'-'*12} + {'-'*12}")
        for e in elements:
             idx = e['element_index']
             deg = e['angle_deg']
             rad = e['angle_rad']
             print(f"  {idx:<6d} | {deg:<12.2f} | {rad:<12.4f}")
        print()
    
if __name__ == "__main__":
    SYMMETRY = "I"
    
    print_group_elements_by_axis(SYMMETRY)