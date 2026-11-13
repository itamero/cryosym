"""
Orientation estimation for molecules with cyclic symmetry C_n, n >= 2.

For C_n the normalizer N(C_n) = O(2) is infinite, so the pairwise estimates of an image do not concentrate on finitely
many rotations and the vote of estimate_rotations does not apply directly. The algorithm here is the same pipeline
(pairwise maximization, vote, spectral synchronization) written in coordinates in which the normalizer acts simply.

Coordinates. Every rotation is written as R = Rz(alpha) Ry(beta) Rz(gamma), alpha, gamma in [0, 2 pi), beta in
[0, pi]. With F = diag(1, -1, -1) (a half turn about e1, an element of O(2) outside SO(2)) and J = diag(1, 1, -1):

    Rz(t) R  = (alpha + t,  beta,      gamma)          rotations of the normalizer
    F R      = (-alpha,     pi - beta, gamma + pi)     half turns of the normalizer
    J R J    = (alpha + pi, beta,      gamma + pi)     handedness

so the normalizer orbit of R, the set O(2) R, is given by (beta, gamma) modulo (beta, gamma) ~ (pi - beta, gamma + pi),
equivalently by the third row of R up to sign. The symmetry group C_n is the subgroup alpha -> alpha + 2 pi k / n.

Algorithm.
  1. Pairwise maximization (pairwise_estimates). The score of a candidate pair (Q_r, Q_s) is invariant to a common
     left multiplication by the normalizer, so only one candidate per orbit of pairs is needed: Q_r = (0, beta_r,
     gamma_r) with beta_r <= pi / 2 and gamma_r < pi, Q_s = (theta, beta_s, gamma_s) with theta in [0, 2 pi / n). This
     is a five dimensional search instead of a six dimensional one.
  2. Vote (vote_orbits). The estimates of image i given by its N - 1 pairs agree on the normalizer orbit of R_i, and
     differ only in alpha. The orbit, folded by J, is a point (beta, gamma) of the grid with beta <= pi / 2 and
     gamma < pi, and is estimated by a majority vote, like the vote for the groups with a finite normalizer. The
     votes in the cells next to the mode are averaged to get below the grid spacing.
  3. Synchronization (synchronize) over (N(C_n) / C_n) x Z_2, with N(C_n) / C_n = O(2) / C_n isomorphic to
     O(2) = SO(2) x| Z_2. As for D_2, where the quotient D_3 = Z_3 x| Z_2 is not abelian, this is done in two stages:
     first two Z_2 synchronizations, of the handedness and of the half turn (the parity homomorphism O(2) -> Z_2),
     and then, with all estimates in the same coset of SO(2) / C_n, a synchronization of the remaining angle
     alpha_i mod 2 pi / n. Since SO(2) / C_n is the circle group through alpha -> n alpha, this is the angular
     synchronization of Singer (2011): the leading eigenvector of H_ij = exp(i n theta_ij), the limit k -> infinity
     of the Z_k synchronization used for the other groups.
"""
import logging
import math

import numpy as np
from joblib import Parallel, delayed
from scipy.linalg import eigh

logger = logging.getLogger(__name__)

J = np.diag([1.0, 1.0, -1.0])
F = np.diag([1.0, -1.0, -1.0])


# ---------------------------------------------------------------------------------------------------------------------
# Rotations
# ---------------------------------------------------------------------------------------------------------------------
def rot_z(a):
    """Rotations about e3 by the angles a (radians): array (..., 3, 3)."""
    a = np.asarray(a, dtype=float)
    out = np.zeros(a.shape + (3, 3))
    c, s = np.cos(a), np.sin(a)
    out[..., 0, 0], out[..., 0, 1], out[..., 1, 0], out[..., 1, 1], out[..., 2, 2] = c, -s, s, c, 1
    return out


def rot_y(b):
    """Rotations about e2 by the angles b (radians): array (..., 3, 3)."""
    b = np.asarray(b, dtype=float)
    out = np.zeros(b.shape + (3, 3))
    c, s = np.cos(b), np.sin(b)
    out[..., 0, 0], out[..., 0, 2], out[..., 2, 0], out[..., 2, 2], out[..., 1, 1] = c, s, -s, c, 1
    return out


def from_euler(alpha, beta, gamma):
    """R = Rz(alpha) Ry(beta) Rz(gamma)."""
    return rot_z(alpha) @ rot_y(beta) @ rot_z(gamma)


def to_euler(R):
    """(alpha, beta, gamma) with R = Rz(alpha) Ry(beta) Rz(gamma), alpha, gamma in [0, 2 pi), beta in [0, pi]."""
    R = np.asarray(R)
    beta = np.arccos(np.clip(R[..., 2, 2], -1, 1))
    alpha = np.arctan2(R[..., 1, 2], R[..., 0, 2]) % (2 * np.pi)
    gamma = np.arctan2(R[..., 2, 1], -R[..., 2, 0]) % (2 * np.pi)
    return alpha, beta, gamma


def cl_angles(rel):
    """
    Common line angles of the relative rotations rel = Q_r^T Q_s, in [0, 2 pi), in the convention of
    cryo_create_rotations_cache.compute_cl_indices.
    """
    a = np.arctan2(-rel[..., 0, 2], rel[..., 1, 2]) + np.pi
    b = np.arctan2(rel[..., 2, 0], -rel[..., 2, 1]) + np.pi
    return a, b


def ray_index(angle, n_theta):
    return np.round(angle / (2 * np.pi) * n_theta).astype(int) % n_theta


# ---------------------------------------------------------------------------------------------------------------------
# Candidate grid
# ---------------------------------------------------------------------------------------------------------------------
class CyclicGrid:
    """
    Candidate set of the pairwise maximization for C_n.

    beta:   n_beta = 180 / step values (j + 1/2) * step, symmetric about 90 degrees; the first half is the range of
            beta_r. Values within eta_a of the poles (top views, where the image itself is C_n symmetric and the
            self common lines are undefined) are not candidates.
    gamma:  rays 0, g, 2g, ... of the polar Fourier transform, g = step in ray units, so that in-plane rotations of a
            candidate are exact shifts of the ray index. gamma_r uses the first half, [0, pi).
    theta:  [0, 2 pi / n) in K steps of about `step` degrees; theta + 2 pi m / n, m = 0..n-1, are the n common lines.

    :param n: Order of the cyclic group
    :param n_theta: Number of rays of the polar Fourier transform
    :param step: Grid spacing in degrees; must divide 180 and be a multiple of 360 / n_theta
    :param eta: Angle in degrees, candidate pairs with viewing directions this close to parallel (modulo C_n) are not
                allowed
    :param eta_a: Angle in degrees around the symmetry axis; candidates whose viewing direction is this close to
                  it are not allowed. Default 2 * step.
    :param eta_p: Angle in degrees; candidate pairs with both viewing directions this close to the plane
                  perpendicular to the symmetry axis are not allowed
    """

    def __init__(self, n, n_theta=360, step=5.0, eta=1.0, eta_a=None, eta_p=22.0):
        self.n, self.n_theta, self.step = n, n_theta, step
        self.eta, self.eta_p = eta, eta_p
        self.eta_a = 2 * step if eta_a is None else eta_a
        ray = 360 / n_theta
        self.g = int(round(step / ray))
        n_beta = int(round(180 / step))
        if abs(self.g * ray - step) > 1e-9 or abs(n_beta * step - 180) > 1e-9 or n_beta % 2 or (n_theta // 2) % self.g:
            raise ValueError(f"step = {step} must divide 180 an even number of times and be a multiple of {ray}")

        beta_deg = (np.arange(n_beta) + 0.5) * step
        keep = np.abs(np.cos(np.radians(beta_deg))) < np.cos(np.radians(self.eta_a))
        self.beta = np.radians(beta_deg[keep])  # symmetric about pi / 2
        self.n_b = len(self.beta)
        self.n_br = self.n_b // 2  # beta_r < pi / 2
        self.gamma_idx = np.arange(0, n_theta, self.g)  # ray indices of gamma
        self.n_g = len(self.gamma_idx)
        self.n_gr = self.n_g // 2  # gamma_r in [0, pi)
        self.gamma = self.gamma_idx * 2 * np.pi / n_theta
        self.K = max(1, int(round(360 / n / step)))
        self.theta = np.arange(self.K) * 2 * np.pi / (n * self.K)

        # Common line ray indices at gamma_r = gamma_s = 0: A, B of shape (n_br, n_b, K, n)
        t = self.theta[:, None] + 2 * np.pi * np.arange(n)[None, :] / n  # (K, n)
        Qr = rot_y(self.beta[: self.n_br])  # (n_br, 3, 3)
        Qs = rot_z(t)[None, :, :] @ rot_y(self.beta)[:, None, None]  # (n_b, K, n, 3, 3)
        rel = np.swapaxes(Qr, -1, -2)[:, None, None, None] @ Qs[None]  # (n_br, n_b, K, n, 3, 3)
        a, b = cl_angles(rel)
        self.A, self.B = ray_index(a, n_theta), ray_index(b, n_theta)

        # Allowed candidates (independent of gamma_r, gamma_s): mask (n_br, n_b, K)
        cos_view = np.abs(np.einsum("ri,sknij,sj->rskn", Qr[:, :, 2], rot_z(t)[None].repeat(self.n_b, 0), rot_y(self.beta)[:, :, 2]))
        mask = np.all(cos_view < np.cos(np.radians(eta)), axis=-1)
        in_band = np.abs(np.cos(self.beta)) <= np.sin(np.radians(eta_p))
        mask &= ~(in_band[: self.n_br, None, None] & in_band[None, :, None])
        self.mask = mask

        # Self common lines at gamma = 0 for one element of each pair {g, g^-1}: (n_b, n_scl)
        ms = np.arange(1, n // 2 + 1)
        Qb = rot_y(self.beta)
        rel = np.swapaxes(Qb, -1, -2)[:, None] @ rot_z(2 * np.pi * ms / n)[None] @ Qb[:, None]
        a, b = cl_angles(rel)
        self.A_self, self.B_self = ray_index(a, n_theta), ray_index(b, n_theta)

        logger.info(
            f"C{n} grid: step {step} deg, {self.n_br} x {self.n_gr} orbits (beta_r, gamma_r), {self.n_b} x {self.n_g} "
            f"(beta_s, gamma_s), {self.K} theta; {self.n_candidates:,} candidate pairs, {mask.mean():.1%} allowed"
        )

    @property
    def n_candidates(self):
        return self.n_br * self.n_gr * self.n_b * self.n_g * self.K

    def self_scores(self, self_corr):
        """
        Self common lines score of every candidate (beta, gamma) for one image.

        :param self_corr: (n_theta, n_theta) correlations between the rays of the image
        :return: (n_b, n_g)
        """
        ii = (self.A_self[:, None, :] - self.gamma_idx[None, :, None]) % self.n_theta
        jj = (self.B_self[:, None, :] - self.gamma_idx[None, :, None]) % self.n_theta
        return np.prod(np.maximum(self_corr[ii, jj], 0), axis=-1)

    def pair_scores(self, corr):
        """
        Product of the n common line correlations of every candidate pair.

        :param corr: (n_theta, n_theta) correlations between the rays of images i and j
        :return: (n_br, n_gr, n_b, n_g, K)
        """
        n_theta = self.n_theta
        gr = self.gamma_idx[: self.n_gr]
        gs = self.gamma_idx
        flat = corr.ravel()
        out = np.empty((self.n_br, self.n_gr, self.n_b, self.n_g, self.K), dtype=corr.dtype)
        for r in range(self.n_br):
            ii = (self.A[r][None, :, None, :, :] - gr[:, None, None, None, None]) % n_theta  # (n_gr, n_b, 1, K, n)
            jj = (self.B[r][None, :, None, :, :] - gs[None, None, :, None, None]) % n_theta  # (1, n_b, n_g, K, n)
            vals = flat[ii * n_theta + jj]  # (n_gr, n_b, n_g, K, n)
            np.maximum(vals, 0, out=vals)
            out[r] = np.prod(vals, axis=-1)
        return out


# ---------------------------------------------------------------------------------------------------------------------
# Step 1: pairwise maximization
# ---------------------------------------------------------------------------------------------------------------------
def _pair_argmax(grid, pf_i, pf_j, s_i, s_j):
    corr = np.real(pf_i.conj() @ pf_j.T)
    score = grid.pair_scores(corr)
    score *= s_i[: grid.n_br, : grid.n_gr, None, None, None]
    score *= s_j[None, None, :, :, None]
    score[~np.broadcast_to(grid.mask[:, None, :, None, :], score.shape)] = -1
    best = np.unravel_index(np.argmax(score), score.shape)
    return best + (float(score[best]),)


def pairwise_estimates(pf, grid, n_jobs=-1):
    """
    For every pair of images i < j, the candidate pair maximizing the score.

    :param pf: (n_images, n_theta, n_r) polar Fourier transforms with normalized rays
    :return: est (n_pairs, 5) grid indices (beta_r, gamma_r, beta_s, gamma_s, theta), scores (n_pairs,),
             pairs (n_pairs, 2), self common line scores (n_images, n_b, n_g)
    """
    n_images = len(pf)
    s_self = np.stack([grid.self_scores(np.real(p.conj() @ p.T)) for p in pf])
    pairs = [(i, j) for i in range(n_images) for j in range(i + 1, n_images)]
    res = Parallel(n_jobs=n_jobs)(
        delayed(_pair_argmax)(grid, pf[i], pf[j], s_self[i], s_self[j]) for i, j in pairs
    )
    res = np.array(res)
    return res[:, :5].astype(int), res[:, 5], np.array(pairs), s_self


# ---------------------------------------------------------------------------------------------------------------------
# Step 2: vote on the normalizer orbit
# ---------------------------------------------------------------------------------------------------------------------
def fold(grid, b_idx, g_idx):
    """Grid indices of the representative with beta < pi / 2 and gamma < pi of the orbit of (beta, gamma) under F, J."""
    b_idx, g_idx = np.asarray(b_idx), np.asarray(g_idx)
    return np.minimum(b_idx, grid.n_b - 1 - b_idx), g_idx % grid.n_gr


def vote_orbits(est, pairs, n_images, grid, refine=True):
    """
    Majority vote of the normalizer orbit (beta, gamma) of every image over its N - 1 pairwise estimates.

    :return: beta (n_images,), gamma (n_images,) in radians with beta <= pi / 2 and gamma in [0, pi), and the number
             of votes of the winning cell
    """
    hist = np.zeros((n_images, grid.n_br, grid.n_gr))
    bi, gi = fold(grid, est[:, 0], est[:, 1])
    np.add.at(hist, (pairs[:, 0], bi, gi), 1)
    bj, gj = fold(grid, est[:, 2], est[:, 3])
    np.add.at(hist, (pairs[:, 1], bj, gj), 1)

    beta, gamma, count = np.zeros(n_images), np.zeros(n_images), np.zeros(n_images, dtype=int)
    step = np.radians(grid.step)
    for i in range(n_images):
        b0, g0 = np.unravel_index(np.argmax(hist[i]), hist[i].shape)
        count[i] = hist[i, b0, g0]
        beta[i], gamma[i] = grid.beta[b0], grid.gamma[g0]
        if refine:  # centroid of the votes in the 3 x 3 cells around the mode
            db, dg = np.meshgrid([-1, 0, 1], [-1, 0, 1], indexing="ij")
            bb, gg = b0 + db, (g0 + dg) % grid.n_gr
            # cells beyond beta = pi / 2 are the reflected cells (pi - beta, gamma + pi) ~ (pi - beta, gamma)
            over = bb >= grid.n_br
            bb = np.where(over, 2 * grid.n_br - 1 - bb, bb)
            ok = bb >= 0
            w = np.where(ok, hist[i, np.clip(bb, 0, None), gg], 0)
            if not np.any(over & (w > 0)):  # the reflection is not a translation of the cell; skip the rare case
                beta[i] += step * np.sum(w * db) / np.sum(w)
                gamma[i] = (gamma[i] + step * np.sum(w * dg) / np.sum(w)) % np.pi
    return beta, gamma, count


# ---------------------------------------------------------------------------------------------------------------------
# Step 3: synchronization over (O(2) / C_n) x Z_2
# ---------------------------------------------------------------------------------------------------------------------
def theta_profile(Qi, Qj, corr, n, n_fine):
    """
    Score of the pairs (Qi, Rz(theta) Qj), theta = 2 pi k / (n * n_fine), k = 0..n_fine-1 (the common lines only).

    :param Qj: (..., 3, 3)
    :return: (..., n_fine)
    """
    n_theta = corr.shape[0]
    t = 2 * np.pi * np.arange(n * n_fine) / (n * n_fine)
    rel = Qi.T @ rot_z(t) @ np.asarray(Qj)[..., None, :, :]  # (..., n * n_fine, 3, 3)
    a, b = cl_angles(rel)
    vals = np.maximum(corr[ray_index(a, n_theta), ray_index(b, n_theta)], 0)
    return np.prod(vals.reshape(vals.shape[:-1] + (n, n_fine)), axis=-2)


def sync_z2(C):
    """Z_2 synchronization: signs of the leading eigenvector of the symmetric +-1 matrix C."""
    w, v = eigh(C)
    s = np.sign(v[:, -1])
    s[s == 0] = 1
    return s.astype(int), w[::-1][:3]


def sync_angle(H):
    """Angular synchronization: phases of the leading eigenvector of the Hermitian matrix H."""
    w, v = eigh(H)
    return np.angle(v[:, -1]), w[::-1][:3]


def synchronize(pf, beta, gamma, n, n_fine=None, weighted=False):
    """
    Resolve the element of (O(2) / C_n) x Z_2 left by the vote for each image, up to a common element.

    :param beta, gamma: Voted orbits (radians)
    :return: Rotations (n_images, 3, 3) and a dict of diagnostics
    """
    n_images, n_theta = pf.shape[0], pf.shape[1]
    n_fine = math.ceil(n_theta / n) if n_fine is None else n_fine
    Q = from_euler(0, beta, gamma)

    # Stage 1: handedness and half turn. The four variants of Q_j: 1, J, F, FJ
    def variants(q):
        return np.stack([q, J @ q @ J, F @ q, F @ J @ q @ J])

    corrs = {}
    C_J, C_F = np.ones((n_images, n_images)), np.ones((n_images, n_images))
    for i in range(n_images):
        for j in range(i + 1, n_images):
            corrs[i, j] = np.real(pf[i].conj() @ pf[j].T)
            best = np.argmax(theta_profile(Q[i], variants(Q[j]), corrs[i, j], n, n_fine).max(axis=-1))
            C_J[i, j] = C_J[j, i] = -1 if best in (1, 3) else 1
            C_F[i, j] = C_F[j, i] = -1 if best in (2, 3) else 1
    s_J, ev_J = sync_z2(C_J)
    s_F, ev_F = sync_z2(C_F)
    for i in range(n_images):
        if s_J[i] < 0:
            Q[i] = J @ Q[i] @ J
        if s_F[i] < 0:
            Q[i] = F @ Q[i]

    # Stage 2: the angle alpha_i mod 2 pi / n. For R_i = Rz(alpha_i) Q_i the maximizer is theta_ij = alpha_j - alpha_i
    H = np.eye(n_images, dtype=complex)
    for i in range(n_images):
        for j in range(i + 1, n_images):
            prof = theta_profile(Q[i], Q[j], corrs[i, j], n, n_fine)
            k = np.argmax(prof)
            w = prof[k] if weighted else 1.0
            H[i, j] = w * np.exp(1j * 2 * np.pi * k / n_fine)  # exp(i n theta_ij)
            H[j, i] = np.conj(H[i, j])
    phase, ev_A = sync_angle(H)  # leading eigenvector is exp(-i n alpha_i) up to a common phase
    alpha = -phase / n
    rots = rot_z(alpha) @ Q
    info = {"s_J": s_J, "s_F": s_F, "alpha": alpha, "ev_J": ev_J, "ev_F": ev_F, "ev_alpha": ev_A, "H": H}
    return rots, info


# ---------------------------------------------------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------------------------------------------------
def polar_ft(images, n_theta=360, n_r_perc=50, radial_power=0.0):
    """
    Polar Fourier transform with normalized rays, as in ab_initio_sym: (n_images, n_theta, n_r).

    :param radial_power: The sample of every ray at radial frequency xi is multiplied by xi^radial_power before the
                         normalization. With 0 (as in ab_initio_sym) the rays are dominated by their low frequencies
                         and unrelated rays of clean images have correlations of 0.85-0.95; with 1 they drop to
                         0.3-0.6 while the common lines stay above 0.99.
    """
    from aspire.operators import PolarFT

    img_size = images.shape[-1]
    pft = PolarFT(img_size, nrad=math.ceil(img_size * n_r_perc / 100), ntheta=n_theta)
    pf = pft.transform(images)
    pf[..., 0] = 0
    if radial_power:
        pf = pf * np.arange(pf.shape[-1], dtype=pf.real.dtype) ** radial_power
    pf /= np.linalg.norm(pf, axis=2, ord=2)[..., np.newaxis]
    return PolarFT.half_to_full(pf)


def estimate_rotations_cn(pf, n, step=5.0, eta=1.0, eta_a=None, eta_p=22.0, n_jobs=-1, refine_vote=True):
    """
    Estimate the orientations of images of a C_n symmetric molecule, up to the symmetry, normalizer and handedness
    ambiguities.

    :param pf: (n_images, n_theta, n_r) polar Fourier transforms (polar_ft)
    :return: Rotations (n_images, 3, 3) and a dict of the intermediate results
    """
    n_images, n_theta = pf.shape[0], pf.shape[1]
    grid = CyclicGrid(n, n_theta, step, eta, eta_a, eta_p)
    est, scores, pairs, s_self = pairwise_estimates(pf, grid, n_jobs)
    beta, gamma, count = vote_orbits(est, pairs, n_images, grid, refine=refine_vote)
    rots, info = synchronize(pf, beta, gamma, n)
    info.update(grid=grid, est=est, scores=scores, pairs=pairs, beta=beta, gamma=gamma, count=count, s_self=s_self)
    return rots, info


# ---------------------------------------------------------------------------------------------------------------------
# Evaluation against the ground truth
# ---------------------------------------------------------------------------------------------------------------------
def register_cn(rots, rots_gt, n):
    """
    Register estimated rotations to the ground truth over the ambiguities of C_n: a global element of
    O(2) x Z_2 (rotation about e3 by phi, half turn F, handedness J) and an element of C_n for each image.

    :return: Registered rotations (n_images, 3, 3), per image errors in degrees, dict with the global element
    """
    best = None
    for f in (0, 1):
        for d in (0, 1):
            X = np.array(rots)
            if d:
                X = J @ X @ J
            if f:
                X = F @ X
            M = X @ np.swapaxes(rots_gt, -1, -2)
            # the rotation about e3 by psi maximizing trace(Rz(psi) M)
            psi = np.arctan2(M[:, 0, 1] - M[:, 1, 0], M[:, 0, 0] + M[:, 1, 1])
            phi = np.angle(np.mean(np.exp(1j * n * psi))) / n
            for _ in range(3):  # reassign the C_n elements and average
                k = np.round((psi - phi) * n / (2 * np.pi))
                phi = phi + np.angle(np.mean(np.exp(1j * (psi - phi - 2 * np.pi * k / n))))
            k = np.round((psi - phi) * n / (2 * np.pi))
            Y = rot_z(phi + 2 * np.pi * k / n) @ X
            cos = (np.trace(Y @ np.swapaxes(rots_gt, -1, -2), axis1=-2, axis2=-1) - 1) / 2
            err = np.degrees(np.arccos(np.clip(cos, -1, 1)))
            # robust variant: phi from the median image would not change the minimizer when most images are right
            if best is None or np.median(err) < np.median(best[1]):
                best = (Y, err, {"flip": f, "J": d, "phi": phi})
    return best


def orbit_error(beta, gamma, rots_gt):
    """Angle in degrees between the voted third rows and the true ones, up to sign and J (per image)."""
    v = np.stack([-np.sin(beta) * np.cos(gamma), np.sin(beta) * np.sin(gamma), np.cos(beta)], axis=-1)
    v_gt = rots_gt[:, 2, :]
    c1 = np.abs(np.sum(v * v_gt, axis=-1))
    c2 = np.abs(np.sum(v * (v_gt * np.array([1, 1, -1])), axis=-1))
    return np.degrees(np.arccos(np.clip(np.maximum(c1, c2), -1, 1)))


def true_candidate(grid, Ri, Rj):
    """
    The candidate (beta_r, gamma_r, beta_s, gamma_s, theta) of the grid's fundamental domain equivalent to the true
    pair (Ri, Rj), as continuous angles and as the nearest grid indices.
    """
    n = grid.n
    ai, bi, gi = to_euler(Ri)
    aj, bj, gj = to_euler(Rj)
    theta = aj - ai
    if bi > np.pi / 2:  # F
        bi, gi, bj, gj, theta = np.pi - bi, gi + np.pi, np.pi - bj, gj + np.pi, -theta
    if gi % (2 * np.pi) >= np.pi:  # J
        gi, gj = gi + np.pi, gj + np.pi
    gi, gj, theta = gi % (2 * np.pi), gj % (2 * np.pi), theta % (2 * np.pi / n)
    cont = np.array([bi, gi, bj, gj, theta])
    idx = (
        int(np.argmin(np.abs(grid.beta[: grid.n_br] - bi))),
        int(np.round(gi / np.radians(grid.step))) % grid.n_gr,
        int(np.argmin(np.abs(grid.beta - bj))),
        int(np.round(gj / np.radians(grid.step))) % grid.n_g,
        int(np.round(theta / (2 * np.pi / (n * grid.K)))) % grid.K,
    )
    return cont, idx
