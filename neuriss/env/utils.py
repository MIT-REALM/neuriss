import torch
import numpy as np
import scipy.linalg

from matplotlib.patches import Rectangle


def lqr(
        A: np.ndarray,
        B: np.ndarray,
        Q: np.ndarray,
        R: np.ndarray,
        return_eigs: bool = False,
):
    """
    Solve the discrete time lqr controller.
        x_{t+1} = A x_t + B u_t
        cost = sum x.T*Q*x + u.T*R*u

    Code adapted from Mark Wilfred Mueller's continuous LQR code at
    https://www.mwm.im/lqr-controllers-with-python/
    Based on Bertsekas, p.151

    Yields the control law u = -K x
    """

    # first, try to solve the ricatti equation
    X = scipy.linalg.solve_discrete_are(A, B, Q, R)

    # compute the LQR gain
    K = scipy.linalg.inv(B.T @ X @ B + R) @ (B.T @ X @ A)

    if not return_eigs:
        return K
    else:
        eigVals, _ = scipy.linalg.eig(A - B * K)
        return K, eigVals


def continuous_lyap(Acl: np.ndarray, Q: np.ndarray):
    """Solve the continuous time lyapunov equation.

    Acl.T P + P Acl + Q = 0

    using scipy, which expects AP + PA.T = Q, so we need to transpose Acl and negate Q
    """
    P = scipy.linalg.solve_continuous_lyapunov(Acl.T, -Q)
    return P


def plot_rectangle(ax, center: torch.Tensor, heading: float, length: float,
                   width: float, color: str, alpha: float = 1.0):
    center = center.cpu().detach().numpy()
    xy = center - np.array([length / 2 * np.cos(heading) - width / 2 * np.sin(heading),
                            length / 2 * np.sin(heading) + width / 2 * np.cos(heading)])
    car = Rectangle(xy, length, width, angle=np.rad2deg(heading), fill=True, color=color, alpha=alpha)
    ax.add_patch(car)
