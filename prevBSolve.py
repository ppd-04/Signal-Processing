import numpy as np
import matplotlib.pyplot as plt


def sinusoid(
    n: np.ndarray,    # array of integer time indices
    A: float,         # amplitude
    Omega0: float,    # digital frequency
    phi: float        # initial phase
) -> np.ndarray:
    """
    Compute x[n] = A * cos(Omega0 * n + phi)
    
    n is an array like [-20, -19, ..., 20]
    The formula is applied to every element of n at once.
    """
    return A * np.cos(Omega0 * n + phi)


def time_shift_sinusoid(
    n: np.ndarray,    # time axis
    A: float,         # amplitude
    Omega0: float,    # digital frequency
    phi: float,       # initial phase
    n0: int           # how many steps to shift
) -> np.ndarray:
    """
    Compute time-shifted signal: x[n - n0] = A * cos(Omega0*(n - n0) + phi)
    
    Shifting by n0 moves the signal to the RIGHT by n0 steps.
    """
    return A * np.cos(Omega0 * (n - n0) + phi)


def phase_change_sinusoid(
    n: np.ndarray,    # time axis
    A: float,         # amplitude
    Omega0: float,    # digital frequency
    phi: float,       # original phase
    phi0: float       # phase change to add
) -> np.ndarray:
    """
    Compute phase-changed signal: A * cos(Omega0*n + phi + phi0)
    
    Adding phi0 to the phase changes where the wave starts.
    """
    return A * np.cos(Omega0 * n + phi + phi0)


# -----------------------------
# 2) Utility functions (already given)
# -----------------------------
def mse(a: np.ndarray, b: np.ndarray) -> float:
    """Mean squared error between two sequences of equal length."""
    return float(np.mean((a - b) ** 2))


def stem_plot(ax, n, x, label):
    """A nicer stem plot for discrete-time sequences."""
    markerline, stemlines, baseline = ax.stem(n, x, label=label)
    baseline.set_visible(False)
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("n")
    ax.set_ylabel("Amplitude")


# -----------------------------
# 3) Main experiment
# -----------------------------
def main():
    # -----------------------------------------------
    # Signal parameters
    # -----------------------------------------------
    A = 1.0           # amplitude
    Omega0 = np.pi / 4   # digital frequency (pi/4)
    phi = 0.0         # starting phase = 0

    # Integer time indices from -20 to 20
    n = np.arange(-20, 21)

    # -----------------------------------------------
    # Original signal x[n] = cos(pi/4 * n)
    # -----------------------------------------------
    x = sinusoid(n, A, Omega0, phi)

    # -----------------------------------------------
    # PART A: Time shift → Phase change
    # -----------------------------------------------
    # Choose an integer time shift
    n0 = 3

    # Compute time-shifted signal: x[n-3] = cos(pi/4 * (n-3))
    x_time = time_shift_sinusoid(n, A, Omega0, phi, n0)

    # Calculate the equivalent phase change
    # From math: phi0 = -Omega0 * n0
    # cos(Omega0*(n-n0) + phi) = cos(Omega0*n + phi + (-Omega0*n0))
    phi0_equiv = -Omega0 * n0

    # Compute phase-changed signal using that phi0
    x_phase_equiv = phase_change_sinusoid(n, A, Omega0, phi, phi0_equiv)

    # MSE should be nearly 0 (they are mathematically equal)
    err_A = mse(x_time, x_phase_equiv)
    print("[Part A] MSE between time-shifted and equivalent phase-changed:", err_A)

    # Plot Part A
    fig1, ax1 = plt.subplots(figsize=(9, 4))
    stem_plot(ax1, n, x,            "original x[n]")
    stem_plot(ax1, n, x_time,       f"time shift by n0={n0}")
    stem_plot(ax1, n, x_phase_equiv, f"phase change by phi0={phi0_equiv:.3f}")
    ax1.set_title(
        f"Part A: Time shift n0={n0} equals phase change phi0={phi0_equiv:.3f}\n"
        f"MSE = {err_A:.6f}"
    )
    ax1.legend()
    fig1.tight_layout()

    # -----------------------------------------------
    # PART B: Phase change → Time shift?
    # -----------------------------------------------
    # Choose a phase change that does NOT divide evenly
    # phi0 = -Omega0 * n0 → n0 = -phi0/Omega0
    # If phi0 = 1.0, then n0 = -1.0/(pi/4) = -1.27... (NOT integer)
    phi0 = 1.0

    # Compute phase-changed signal
    x_phase = phase_change_sinusoid(n, A, Omega0, phi, phi0)

    # Calculate what the ideal (possibly non-integer) shift would be
    ideal_shift = -phi0 / Omega0
    print(f"[Part B] Ideal (non-integer) shift = {ideal_shift:.4f}")

    # Search over integer shifts to find the closest match
    k_min, k_max = -12, 12
    best_k = None
    best_err = None

    for k in range(k_min, k_max + 1):
        # Try every integer shift from -12 to 12
        x_time_k = time_shift_sinusoid(n, A, Omega0, phi, k)

        # Calculate how different it is from x_phase
        e = mse(x_time_k, x_phase)

        # Keep track of the best (smallest error) shift
        if (best_err is None) or (e < best_err):
            best_err = e
            best_k = k

    print(f"[Part B] Best matching integer shift in [{k_min},{k_max}] is k={best_k} with MSE={best_err}")

    # Best time-shifted signal
    x_time_best = time_shift_sinusoid(n, A, Omega0, phi, best_k)

    # Plot Part B
    fig2, ax2 = plt.subplots(figsize=(9, 4))
    stem_plot(ax2, n, x_phase,     f"phase change by phi0={phi0:.3f}")
    stem_plot(ax2, n, x_time_best, f"best time shift k={best_k}")
    ax2.set_title(
        f"Part B: Phase change phi0={phi0} vs best integer shift k={best_k}\n"
        f"Ideal shift={ideal_shift:.4f}, MSE={best_err:.6f}"
    )
    ax2.legend()
    fig2.tight_layout()

    plt.show()


if __name__ == "__main__":
    main()