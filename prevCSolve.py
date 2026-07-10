import numpy as np
import matplotlib.pyplot as plt

T_MIN, T_MAX, N = -4.0, 4.0, 4001


def x_of_t(t: np.ndarray) -> np.ndarray:
    """
    Base signal x(t).
    """
    tri0 = np.zeros_like(t, dtype=float)
    m0 = np.abs(t) <= 1.0
    tri0[m0] = 1.0 - np.abs(t[m0])

    ramp = np.zeros_like(t, dtype=float)
    m1 = np.abs(t) <= 1.0
    ramp[m1] = t[m1]

    tri_shift = np.zeros_like(t, dtype=float)
    u = t - 1.2
    m2 = np.abs(u) <= 1.0
    tri_shift[m2] = 1.0 - np.abs(u[m2])

    return tri0 + 0.6 * ramp + 0.4 * tri_shift


def time_reverse(x: np.ndarray) -> np.ndarray:
    """
    Given samples x(t), return samples of x(-t).

    The time axis is symmetric: t = [-4, -3.998, ..., 0, ..., 3.998, 4]
    Reversing x means flipping the array left to right.

    Why flipping works:
    - x[0]    is the value at t = -4
    - x[-1]   is the value at t = +4
    - x[2000] is the value at t =  0  (middle)

    After flipping:
    - position 0 gets the value that was at position -1 (t=+4)
    - position -1 gets the value that was at position 0 (t=-4)
    
    So x_reversed[i] = value at t = -t[i]  ✓
    """
    return x[::-1]


def even_odd_decompose(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Decompose x into even and odd parts.

    Formula:
        xe(t) = [x(t) + x(-t)] / 2
        xo(t) = [x(t) - x(-t)] / 2

    We get x(-t) by calling time_reverse(x).
    """
    # Get the time-reversed signal x(-t)
    xr = time_reverse(x)

    # Even part: average of x(t) and x(-t)
    xe = (x + xr) / 2

    # Odd part: half the difference of x(t) and x(-t)
    xo = (x - xr) / 2

    return xe, xo


# ----------------------------
# Provided plotting (do not modify)
# ----------------------------
def plot_three(t: np.ndarray, x: np.ndarray, xe: np.ndarray, xo: np.ndarray):
    plt.figure()
    plt.plot(t, x,  label="x(t)")
    plt.plot(t, xe, label="xe(t)")
    plt.plot(t, xo, label="xo(t)")
    plt.title("Even-Odd Decomposition")
    plt.xlabel("t")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.legend()


def plot_pair(t: np.ndarray, x: np.ndarray, xr: np.ndarray):
    plt.figure()
    plt.plot(t, x,  label="x(t)")
    plt.plot(t, xr, label="x(-t)")
    plt.title("Time Reversal")
    plt.xlabel("t")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.legend()


# ----------------------------
# Main
# ----------------------------
def main():
    # Step 1: Create time axis and compute x(t)
    t = np.linspace(T_MIN, T_MAX, N)
    x = x_of_t(t)

    # Step 2: Compute time reversal x(-t)
    xr = time_reverse(x)

    # Step 3: Compute even and odd parts
    xe, xo = even_odd_decompose(x)

    # Step 4: Plot x(t) and x(-t) together
    plot_pair(t, x, xr)

    # Step 5: Plot x(t), xe(t), xo(t) together
    plot_three(t, x, xe, xo)

    # Show all figures
    plt.show()


if __name__ == "__main__":
    main()