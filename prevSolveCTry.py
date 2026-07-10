import numpy as np
import matplotlib.pyplot as plt

T_MIN, T_MAX, N = -4.0, 4.0, 4001

def x_of_t(t: np.ndarray) -> np.ndarray:
    """
    Base signal x(t).
    """
    # Combination of components (can be replaced)
    # 1) Triangular pulse centered at 0
    tri0 = np.zeros_like(t, dtype=float)
    m0 = np.abs(t) <= 1.0
    tri0[m0] = 1.0 - np.abs(t[m0])

    # 2) Windowed ramp (odd-ish component)
    ramp = np.zeros_like(t, dtype=float)
    m1 = np.abs(t) <= 1.0
    ramp[m1] = t[m1]

    # 3) Shifted triangular pulse (breaks symmetry)
    tri_shift = np.zeros_like(t, dtype=float)
    u = t - 1.2
    m2 = np.abs(u) <= 1.0
    tri_shift[m2] = 1.0 - np.abs(u[m2])

    return tri0 + 0.6 * ramp + 0.4 * tri_shift


def time_reverse(x: np.ndarray) -> np.ndarray:
    """
    Given samples x(t), return samples of x(-t)
    """
    # raise NotImplementedError
    return x[::-1]

def time_reverse_ifnotSymmetric(t: np.ndarray, x: np.ndarray) -> np.ndarray:
    """
    Compute x(-t) for ANY time array, symmetric or not.
    
    For each time point t[i], we need the value x had at -t[i].
    We find this by looking up -t[i] in the original t array.
    """

    # Step 1: Create output array filled with zeros
    # Same size as x, starts as all zeros
    result = np.zeros_like(x, dtype=float)

    # Step 2: Compute the spacing between time samples
    # (assumes equally spaced time array)
    dt = t[1] - t[0]

    # Step 3: For each position, find x at -t[i]
    for i in range(len(t)):

        # The time we need to look up
        t_needed = -t[i]

        # Check if t_needed is within our original time range
        if t_needed < t[0] or t_needed > t[-1]:
            # This time does not exist in our data
            # Leave it as 0 (or you could use NaN)
            result[i] = 0.0
            continue

        # Find the closest index in t that matches t_needed
        # (t_needed - t[0]) / dt gives the exact position
        idx = round((t_needed - t[0]) / dt)

        # Safety clamp: make sure idx is within array bounds
        idx = int(np.clip(idx, 0, len(t) - 1))

        # Copy the value from that position
        result[i] = x[idx]

    return result




def even_odd_decompose(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Must call time_reverse(...) inside this function.
    """
    # raise NotImplementedError
    x_rev = time_reverse(x)
    x_even = 0.5*(x+x_rev)
    x_odd = 0.5*(x-x_rev)
    return x_even, x_odd


# ----------------------------
# Provided plotting (do not modify)
# ----------------------------
def plot_three(t: np.ndarray, x: np.ndarray, xe: np.ndarray, xo: np.ndarray):
    plt.figure()
    plt.plot(t, x, label="x(t)")
    plt.plot(t, xe, label="xe(t)")
    plt.plot(t, xo, label="xo(t)")
    plt.title("Even–Odd Decomposition")
    plt.xlabel("t")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.legend()


def plot_pair(t: np.ndarray, x: np.ndarray, xr: np.ndarray):
    plt.figure()
    plt.plot(t, x, label="x(t)")
    plt.plot(t, xr, label="x(-t)")
    plt.title("Time Reversal")
    plt.xlabel("t")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.legend()


# ----------------------------
# Main (provided)
# ----------------------------
def main():
    t = np.linspace(T_MIN, T_MAX, N)
    # x = None
    x = x_of_t(t)

    # Compute time reverse and even odd components
    xr = time_reverse(x)
    xe, xo = even_odd_decompose(x)

    plot_pair(t,x,xr)

    plot_three(t, x, xe, xo)

    plt.show()
    # Plot x(t), x(-t), xe(t) and xo(t) using the previously defined functions


if __name__ == "__main__":
    main()