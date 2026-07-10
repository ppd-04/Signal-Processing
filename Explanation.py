# ============================================================
# COMPLETE SIGNAL PROCESSING REFERENCE
# Covers: Time scaling, Time reversal, Even-Odd decomposition,
#         Phase change, Time shift, Interpolation
# ============================================================

import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# SECTION 0: THE MOST IMPORTANT CONCEPT
# ============================================================
# A signal is just a TABLE with two columns:
#
#   Time (t or n)  |  Value (x)
#   ---------------|----------
#        -2        |   0.5
#        -1        |   2.0
#         0        |   1.0
#         1        |   0.5
#         2        |   1.0
#
# t or n = WHERE we are measuring (the ruler, NEVER changes)
# x      = WHAT the signal value is at each time
# y      = the NEW signal after doing some operation on x
#
# When we do any operation:
#   - The time axis (t or n) STAYS THE SAME
#   - The values (x) change to produce y
# ============================================================


# ============================================================
# SECTION 1: CONTINUOUS TIME SIGNALS
# Signal lives on a float time axis like -4.0 to 4.0
# ============================================================

T_MIN = -4.0
T_MAX =  4.0
N     = 4001   # number of samples

def make_time_axis():
    """
    Create the time axis.
    np.linspace(start, stop, num) creates 'num' equally spaced
    values from start to stop.

    Result: [-4.0, -3.998, -3.996, ..., 0.0, ..., 3.998, 4.0]
    """
    return np.linspace(T_MIN, T_MAX, N)


def continuous_signal(t):
    """
    A sample continuous-time signal x(t).
    NumPy applies the formula to EVERY element of t at once.

    np.sin(arr)  → sine of every element
    np.pi        → the value of π
    """
    return (
        np.sin(2 * np.pi * 0.5 * t)       # slow sine wave
        + 0.5 * np.sin(2 * np.pi * 1.5 * t)  # faster sine wave, smaller
    )


# ============================================================
# SECTION 2: INTERPOLATION (for continuous signals)
# ============================================================
# When we need x(0.5) but only have x(0) and x(1),
# we ESTIMATE it as the average:
#   x(0.5) ≈ (x(0) + x(1)) / 2
# ============================================================

def interpolate(t_original, x_original, t_query):
    """
    Given the original (t, x) table, estimate x at new time points t_query.

    Parameters:
        t_original : the original time axis (sorted, equally spaced)
        x_original : signal values at those times
        t_query    : the new time points we want values for

    Returns:
        result : estimated values at t_query points
    """

    # Create output array filled with zeros
    # np.zeros_like creates an array of zeros with same size as t_query
    result = np.zeros_like(t_query, dtype=float)

    # Time spacing between samples
    # t_original[1] - t_original[0] = distance between two neighbors
    dt = t_original[1] - t_original[0]

    # Loop through every requested time point
    # enumerate gives both index i and value tq
    for i, tq in enumerate(t_query):

        # Skip if outside the original time range
        # t_original[0]  = first time value (e.g., -4.0)
        # t_original[-1] = last time value  (e.g., +4.0)
        if tq < t_original[0] or tq > t_original[-1]:
            result[i] = 0.0
            continue   # go to next iteration

        # Find the fractional position of tq in the array
        # Example: t_original starts at -4, dt=0.002, tq=-3.001
        # position = (-3.001 - (-4)) / 0.002 = 499.5
        position = (tq - t_original[0]) / dt

        # Check if tq is exactly on an existing sample
        # np.isclose handles floating point rounding errors
        nearest = int(round(position))
        if np.isclose(position, nearest):
            # Exact sample exists, use it directly
            result[i] = x_original[nearest]
        else:
            # Between two samples, use average of neighbors
            left  = int(np.floor(position))   # round DOWN
            right = left + 1                  # next sample

            # Safety check: right must not go beyond array
            if right >= len(x_original):
                result[i] = x_original[left]
            else:
                # Average the left and right values
                result[i] = 0.5 * (x_original[left] + x_original[right])

    return result


# ============================================================
# SECTION 3: TIME SCALING (continuous)
# y(t) = x(t/k)  → signal gets STRETCHED by factor k
# ============================================================
# For k=2:
#   y(0) = x(0/2) = x(0)    ← same
#   y(2) = x(2/2) = x(1)    ← value from t=1 appears at t=2
#   y(1) = x(1/2) = x(0.5)  ← needs interpolation
# The signal becomes WIDER / SLOWER
# ============================================================

def time_scale_continuous(t, x, k):
    """
    Compute y(t) = x(t/k) for a continuous sampled signal.

    Steps:
    1. For each output time t, compute t/k (the input time needed)
    2. If t/k is outside original range, y=0
    3. Otherwise, interpolate to find x(t/k)
    """

    # t / k divides EVERY element of t by k (element-wise)
    # Example: t=[-4,-3,-2,-1,0,1,2,3,4], k=2
    # t_query = [-2,-1.5,-1,-0.5,0,0.5,1,1.5,2]
    t_query = t / k

    # Create output filled with zeros
    y = np.zeros_like(t, dtype=float)

    # Find which query points are within the valid range
    # (t_query >= T_MIN) creates a True/False array
    # & means AND: both conditions must be True
    valid = (t_query >= T_MIN) & (t_query <= T_MAX)

    # Only interpolate for valid points
    # t_query[valid] keeps only the True positions
    y[valid] = interpolate(t, x, t_query[valid])

    return y


# ============================================================
# SECTION 4: TIME REVERSAL (continuous)
# y(t) = x(-t) → signal gets MIRRORED around t=0
# ============================================================
# The time axis t is symmetric: [-4,...,0,...,4]
# Flipping the values array gives us x(-t)
#
# Why?
#   Original:  position 0 → t=-4, value A
#              position N → t=+4, value B
#   Flipped:   position 0 → t=-4, value B  (was at t=+4)
#              position N → t=+4, value A  (was at t=-4)
# This is exactly x(-t)!
# ============================================================

def time_reverse_continuous(t, x):
    """
    Compute x(-t) for a continuous signal.

    If t is symmetric around 0 (like -4 to +4):
        Just flip the array: x[::-1]

    If t is NOT symmetric:
        Must look up each -t[i] in the original t array.
    """

    # Check if t is symmetric
    # A symmetric t has t[0] = -t[-1]
    if np.isclose(t[0], -t[-1]):
        # SYMMETRIC CASE: simple flip
        # x[::-1] reverses the array
        # [1,2,3,4,5][::-1] = [5,4,3,2,1]
        return x[::-1]

    else:
        # NON-SYMMETRIC CASE: look up each -t[i]
        result = np.zeros_like(x, dtype=float)
        dt = t[1] - t[0]

        for i in range(len(t)):
            # We need x at time -t[i]
            t_needed = -t[i]

            # Check if t_needed is in range
            if t_needed < t[0] or t_needed > t[-1]:
                result[i] = 0.0
                continue

            # Find array position of t_needed
            # (t_needed - t[0]) / dt gives fractional position
            idx = int(round((t_needed - t[0]) / dt))

            # np.clip keeps idx within [0, len(t)-1]
            idx = int(np.clip(idx, 0, len(t) - 1))

            result[i] = x[idx]

        return result


# ============================================================
# SECTION 5: EVEN-ODD DECOMPOSITION (continuous)
# Every signal = even part + odd part
#
# xe(t) = [x(t) + x(-t)] / 2   ← symmetric (mirror image = same)
# xo(t) = [x(t) - x(-t)] / 2   ← anti-symmetric (mirror = negative)
#
# Check: xe(t) + xo(t) = x(t)  always true
# ============================================================

def even_odd_continuous(t, x):
    """
    Decompose x(t) into even and odd parts.
    Returns (xe, xo)
    """

    # Get x(-t) using time reversal
    x_rev = time_reverse_continuous(t, x)

    # Element-wise operations on arrays:
    # (x + x_rev) adds corresponding elements
    # / 2 divides every element by 2
    xe = (x + x_rev) / 2   # even part
    xo = (x - x_rev) / 2   # odd part

    return xe, xo


# ============================================================
# SECTION 6: TIME SHIFT (discrete sinusoid)
# y[n] = x[n - n0]
# Shifts the signal RIGHT by n0 steps (for positive n0)
# ============================================================
# This appears in the sinusoid/phase question.
# Formula: A*cos(Ω₀*(n-n0) + φ)
# ============================================================

def time_shift_sinusoid(n, A, Omega0, phi, n0):
    """
    Compute time-shifted sinusoid: y[n] = A*cos(Ω₀*(n-n0) + φ)

    Parameters:
        n      : array of integer time indices
        A      : amplitude
        Omega0 : digital frequency
        phi    : initial phase
        n0     : integer time shift (positive = shift right)
    """
    # (n - n0) subtracts n0 from every element of n
    # np.cos computes cosine of every element
    return A * np.cos(Omega0 * (n - n0) + phi)


# ============================================================
# SECTION 7: PHASE CHANGE (discrete sinusoid)
# y[n] = A*cos(Ω₀*n + φ + φ₀)
# Adds φ₀ to the phase → changes starting position of wave
# ============================================================
# KEY RELATIONSHIP:
#   Time shift n0  ↔  Phase change φ₀ = -Ω₀ * n0
#
#   cos(Ω₀*(n-n0) + φ) = cos(Ω₀*n + φ + (-Ω₀*n0))
#
# Part A: Given n0, find φ₀ → ALWAYS possible, φ₀ = -Ω₀*n0
# Part B: Given φ₀, find n0 → n0 = -φ₀/Ω₀ (may NOT be integer)
# ============================================================

def phase_change_sinusoid(n, A, Omega0, phi, phi0):
    """
    Compute phase-changed sinusoid: y[n] = A*cos(Ω₀*n + φ + φ₀)
    """
    return A * np.cos(Omega0 * n + phi + phi0)


def sinusoid(n, A, Omega0, phi):
    """Base sinusoid: x[n] = A*cos(Ω₀*n + φ)"""
    return A * np.cos(Omega0 * n + phi)


# ============================================================
# SECTION 8: MSE (Mean Squared Error)
# Measures how DIFFERENT two signals are
# MSE = 0 means identical
# MSE > 0 means different
# ============================================================

def mse(a, b):
    """
    Mean Squared Error between two signals.

    Formula: (1/N) * sum((a[i] - b[i])^2)

    np.mean computes the average
    (a - b) subtracts element by element
    ** 2 squares every element
    """
    return float(np.mean((a - b) ** 2))


# ============================================================
# SECTION 9: DISCRETE SIGNAL (array representation)
# Signal lives on n = -8 to n = +8
# Array has 17 elements
# position = n + INF  (INF = 8)
# n        = position - INF
# ============================================================

INF = 8   # signal range is -INF to +INF

def init_discrete_signal():
    """
    Create a blank discrete signal (all zeros).
    Size = 2*INF+1 = 17 elements
    Represents n = -8 to n = +8
    """
    # np.zeros(n) creates array of n zeros
    return np.zeros(2 * INF + 1)


def n_to_pos(n):
    """Convert time index n to array position"""
    return n + INF   # n=0 → position 8, n=-3 → position 5


def pos_to_n(pos):
    """Convert array position to time index n"""
    return pos - INF  # position 0 → n=-8, position 8 → n=0


# ============================================================
# SECTION 10: DISCRETE TIME REVERSAL
# y[n] = x[-n]  → flip the array
# ============================================================

def time_reverse_discrete(x):
    """
    Compute x[-n] for discrete signal.

    Since array represents n = -8 to +8 (symmetric),
    flipping the array gives x[-n].

    x[::-1] reverses the array.
    """
    return x[::-1]

    # LOOP VERSION (same result, easier to understand):
    # result = np.zeros_like(x)
    # for p_out in range(2 * INF + 1):
    #     n = pos_to_n(p_out)        # output time index
    #     p_in = n_to_pos(-n)        # position of -n in input
    #     if 0 <= p_in <= 2 * INF:   # check within range
    #         result[p_out] = x[p_in]
    # return result


# ============================================================
# SECTION 11: DISCRETE EVEN-ODD DECOMPOSITION
# Same formulas as continuous, but with arrays
# ============================================================

def even_odd_discrete(x):
    """
    Decompose discrete signal x[n] into even and odd parts.

    xe[n] = (x[n] + x[-n]) / 2
    xo[n] = (x[n] - x[-n]) / 2

    Returns (xe, xo)
    """
    x_rev = time_reverse_discrete(x)   # get x[-n]
    xe = (x + x_rev) / 2               # even part
    xo = (x - x_rev) / 2               # odd part
    return xe, xo


# ============================================================
# SECTION 12: DISCRETE TIME SCALING
# y[n] = x[n/k]  → signal gets STRETCHED by k
# ============================================================
# n/k must be an INTEGER for discrete signals.
# If n/k is not integer:
#   Task 1: set to 0
#   Task 2: interpolate (average neighbors)
# ============================================================

def time_scale_discrete(x, k):
    """
    Compute y[n] = x[n/k], intermediate samples = 0.

    For each output position p_out:
        n = p_out - INF           (output time index)
        if n % k == 0:            (n/k is integer)
            y[p_out] = x[n/k]
        else:
            y[p_out] = 0
    """
    y = np.zeros(2 * INF + 1)

    for p_out in range(2 * INF + 1):
        n = pos_to_n(p_out)   # output time index

        # n % k gives remainder: if 0, n divides evenly by k
        # Example: 6 % 3 = 0 (evenly divisible)
        #          5 % 3 = 2 (not evenly divisible)
        if n % k == 0:
            n_over_k = n // k   # integer division: 6//3 = 2
            p_in = n_to_pos(n_over_k)

            # Check if p_in is within valid array range
            if 0 <= p_in <= 2 * INF:
                y[p_out] = x[p_in]

    return y


def time_scale_discrete_interpolate(x, k):
    """
    Compute y[n] = x[n/k], intermediate samples = average of neighbors.

    For non-integer n/k:
        left  = floor(n/k)  → nearest original sample on left
        right = ceil(n/k)   → nearest original sample on right
        y[n]  = (x[left] + x[right]) / 2
    """
    y = np.zeros(2 * INF + 1)

    for p_out in range(2 * INF + 1):
        n = pos_to_n(p_out)

        if n % k == 0:
            # Exact sample: same as Task 1
            n_over_k = n // k
            p_in = n_to_pos(n_over_k)
            if 0 <= p_in <= 2 * INF:
                y[p_out] = x[p_in]
        else:
            # Between samples: interpolate
            n_over_k_decimal = n / k   # exact decimal value

            # np.floor always rounds DOWN
            # np.ceil  always rounds UP
            left_n  = int(np.floor(n_over_k_decimal))
            right_n = int(np.ceil(n_over_k_decimal))

            # Get values (use 0 if outside range)
            if -INF <= left_n <= INF:
                left_val = x[n_to_pos(left_n)]
            else:
                left_val = 0.0

            if -INF <= right_n <= INF:
                right_val = x[n_to_pos(right_n)]
            else:
                right_val = 0.0

            y[p_out] = (left_val + right_val) / 2

    return y


# ============================================================
# SECTION 13: PLOTTING FUNCTIONS
# ============================================================

def plot_continuous(t, signals_dict, title):
    """
    Plot one or more continuous signals on the same figure.

    signals_dict = {"label": signal_array, ...}
    Example: {"x(t)": x, "y(t)": y}
    """
    plt.figure(figsize=(10, 4))

    # .items() gives (key, value) pairs from dictionary
    for label, signal in signals_dict.items():
        plt.plot(t, signal, label=label, linewidth=1.5)

    plt.title(title)
    plt.xlabel("t")
    plt.ylabel("Amplitude")
    plt.legend()   # show the labels box
    plt.grid(True)
    plt.tight_layout()


def plot_discrete(n_axis, signals_dict, title):
    """
    Plot one or more discrete signals as stem plots.

    Stem plot = vertical lines with dots (correct for discrete signals)
    """
    fig, ax = plt.subplots(figsize=(10, 4))

    for label, signal in signals_dict.items():
        markerline, stemlines, baseline = ax.stem(
            n_axis, signal, label=label
        )
        baseline.set_visible(False)   # hide the horizontal baseline

    ax.set_title(title)
    ax.set_xlabel("n")
    ax.set_ylabel("Amplitude")
    ax.legend()
    ax.grid(True, alpha=0.3)   # alpha makes grid transparent (0=invisible, 1=solid)
    plt.tight_layout()


# ============================================================
# SECTION 14: EXTRA USEFUL THINGS FOR EXAM
# ============================================================

# --- 14a: Unit Impulse (delta function) ---
# δ[n] = 1 if n=0, else 0
# Most fundamental discrete signal
def unit_impulse():
    """
    Create a unit impulse signal δ[n].
    δ[n] = 1 at n=0, zero everywhere else.
    """
    sig = init_discrete_signal()
    sig[n_to_pos(0)] = 1   # set position of n=0 to 1
    return sig


# --- 14b: Unit Step ---
# u[n] = 1 if n >= 0, else 0
def unit_step():
    """
    Create a unit step signal u[n].
    u[n] = 1 for n >= 0, zero for n < 0.
    """
    sig = init_discrete_signal()
    for pos in range(2 * INF + 1):
        n = pos_to_n(pos)
        if n >= 0:
            sig[pos] = 1
    return sig


# --- 14c: Check if signal is even ---
def is_even(x):
    """
    Check if signal x is even: x[n] = x[-n] for all n.
    Returns True if even, False if not.
    """
    x_rev = time_reverse_discrete(x)
    # np.allclose checks if all elements are close to each other
    return np.allclose(x, x_rev)


# --- 14d: Check if signal is odd ---
def is_odd(x):
    """
    Check if signal x is odd: x[n] = -x[-n] for all n.
    Returns True if odd, False if not.
    """
    x_rev = time_reverse_discrete(x)
    return np.allclose(x, -x_rev)


# --- 14e: Signal energy ---
def energy(x):
    """
    Compute the energy of a signal.
    E = sum of x[n]^2 for all n

    np.sum adds all elements
    x**2 squares every element
    """
    return float(np.sum(x ** 2))


# --- 14f: Signal power ---
def power(x):
    """
    Compute the average power of a signal.
    P = (1/N) * sum of x[n]^2
    """
    return float(np.mean(x ** 2))


# ============================================================
# SECTION 15: MAIN - RUNS ALL EXAMPLES
# ============================================================

def main():

    print("=" * 60)
    print("CONTINUOUS SIGNAL EXAMPLES")
    print("=" * 60)

    # --- Setup ---
    t = make_time_axis()
    x = continuous_signal(t)

    # --- Time Scaling ---
    k = 2
    y_scaled = time_scale_continuous(t, x, k)
    plot_continuous(
        t,
        {"x(t)": x, f"x(t/{k})": y_scaled},
        f"Continuous Time Scaling: y(t) = x(t/{k})"
    )

    # --- Time Reversal ---
    x_rev = time_reverse_continuous(t, x)
    plot_continuous(
        t,
        {"x(t)": x, "x(-t)": x_rev},
        "Continuous Time Reversal"
    )

    # --- Even-Odd Decomposition ---
    xe, xo = even_odd_continuous(t, x)
    plot_continuous(
        t,
        {"x(t)": x, "xe(t)": xe, "xo(t)": xo},
        "Continuous Even-Odd Decomposition"
    )

    # Verify: xe + xo should equal x
    err = np.max(np.abs(xe + xo - x))
    print(f"Even-Odd reconstruction error (continuous): {err:.10f}")

    print("\n" + "=" * 60)
    print("SINUSOID PHASE/TIME SHIFT EXAMPLES")
    print("=" * 60)

    # --- Sinusoid Setup ---
    A = 1.0
    Omega0 = np.pi / 4
    phi = 0.0
    n_sin = np.arange(-20, 21)   # integer array -20 to 20

    x_sin = sinusoid(n_sin, A, Omega0, phi)

    # --- Part A: Time shift → Phase change ---
    n0 = 3
    x_time_shifted = time_shift_sinusoid(n_sin, A, Omega0, phi, n0)

    # Equivalent phase change
    phi0_equiv = -Omega0 * n0
    x_phase_equiv = phase_change_sinusoid(n_sin, A, Omega0, phi, phi0_equiv)

    err_A = mse(x_time_shifted, x_phase_equiv)
    print(f"Part A: n0={n0}, phi0_equiv={phi0_equiv:.4f}, MSE={err_A:.8f}")
    print(f"  MSE ≈ 0 means time shift and phase change are IDENTICAL")

    plot_discrete(
        n_sin,
        {
            "x[n]": x_sin,
            f"time shift n0={n0}": x_time_shifted,
            f"phase change φ={phi0_equiv:.3f}": x_phase_equiv
        },
        f"Part A: Time shift n0={n0} = Phase change φ₀={phi0_equiv:.3f}"
    )

    # --- Part B: Phase change → Time shift? ---
    phi0 = 1.0
    x_phase_changed = phase_change_sinusoid(n_sin, A, Omega0, phi, phi0)

    ideal_shift = -phi0 / Omega0
    print(f"\nPart B: phi0={phi0}, ideal shift={ideal_shift:.4f}")
    print(f"  {ideal_shift:.4f} is NOT an integer → no perfect time shift exists")

    # Search for best integer shift
    best_k = None
    best_err = None
    for k_try in range(-12, 13):   # range(-12, 13) goes from -12 to 12
        x_try = time_shift_sinusoid(n_sin, A, Omega0, phi, k_try)
        e = mse(x_try, x_phase_changed)
        if best_err is None or e < best_err:
            best_err = e
            best_k = k_try

    print(f"  Best integer shift: k={best_k}, MSE={best_err:.6f}")
    print(f"  MSE > 0 means they are NOT identical")

    x_best_shift = time_shift_sinusoid(n_sin, A, Omega0, phi, best_k)
    plot_discrete(
        n_sin,
        {
            f"phase change φ={phi0}": x_phase_changed,
            f"best time shift k={best_k}": x_best_shift
        },
        f"Part B: Phase change φ₀={phi0} vs best integer shift k={best_k}"
    )

    print("\n" + "=" * 60)
    print("DISCRETE SIGNAL EXAMPLES")
    print("=" * 60)

    n_axis = np.arange(-INF, INF + 1)   # [-8,-7,...,7,8]

    # --- Setup discrete signal ---
    sig = init_discrete_signal()
    sig[n_to_pos(0)]  = 1.0
    sig[n_to_pos(1)]  = 0.5
    sig[n_to_pos(-1)] = 2.0
    sig[n_to_pos(2)]  = 1.0
    sig[n_to_pos(-2)] = 0.5

    print("Original signal values:")
    for pos in range(2 * INF + 1):
        n = pos_to_n(pos)
        if sig[pos] != 0:
            print(f"  x[{n:3d}] = {sig[pos]}")

    # --- Discrete Time Reversal ---
    sig_rev = time_reverse_discrete(sig)
    plot_discrete(
        n_axis,
        {"x[n]": sig, "x[-n]": sig_rev},
        "Discrete Time Reversal"
    )

    # --- Discrete Even-Odd ---
    xe_d, xo_d = even_odd_discrete(sig)
    plot_discrete(
        n_axis,
        {"x[n]": sig, "xe[n]": xe_d, "xo[n]": xo_d},
        "Discrete Even-Odd Decomposition"
    )

    err_d = np.max(np.abs(xe_d + xo_d - sig))
    print(f"Even-Odd reconstruction error (discrete): {err_d:.10f}")

    # Check symmetry properties
    print(f"xe is even: {is_even(xe_d)}")   # should be True
    print(f"xo is odd:  {is_odd(xo_d)}")    # should be True

    # --- Discrete Time Scaling (zeros) ---
    k = 3
    sig_scaled = time_scale_discrete(sig, k)
    plot_discrete(
        n_axis,
        {"x[n]": sig, f"x[n/{k}] zeros": sig_scaled},
        f"Discrete Time Scaling k={k} (zeros between samples)"
    )

    # --- Discrete Time Scaling (interpolate) ---
    sig_interp = time_scale_discrete_interpolate(sig, k)
    plot_discrete(
        n_axis,
        {"x[n]": sig, f"x[n/{k}] interp": sig_interp},
        f"Discrete Time Scaling k={k} (interpolated between samples)"
    )

    # --- Extra: Unit Impulse and Step ---
    impulse = unit_impulse()
    step    = unit_step()
    plot_discrete(
        n_axis,
        {"δ[n]": impulse, "u[n]": step},
        "Unit Impulse and Unit Step"
    )

    print("\n" + "=" * 60)
    print("SIGNAL PROPERTIES")
    print("=" * 60)
    print(f"Energy of original signal:  {energy(sig):.4f}")
    print(f"Power of original signal:   {power(sig):.4f}")
    print(f"Energy of unit impulse:     {energy(impulse):.4f}")
    print(f"Energy of unit step:        {energy(step):.4f}")

    plt.show()   # show all figures at once


# ============================================================
# SECTION 16: QUICK SYNTAX REFERENCE (as comments)
# ============================================================
#
# ARRAYS:
#   np.zeros(n)           → [0, 0, ..., 0]  (n zeros)
#   np.zeros_like(arr)    → zeros with same shape as arr
#   np.ones(n)            → [1, 1, ..., 1]
#   np.arange(a, b)       → [a, a+1, ..., b-1]  (integers)
#   np.linspace(a, b, n)  → n equally spaced floats from a to b
#
# ARRAY OPERATIONS (all element-wise):
#   arr + 5               → add 5 to every element
#   arr * 2               → multiply every element by 2
#   arr / 2               → divide every element by 2
#   arr ** 2              → square every element
#   arr1 + arr2           → add corresponding elements
#   arr1 - arr2           → subtract corresponding elements
#
# ARRAY INDEXING:
#   arr[0]                → first element
#   arr[-1]               → last element
#   arr[2:5]              → elements at positions 2, 3, 4
#   arr[::-1]             → reversed array
#   arr[mask]             → keep only True positions
#
# MATH FUNCTIONS:
#   np.sin(arr)           → sine of every element
#   np.cos(arr)           → cosine of every element
#   np.abs(arr)           → absolute value of every element
#   np.floor(x)           → round DOWN
#   np.ceil(x)            → round UP
#   round(x)              → round to nearest integer
#   np.sqrt(x)            → square root
#   np.pi                 → π = 3.14159...
#
# ARRAY INFO:
#   len(arr)              → number of elements
#   arr.shape             → dimensions
#   np.max(arr)           → largest value
#   np.min(arr)           → smallest value
#   np.sum(arr)           → sum of all elements
#   np.mean(arr)          → average of all elements
#
# COMPARISONS (return True/False arrays):
#   arr >= 0              → True where arr is non-negative
#   arr == 5              → True where arr equals 5
#   (a >= 0) & (a <= 4)   → AND for arrays (use & not 'and')
#   (a < 0) | (a > 4)     → OR for arrays  (use | not 'or')
#   np.isclose(a, b)      → True if a and b are nearly equal
#   np.allclose(a, b)     → True if ALL elements are nearly equal
#
# LOOPS:
#   for i in range(n):           → i = 0, 1, ..., n-1
#   for i, v in enumerate(arr):  → i = index, v = value
#   continue                     → skip to next iteration
#   break                        → exit the loop entirely
#
# OPERATORS:
#   n % k    → remainder (6%3=0, 7%3=1)
#   n // k   → integer division (7//3=2)
#   n / k    → regular division (7/3=2.333)
#
# FUNCTIONS:
#   def func(a, b):       → define function with params a and b
#       return result     → return one value
#   def func(a):
#       return x, y       → return two values (tuple)
#   a, b = func(x)        → receive two return values
#
# PLOTTING:
#   plt.figure()                  → new figure
#   plt.plot(t, x, label="name")  → line plot
#   ax.stem(n, x, label="name")   → stem plot (discrete signals)
#   plt.title("text")             → add title
#   plt.xlabel("text")            → x-axis label
#   plt.ylabel("text")            → y-axis label
#   plt.legend()                  → show labels box
#   plt.grid(True)                → show grid
#   plt.tight_layout()            → fix spacing
#   plt.show()                    → display all figures
#   fig, ax = plt.subplots()      → figure + axes object
#
# F-STRINGS:
#   f"value is {x}"               → insert variable into string
#   f"value is {x:.2f}"           → 2 decimal places
#   f"value is {x:.4f}"           → 4 decimal places
#   f"n={n}\nMSE={e:.6f}"         → \n means new line


if __name__ == "__main__":
    main()