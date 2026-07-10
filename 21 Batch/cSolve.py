import numpy as np
import matplotlib.pyplot as plt

INF = 8


def plot(
        signal,
        title=None,
        y_range=(-1, 3),
        figsize=(8, 3),
        x_label='n (Time Index)',
        y_label='x[n]',
        saveTo=None
):
    plt.figure(figsize=figsize)
    plt.xticks(np.arange(-INF, INF + 1, 1))
    y_range = (y_range[0], max(np.max(signal), y_range[1]) + 1)
    plt.ylim(*y_range)
    plt.stem(np.arange(-INF, INF + 1, 1), signal)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.grid(True)
    if saveTo is not None:
        plt.savefig(saveTo)


def init_signal():
    return np.zeros(2 * INF + 1)


# ============================================================
# TASK 1: Time Reversal
# ============================================================

def time_reverse_signal(x: np.ndarray) -> np.ndarray:
    """
    Compute x[-n] for a discrete signal stored as an array.

    The array represents n = -8 to n = +8.
    Time reversal means: output at position p gets the value
    that was at the MIRROR position.

    Mirror of position p = (2*INF) - p = 16 - p

    Example:
        position 0  (n=-8) gets value from position 16 (n=+8)
        position 6  (n=-2) gets value from position 10 (n=+2)
        position 8  (n= 0) gets value from position  8 (n= 0)
        position 10 (n=+2) gets value from position  6 (n=-2)
        position 16 (n=+8) gets value from position  0 (n=-8)

    This is exactly what reversing the array does.
    """

    # METHOD 1: Simple one-liner (numpy, no loop)
    # Just flip the array left to right
    return x[::-1]


    # METHOD 2: Using a loop (easier to understand)
    # (Remove the return above and use this instead)
    #
    # result = np.zeros(2 * INF + 1)
    #
    # for p_out in range(2 * INF + 1):
    #     # Convert output position to time index n
    #     n = p_out - INF
    #
    #     # We need x[-n]
    #     # Convert -n back to an array position
    #     p_in = (-n) + INF
    #
    #     # Check if p_in is within valid range
    #     if 0 <= p_in <= 2 * INF:
    #         result[p_out] = x[p_in]
    #     else:
    #         result[p_out] = 0   # outside range = 0
    #
    # return result


# ============================================================
# TASK 2: Even-Odd Decomposition
# ============================================================

def odd_even_decomposition(x: np.ndarray):
    """
    Decompose x into odd and even components.

    Formula:
        xe[n] = (x[n] + x[-n]) / 2    (even part)
        xo[n] = (x[n] - x[-n]) / 2    (odd part)

    Returns: (xo, xe)  →  odd first, even second
    (as specified in the question: "first array odd, second even")
    """

    # Step 1: Get x[-n] using the time_reverse_signal function
    x_reversed = time_reverse_signal(x)
    # x_reversed[p] = x[-n] at position p

    # Step 2: Compute even part
    # Element-wise: (x[n] + x[-n]) / 2
    xe = (x + x_reversed) / 2

    # Step 3: Compute odd part
    # Element-wise: (x[n] - x[-n]) / 2
    xo = (x - x_reversed) / 2

    # Return odd first, even second (as question specifies)
    return xo, xe


# ============================================================
# MAIN
# ============================================================

def main():
    img_root = '.'

    # Create the signal
    signal = init_signal()
    signal[INF]     = 1      # n= 0, value=1
    signal[INF + 1] = 0.5   # n= 1, value=0.5
    signal[INF - 1] = 2     # n=-1, value=2
    signal[INF + 2] = 1     # n= 2, value=1
    signal[INF - 2] = 0.5   # n=-2, value=0.5

    # Plot original signal
    plot(signal,
         title='Original Signal x[n]',
         saveTo=f'{img_root}/x[n].png')

    # Task 1: Time reversal
    x_reversed = time_reverse_signal(signal)
    plot(x_reversed,
         title='Time Reversed x[-n]',
         saveTo=f'{img_root}/x[-n].png')

    # Task 2: Even-odd decomposition
    xo, xe = odd_even_decomposition(signal)

    plot(xe,
         title='Even Component xe[n]',
         y_range=(-2, 3),
         saveTo=f'{img_root}/xe[n].png')

    plot(xo,
         title='Odd Component xo[n]',
         y_range=(-2, 3),
         saveTo=f'{img_root}/xo[n].png')

    # Verify: xe + xo should equal original x
    reconstruction = xe + xo
    error = np.max(np.abs(reconstruction - signal))
    print("Max reconstruction error:", error)
    # Should print 0.0 or a number very close to 0

    plt.show()


main()