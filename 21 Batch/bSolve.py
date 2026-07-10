import numpy as np
import matplotlib.pyplot as plt

INF = 8


def plot(signal, title=None, y_range=(-1, 3), figsize=(8, 3),
         x_label='n (Time Index)', y_label='x[n]', saveTo=None):
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


def time_scale_signal(x: np.ndarray, k: int) -> np.ndarray:
    """
    Compute y[n] = x[n/k]
    
    Rules:
    - If n/k is an integer AND within [-8, 8]: use x[n/k]
    - Otherwise: set to 0
    
    We work with array positions, not actual n values.
    Array position p corresponds to n = p - INF
    """

    # Create output array filled with zeros
    # Size is 17 (same as input: from n=-8 to n=8)
    y = np.zeros(2 * INF + 1)

    # Loop through every position in the OUTPUT array
    for p_out in range(2 * INF + 1):

        # Convert output position to actual time index n
        # p_out = 0  → n = -8
        # p_out = 8  → n =  0
        # p_out = 16 → n = +8
        n = p_out - INF

        # We need x[n/k]
        # n/k must be an integer for discrete signals
        # Check if n is divisible by k (no remainder)
        if n % k == 0:

            # n/k is an integer
            n_over_k = n // k
            # n // k is INTEGER division (gives whole number)
            # Example: -6 // 3 = -2,  3 // 3 = 1

            # Check if n/k is within our signal range [-8, 8]
            if -INF <= n_over_k <= INF:

                # Convert n/k to array position in the INPUT array
                p_in = n_over_k + INF

                # Copy the value from input to output
                y[p_out] = x[p_in]

        # If n % k != 0, y[p_out] stays 0 (already initialized to 0)

    return y


def time_scale_signal_interpolate(x: np.ndarray, k: int) -> np.ndarray:
    """
    Compute y[n] = x[n/k] with interpolation for intermediate samples.
    
    Rules:
    - If n/k is an integer AND within [-8, 8]: use x[n/k]  (same as before)
    - If n/k is NOT an integer: average the two neighboring original samples
    - If neighbors are outside range: use 0 for that neighbor
    """

    # Create output array filled with zeros
    y = np.zeros(2 * INF + 1)

    # Loop through every position in the output array
    for p_out in range(2 * INF + 1):

        # Convert position to actual time index n
        n = p_out - INF

        # Check if n/k is exactly an integer
        if n % k == 0:
            # Same as Task 1: copy directly
            n_over_k = n // k

            if -INF <= n_over_k <= INF:
                p_in = n_over_k + INF
                y[p_out] = x[p_in]

        else:
            # n/k is NOT an integer
            # Find the two original samples on either side

            # n/k as a decimal number
            n_over_k_decimal = n / k

            # Left neighbor: floor (round down)
            left_n = int(np.floor(n_over_k_decimal))

            # Right neighbor: ceil (round up)
            right_n = int(np.ceil(n_over_k_decimal))

            # Get left value (use 0 if outside range)
            if -INF <= left_n <= INF:
                left_val = x[left_n + INF]
            else:
                left_val = 0.0

            # Get right value (use 0 if outside range)
            if -INF <= right_n <= INF:
                right_val = x[right_n + INF]
            else:
                right_val = 0.0

            # Average the two neighbors
            y[p_out] = (left_val + right_val) / 2

    return y


def main():
    img_root = '.'

    # Create and set up the signal
    signal = init_signal()
    signal[INF]     = 1      # n=0,  value=1
    signal[INF + 1] = 0.5   # n=1,  value=0.5
    signal[INF - 1] = 2     # n=-1, value=2
    signal[INF + 2] = 1     # n=2,  value=1
    signal[INF - 2] = 0.5   # n=-2, value=0.5

    plot(signal,
         title='Original Signal x[n]',
         saveTo=f'{img_root}/x[n].png')

    plot(time_scale_signal(signal, 3),
         title='x[n/3]',
         saveTo=f'{img_root}/x[n divided by 3].png')

    plot(time_scale_signal(signal, 1),
         title='x[n/1]',
         saveTo=f'{img_root}/x[n divided by 1].png')

    plot(time_scale_signal_interpolate(signal, 3),
         title='x[n/3] with interpolation',
         saveTo=f'{img_root}/x[n divided by 3]_with_interpolation.png')

    plot(time_scale_signal_interpolate(signal, 1),
         title='x[n/1] with interpolation',
         saveTo=f'{img_root}/x[n divided by 1]_with_interpolation.png')

    plt.show()


main()