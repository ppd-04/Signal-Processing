import numpy as np
import matplotlib.pyplot as plt

# ----------------------------
# Time axis
# ----------------------------
T_MIN, T_MAX, N = -4.0, 4.0, 4001


def x_of_t(t: np.ndarray) -> np.ndarray:
    """
    Base signal x(t): sinusoidal signal
    """
    return (
        np.sin(2 * np.pi * 0.5 * t)
        + 0.5 * np.sin(2 * np.pi * 1.5 * t)
    )


# ==========================================================
# ANSWER IMPLEMENTATION
# ==========================================================

def interpolate_signal(
    t_original: np.ndarray,
    x_original: np.ndarray,
    t_query: np.ndarray
) -> np.ndarray:
    """
    Interpolate using average of two neighboring samples.
    """
    result = np.zeros_like(t_query, dtype=float)
    
    for i, tq in enumerate(t_query):
        idx = np.searchsorted(t_original, tq)

        if idx==0:
            result[i]=x_original[0]
        elif idx>=len(t_original):
            result[i]=x_original[-1]
        else:
            x_left = x_original[idx-1]
            x_right = x_original[idx]
            result[i] = 0.5*(x_left+x_right)

    return result

def time_scale(
    t: np.ndarray,
    x: np.ndarray,
    k: int
) -> np.ndarray:
    """
    Time sub-scaling:
        y(t) = x(t / k)
    """
    
    # raise NotImplementedError
    t_query = t/k
    validMask = (t_query>=T_MIN) & (t_query<=T_MAX)
    y = np.zeros_like(t, dtype=float)
    y[validMask]=interpolate_signal(t,x,t_query[validMask])
    return y



def plot_pair(t: np.ndarray, x: np.ndarray, y: np.ndarray, title: str):
    """
    Plot graphs.
    """
    # raise NotImplementedError
    plt.figure(figsize=(10,4))
    plt.plot(t, x, label="x(t)", linewidth=1.5)
    plt.plot(t, y, label="y", linestyle = '--')


    plt.title(title)          # Title at top
    plt.xlabel("t")           # x-axis label
    plt.ylabel("Amplitude")   # y-axis label
    plt.legend()              # Show the legend box (uses 'label' from plot())
    plt.grid(True)            # Show grid lines
    plt.tight_layout() 


# ----------------------------
# Main
# ----------------------------
def main():
    t = np.linspace(T_MIN, T_MAX, N)
    x = x_of_t(t)

    k = 2   # sub-scaling factor
    y = time_scale(t, x, k)

    plot_pair(
        t,
        x,
        y,
        title=f"Time Sub-scaling: y(t) = x(t / {k})"
    )
    plt.show()


if __name__ == "__main__":
    main()