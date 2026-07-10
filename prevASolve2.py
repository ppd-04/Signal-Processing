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
    t_original: np.ndarray,   # the original time axis (e.g., [-4, ..., 4])
    x_original: np.ndarray,   # the original signal values
    t_query: np.ndarray       # the NEW time points we want values for
) -> np.ndarray:
    """
    Interpolate using average of two neighboring samples.
    
    For each query time point, find its two nearest neighbors
    in the original signal and return their average.
    """
    
    # Create output array, same size as t_query, filled with 0s
    # Shape will be (len(t_query),)
    result = np.zeros_like(t_query, dtype=float)
    
    # Loop through each query time point
    for i, tq in enumerate(t_query):
        # searchsorted finds the index where tq would fit in t_original
        # to keep the array sorted
        # Example: t_original=[0,1,2,3], tq=1.5 → idx=2
        idx = np.searchsorted(t_original, tq)
        
        # Edge case: if tq is exactly at or before start
        if idx == 0:
            result[i] = x_original[0]
        
        # Edge case: if tq is exactly at or beyond end
        elif idx >= len(t_original):
            result[i] = x_original[-1]   # -1 means last element
        
        # Normal case: tq is between two samples
        else:
            # Left neighbor
            x_left = x_original[idx - 1]
            # Right neighbor  
            x_right = x_original[idx]
            
            # Average of the two neighbors
            result[i] = 0.5 * (x_left + x_right)
    
    return result


def time_scale(
    t: np.ndarray,   # original time axis
    x: np.ndarray,   # original signal
    k: int           # scaling factor
) -> np.ndarray:
    """
    Time sub-scaling: y(t) = x(t / k)
    
    Steps:
    1. For each time point t, compute t/k
    2. Find x(t/k) using interpolation
    3. Only keep values where t/k is within original time range
    """
    
    # Step 1: Compute the "query" time points
    # If t = [-4,...,4] and k=2, then t_query = [-2,...,2]
    # We are asking: "give me x at these scaled time points"
    t_query = t / k   # Element-wise division (every element divided by k)
    
    # Step 2: Find which query points are WITHIN the original range
    # t/k must be between T_MIN and T_MAX
    # Example: t=3, k=2 → t/k=1.5 ✓ (within [-4,4])
    # Example: t=10, k=2 → t/k=5.0 ✗ (outside [-4,4])
    valid_mask = (t_query >= T_MIN) & (t_query <= T_MAX)
    #             ^^^^^^^^^^^^^^^^     ^^^^^^^^^^^^^^^^
    #             left condition  AND  right condition
    # valid_mask is array of True/False
    
    # Step 3: Create output array filled with 0 (for out-of-range points)
    y = np.zeros_like(t, dtype=float)
    
    # Step 4: Only interpolate for valid (in-range) points
    # t_query[valid_mask] → only the valid query time points
    y[valid_mask] = interpolate_signal(
        t,                      # original time axis
        x,                      # original signal values
        t_query[valid_mask]     # only valid query points
    )
    
    return y


def plot_pair(t: np.ndarray, x: np.ndarray, y: np.ndarray, title: str):
    """
    Plot x(t) and y(t) on the same figure.
    """
    
    # Create a figure with specific size (width=10, height=4 inches)
    plt.figure(figsize=(10, 4))
    
    # Plot x(t)
    # t → x-axis values
    # x → y-axis values  
    # label → text that appears in legend
    # linewidth → thickness of line
    plt.plot(t, x, label="x(t)", linewidth=1.5)
    
    # Plot y(t) on SAME figure (no new plt.figure())
    # '--' makes it a dashed line
    plt.plot(t, y, label="y(t) = x(t/k)", linewidth=1.5, linestyle='--')
    
    # Add labels and formatting
    plt.title(title)          # Title at top
    plt.xlabel("t")           # x-axis label
    plt.ylabel("Amplitude")   # y-axis label
    plt.legend()              # Show the legend box (uses 'label' from plot())
    plt.grid(True)            # Show grid lines
    plt.tight_layout()        # Automatically adjust spacing


# ----------------------------
# Main
# ----------------------------
def main():
    # Create time axis: 4001 points from -4 to 4
    t = np.linspace(T_MIN, T_MAX, N)
    
    # Compute base signal
    x = x_of_t(t)

    k = 2   # sub-scaling factor
    
    # Compute scaled signal
    y = time_scale(t, x, k)

    plot_pair(
        t,
        x,
        y,
        title=f"Time Sub-scaling: y(t) = x(t / {k})"
        # f"..." is an f-string: {k} gets replaced with value of k
        # So if k=2, title becomes "Time Sub-scaling: y(t) = x(t / 2)"
    )
    plt.show()   # Display the figure window


if __name__ == "__main__":
    main()