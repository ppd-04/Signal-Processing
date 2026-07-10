# ============================================================
# COMPLETE NUMPY AND MATPLOTLIB SYNTAX REFERENCE
# Every syntax is explained with comments and examples
# ============================================================

import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# SECTION 1: CREATING ARRAYS
# ============================================================

def section1_creating_arrays():
    print("\n" + "="*50)
    print("SECTION 1: CREATING ARRAYS")
    print("="*50)

    # ----------------------------------------------------------
    # np.array([...])
    # Creates a numpy array from a Python list
    # Syntax: np.array(list)
    # ----------------------------------------------------------
    a = np.array([1, 2, 3, 4, 5])
    print("np.array([1,2,3,4,5])  =", a)
    # Output: [1 2 3 4 5]

    # 2D array (matrix)
    b = np.array([[1, 2, 3],
                  [4, 5, 6]])
    print("2D array:\n", b)
    # Output: [[1 2 3]
    #          [4 5 6]]

    # ----------------------------------------------------------
    # np.zeros(n)
    # Creates array of n zeros
    # Syntax: np.zeros(size)
    # ----------------------------------------------------------
    z = np.zeros(5)
    print("\nnp.zeros(5)           =", z)
    # Output: [0. 0. 0. 0. 0.]

    # np.zeros with dtype
    # dtype=float  → decimal numbers (default)
    # dtype=int    → whole numbers
    z_int = np.zeros(5, dtype=int)
    print("np.zeros(5, dtype=int)=", z_int)
    # Output: [0 0 0 0 0]

    # ----------------------------------------------------------
    # np.ones(n)
    # Creates array of n ones
    # ----------------------------------------------------------
    o = np.ones(5)
    print("\nnp.ones(5)            =", o)
    # Output: [1. 1. 1. 1. 1.]

    # ----------------------------------------------------------
    # np.zeros_like(arr)
    # Creates array of zeros with SAME SIZE as arr
    # Useful when you want output same size as input
    # ----------------------------------------------------------
    original = np.array([10.0, 20.0, 30.0])
    like_z   = np.zeros_like(original)
    print("\nnp.zeros_like([10,20,30]) =", like_z)
    # Output: [0. 0. 0.]

    # np.ones_like(arr) - same but with ones
    like_o = np.ones_like(original)
    print("np.ones_like([10,20,30])  =", like_o)
    # Output: [1. 1. 1.]

    # ----------------------------------------------------------
    # np.full(size, value)
    # Creates array filled with a specific value
    # ----------------------------------------------------------
    f = np.full(5, 7.0)
    print("\nnp.full(5, 7.0)       =", f)
    # Output: [7. 7. 7. 7. 7.]

    # np.full_like(arr, value)
    # Same size as arr, filled with value
    fl = np.full_like(original, np.nan)
    print("np.full_like(arr, nan) =", fl)
    # Output: [nan nan nan]

    # ----------------------------------------------------------
    # np.arange(start, stop, step)
    # Creates array of evenly spaced VALUES (like Python range)
    # IMPORTANT: stops BEFORE stop value
    # Syntax: np.arange(start, stop, step)
    #         np.arange(stop)           → starts at 0
    #         np.arange(start, stop)    → step=1
    # ----------------------------------------------------------
    r1 = np.arange(5)
    print("\nnp.arange(5)          =", r1)
    # Output: [0 1 2 3 4]

    r2 = np.arange(2, 8)
    print("np.arange(2, 8)       =", r2)
    # Output: [2 3 4 5 6 7]

    r3 = np.arange(-8, 9)      # -8 to 8 (stops before 9)
    print("np.arange(-8, 9)      =", r3)
    # Output: [-8 -7 -6 ... 6 7 8]

    r4 = np.arange(0, 1, 0.25)  # step of 0.25
    print("np.arange(0,1,0.25)   =", r4)
    # Output: [0.   0.25 0.5  0.75]

    # ----------------------------------------------------------
    # np.linspace(start, stop, num)
    # Creates array of num equally spaced points from start to stop
    # IMPORTANT: INCLUDES the stop value (unlike arange)
    # Used for continuous time axis
    # ----------------------------------------------------------
    ls = np.linspace(0, 1, 5)
    print("\nnp.linspace(0, 1, 5)  =", ls)
    # Output: [0.   0.25 0.5  0.75 1.  ]

    t = np.linspace(-4, 4, 9)
    print("np.linspace(-4,4,9)   =", t)
    # Output: [-4. -3. -2. -1.  0.  1.  2.  3.  4.]


# ============================================================
# SECTION 2: ARRAY INDEXING AND SLICING
# ============================================================

def section2_indexing():
    print("\n" + "="*50)
    print("SECTION 2: ARRAY INDEXING AND SLICING")
    print("="*50)

    arr = np.array([10, 20, 30, 40, 50, 60, 70])
    print("arr =", arr)
    print("     positions: 0   1   2   3   4   5   6")

    # ----------------------------------------------------------
    # Single element access
    # arr[index]
    # Positive index: count from left  (starts at 0)
    # Negative index: count from right (starts at -1)
    # ----------------------------------------------------------
    print("\narr[0]   =", arr[0])    # first element  → 10
    print("arr[1]   =", arr[1])     # second element → 20
    print("arr[-1]  =", arr[-1])    # last element   → 70
    print("arr[-2]  =", arr[-2])    # second to last → 60

    # ----------------------------------------------------------
    # Slicing: arr[start:stop:step]
    # Gets elements from start to stop-1
    # If start is empty: begins from 0
    # If stop is empty:  goes to end
    # If step is empty:  step=1
    # ----------------------------------------------------------
    print("\narr[1:4]    =", arr[1:4])    # positions 1,2,3  → [20 30 40]
    print("arr[:3]     =", arr[:3])      # positions 0,1,2  → [10 20 30]
    print("arr[4:]     =", arr[4:])      # position 4 to end→ [50 60 70]
    print("arr[::2]    =", arr[::2])     # every 2nd        → [10 30 50 70]
    print("arr[::-1]   =", arr[::-1])    # REVERSED!        → [70 60 50 40 30 20 10]
    print("arr[1:6:2]  =", arr[1:6:2])  # 1,3,5            → [20 40 60]

    # WHY arr[::-1] reverses:
    # step=-1 means go backwards from end to start
    # This is how we do time reversal x(-t) or x[-n]

    # ----------------------------------------------------------
    # Boolean (True/False) Masking
    # arr[condition] keeps only elements where condition is True
    # ----------------------------------------------------------
    arr2 = np.array([1, 5, 2, 8, 3, 7, 4])
    print("\narr2 =", arr2)

    mask = arr2 > 4
    print("arr2 > 4          =", mask)
    # Output: [False  True False  True False  True False]

    filtered = arr2[mask]
    print("arr2[arr2 > 4]    =", filtered)
    # Output: [5 8 7]

    # Can also write directly:
    print("arr2[arr2 >= 3]   =", arr2[arr2 >= 3])
    # Output: [5 8 3 7 4]

    # ----------------------------------------------------------
    # Setting values using mask
    # arr[mask] = value  → only change True positions
    # ----------------------------------------------------------
    arr3 = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    arr3[arr3 > 3] = 0.0    # set values > 3 to zero
    print("\nAfter arr3[arr3>3]=0:", arr3)
    # Output: [1. 2. 3. 0. 0.]

    # ----------------------------------------------------------
    # Combined conditions
    # & means AND (both must be True)
    # | means OR  (at least one must be True)
    # IMPORTANT: use & and |, NOT 'and' and 'or' for arrays
    # ----------------------------------------------------------
    arr4 = np.array([-3, -1, 0, 2, 4, 6])
    both = arr4[(arr4 >= 0) & (arr4 <= 4)]
    print("\narr4[(arr4>=0)&(arr4<=4)] =", both)
    # Output: [0 2 4]


# ============================================================
# SECTION 3: ARRAY OPERATIONS (all element-wise)
# ============================================================

def section3_operations():
    print("\n" + "="*50)
    print("SECTION 3: ARRAY OPERATIONS")
    print("="*50)

    a = np.array([1.0, 2.0, 3.0, 4.0])
    b = np.array([10.0, 20.0, 30.0, 40.0])
    print("a =", a)
    print("b =", b)

    # ----------------------------------------------------------
    # Basic arithmetic (element-wise)
    # Every operation applies to EACH element
    # ----------------------------------------------------------
    print("\na + b    =", a + b)     # [11. 22. 33. 44.]
    print("a - b    =", a - b)      # [-9. -18. -27. -36.]
    print("a * b    =", a * b)      # [10. 40. 90. 160.]
    print("a / b    =", a / b)      # [0.1 0.1 0.1 0.1]
    print("a ** 2   =", a ** 2)     # [1. 4. 9. 16.]   (square)
    print("a * 5    =", a * 5)      # [5. 10. 15. 20.] (scalar)
    print("a + 100  =", a + 100)    # [101. 102. 103. 104.]
    print("a / 2    =", a / 2)      # [0.5 1. 1.5 2.]

    # ----------------------------------------------------------
    # Integer operations
    # %  → remainder (modulo)
    # // → integer division (floor division)
    # ----------------------------------------------------------
    n = np.array([-6, -5, -4, -3, -2, -1, 0, 1, 2, 3])
    k = 3
    print("\nn    =", n)
    print("n % 3 =", n % 3)   # remainder when divided by 3
    # Output: [0 1 2 0 1 2 0 1 2 0]
    # 0 means n divides evenly by 3

    print("n // 3=", n // 3)  # integer division
    # Output: [-2 -2 -2 -1 -1 -1  0  0  0  1]

    # Used in discrete time scaling:
    # n % k == 0 means n/k is an integer (valid sample)
    divisible = n[n % k == 0]
    print("n where n%3==0:", divisible)
    # Output: [-6 -3  0  3]


# ============================================================
# SECTION 4: MATH FUNCTIONS
# ============================================================

def section4_math():
    print("\n" + "="*50)
    print("SECTION 4: MATH FUNCTIONS")
    print("="*50)

    arr = np.array([-3.7, -1.2, 0.0, 1.5, 3.9])
    print("arr =", arr)

    # ----------------------------------------------------------
    # np.abs(arr)
    # Absolute value of every element
    # |-3| = 3,  |1.5| = 1.5
    # ----------------------------------------------------------
    print("\nnp.abs(arr)   =", np.abs(arr))
    # Output: [3.7 1.2 0.  1.5 3.9]

    # ----------------------------------------------------------
    # np.floor(arr)
    # Round DOWN to nearest integer (towards negative infinity)
    # ----------------------------------------------------------
    print("np.floor(arr) =", np.floor(arr))
    # Output: [-4. -2.  0.  1.  3.]
    # -3.7 → -4 (rounds DOWN, not towards zero)
    #  1.5 → 1  (rounds DOWN)

    # ----------------------------------------------------------
    # np.ceil(arr)
    # Round UP to nearest integer (towards positive infinity)
    # ----------------------------------------------------------
    print("np.ceil(arr)  =", np.ceil(arr))
    # Output: [-3. -1.  0.  2.  4.]
    # -3.7 → -3 (rounds UP, closer to zero)
    #  1.5 → 2  (rounds UP)

    # ----------------------------------------------------------
    # round(x) or np.round(arr)
    # Round to nearest integer (0.5 rounds to nearest even)
    # ----------------------------------------------------------
    print("np.round(arr) =", np.round(arr))
    # Output: [-4. -1.  0.  2.  4.]

    print("round(2.3)    =", round(2.3))   # 2
    print("round(2.7)    =", round(2.7))   # 3
    print("round(2.5)    =", round(2.5))   # 2 (rounds to even)

    # ----------------------------------------------------------
    # int(x)
    # Converts a single value to integer (truncates decimal)
    # ONLY works on single values, not arrays
    # ----------------------------------------------------------
    print("\nint(2.9)      =", int(2.9))   # 2 (not rounded, just cuts off)
    print("int(-2.9)     =", int(-2.9))   # -2 (towards zero)
    print("int(np.floor(-3.7)) =", int(np.floor(-3.7)))  # -4

    # ----------------------------------------------------------
    # np.sqrt(arr)  → square root
    # np.exp(arr)   → e^x
    # np.log(arr)   → natural log
    # ----------------------------------------------------------
    print("\nnp.sqrt([4,9,16]) =", np.sqrt([4, 9, 16]))
    # Output: [2. 3. 4.]

    # ----------------------------------------------------------
    # Trigonometry
    # np.sin, np.cos, np.tan
    # Input is in RADIANS
    # ----------------------------------------------------------
    t = np.array([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi])
    print("\nnp.sin([0,pi/2,pi,...]) =", np.round(np.sin(t), 2))
    # Output: [0. 1. 0. -1. 0.]  (round removes floating point errors)

    print("np.cos([0,pi/2,pi,...]) =", np.round(np.cos(t), 2))
    # Output: [ 1.  0. -1.  0.  1.]

    print("np.pi =", np.pi)
    # Output: 3.141592653589793

    # ----------------------------------------------------------
    # np.clip(arr, min, max)
    # Keeps values within [min, max]
    # Values below min become min
    # Values above max become max
    # ----------------------------------------------------------
    c = np.array([-5, -2, 0, 3, 8, 15])
    print("\nnp.clip(arr, 0, 10) =", np.clip(c, 0, 10))
    # Output: [ 0  0  0  3  8 10]

    # ----------------------------------------------------------
    # np.isclose(a, b)
    # Returns True if a and b are nearly equal
    # Used because floating point math has tiny errors
    # Example: 0.1 + 0.2 = 0.30000000000000004 (not exactly 0.3)
    # ----------------------------------------------------------
    print("\n0.1 + 0.2 == 0.3   :", 0.1 + 0.2 == 0.3)          # False!
    print("np.isclose(0.1+0.2, 0.3):", np.isclose(0.1+0.2, 0.3))  # True

    # np.allclose checks if ALL elements are close
    a1 = np.array([1.0, 2.0, 3.0])
    a2 = np.array([1.0, 2.0000001, 3.0])
    print("np.allclose(a1, a2):", np.allclose(a1, a2))   # True


# ============================================================
# SECTION 5: ARRAY STATISTICS
# ============================================================

def section5_statistics():
    print("\n" + "="*50)
    print("SECTION 5: ARRAY STATISTICS")
    print("="*50)

    arr = np.array([3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0])
    print("arr =", arr)

    # ----------------------------------------------------------
    # np.sum(arr)   → add all elements
    # np.mean(arr)  → average (sum / count)
    # np.max(arr)   → largest value
    # np.min(arr)   → smallest value
    # np.std(arr)   → standard deviation
    # ----------------------------------------------------------
    print("\nnp.sum(arr)  =", np.sum(arr))    # 25.0
    print("np.mean(arr) =", np.mean(arr))    # 3.571...
    print("np.max(arr)  =", np.max(arr))     # 9.0
    print("np.min(arr)  =", np.min(arr))     # 1.0
    print("np.std(arr)  =", round(float(np.std(arr)), 3))  # 2.621

    # ----------------------------------------------------------
    # np.argmax(arr)  → INDEX of largest value
    # np.argmin(arr)  → INDEX of smallest value
    # ----------------------------------------------------------
    print("\nnp.argmax(arr) =", np.argmax(arr))  # 5 (index of 9.0)
    print("np.argmin(arr) =", np.argmin(arr))    # 1 (index of first 1.0)

    # ----------------------------------------------------------
    # len(arr)   → number of elements
    # arr.shape  → dimensions as tuple
    # arr.size   → total number of elements
    # ----------------------------------------------------------
    print("\nlen(arr)    =", len(arr))      # 7
    print("arr.shape   =", arr.shape)      # (7,)
    print("arr.size    =", arr.size)       # 7

    mat = np.array([[1, 2, 3], [4, 5, 6]])
    print("\nmatrix.shape =", mat.shape)   # (2, 3) → 2 rows, 3 cols

    # ----------------------------------------------------------
    # MSE (Mean Squared Error) - used to compare signals
    # mse = mean((a - b)^2)
    # MSE = 0 means signals are identical
    # ----------------------------------------------------------
    sig1 = np.array([1.0, 2.0, 3.0, 4.0])
    sig2 = np.array([1.0, 2.0, 3.0, 4.0])
    sig3 = np.array([1.0, 2.5, 3.0, 4.0])

    mse12 = float(np.mean((sig1 - sig2)**2))
    mse13 = float(np.mean((sig1 - sig3)**2))
    print("\nMSE(sig1, sig2) =", mse12)   # 0.0 (identical)
    print("MSE(sig1, sig3) =", mse13)    # 0.0625 (different)


# ============================================================
# SECTION 6: ARRAY MANIPULATION
# ============================================================

def section6_manipulation():
    print("\n" + "="*50)
    print("SECTION 6: ARRAY MANIPULATION")
    print("="*50)

    # ----------------------------------------------------------
    # np.concatenate([arr1, arr2])
    # Joins arrays together end to end
    # ----------------------------------------------------------
    a = np.array([1, 2, 3])
    b = np.array([4, 5, 6])
    c = np.concatenate([a, b])
    print("np.concatenate([a,b]) =", c)
    # Output: [1 2 3 4 5 6]

    # ----------------------------------------------------------
    # np.flip(arr)
    # Reverses the array (same as arr[::-1])
    # ----------------------------------------------------------
    arr = np.array([10, 20, 30, 40, 50])
    print("\nnp.flip(arr)  =", np.flip(arr))
    print("arr[::-1]     =", arr[::-1])
    # Both give: [50 40 30 20 10]

    # ----------------------------------------------------------
    # np.searchsorted(arr, value)
    # Finds the INDEX where value would be inserted to keep array sorted
    # Used to find neighboring samples for interpolation
    # ----------------------------------------------------------
    sorted_arr = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    print("\nsorted_arr =", sorted_arr)
    print("searchsorted(arr, 1.5) =", np.searchsorted(sorted_arr, 1.5))
    # Output: 2  (1.5 fits between index 1 and 2)
    print("searchsorted(arr, 2.0) =", np.searchsorted(sorted_arr, 2.0))
    # Output: 2  (2.0 fits at index 2)
    print("searchsorted(arr, 0.0) =", np.searchsorted(sorted_arr, 0.0))
    # Output: 0  (0.0 fits at start)

    # How to use for interpolation:
    tq = 1.7
    idx = np.searchsorted(sorted_arr, tq)
    print(f"\nFor tq={tq}: idx={idx}")
    print(f"  Left  neighbor: arr[{idx-1}] = {sorted_arr[idx-1]}")
    print(f"  Right neighbor: arr[{idx}]   = {sorted_arr[idx]}")
    print(f"  Average: {0.5*(sorted_arr[idx-1]+sorted_arr[idx])}")

    # ----------------------------------------------------------
    # np.where(condition, value_if_true, value_if_false)
    # Element-wise if-else
    # ----------------------------------------------------------
    arr2 = np.array([-3, -1, 0, 2, 4])
    result = np.where(arr2 >= 0, arr2, 0)   # keep positives, set negatives to 0
    print("\nnp.where(arr>=0, arr, 0) =", result)
    # Output: [0 0 0 2 4]


# ============================================================
# SECTION 7: PYTHON LOOPS AND CONTROL FLOW
# ============================================================

def section7_loops():
    print("\n" + "="*50)
    print("SECTION 7: LOOPS AND CONTROL FLOW")
    print("="*50)

    # ----------------------------------------------------------
    # for loop with range
    # range(n)        → 0, 1, ..., n-1
    # range(a, b)     → a, a+1, ..., b-1
    # range(a, b, s)  → a, a+s, a+2s, ... (step s)
    # ----------------------------------------------------------
    print("range(5)       :", list(range(5)))
    # [0, 1, 2, 3, 4]

    print("range(2, 7)    :", list(range(2, 7)))
    # [2, 3, 4, 5, 6]

    print("range(-3, 4)   :", list(range(-3, 4)))
    # [-3, -2, -1, 0, 1, 2, 3]

    print("range(0, 10, 3):", list(range(0, 10, 3)))
    # [0, 3, 6, 9]

    # ----------------------------------------------------------
    # enumerate(arr)
    # Gives BOTH index AND value in a loop
    # Without enumerate: for v in arr   → only values
    # With enumerate:    for i,v in ... → index AND value
    # ----------------------------------------------------------
    arr = np.array([10, 20, 30])
    print("\nenumerate example:")
    for i, v in enumerate(arr):
        print(f"  i={i}, v={v}")
    # i=0, v=10
    # i=1, v=20
    # i=2, v=30

    # ----------------------------------------------------------
    # continue → skip rest of this iteration, go to next
    # break    → exit the loop completely
    # ----------------------------------------------------------
    print("\ncontinue example (skip negatives):")
    for v in [-2, -1, 0, 1, 2]:
        if v < 0:
            continue     # skip negative numbers
        print(f"  v = {v}")
    # prints 0, 1, 2

    print("\nbreak example (stop at 3):")
    for v in [1, 2, 3, 4, 5]:
        if v > 3:
            break        # stop when v exceeds 3
        print(f"  v = {v}")
    # prints 1, 2, 3

    # ----------------------------------------------------------
    # if / elif / else
    # ----------------------------------------------------------
    def classify(n, k):
        """Classify n with respect to k"""
        if n % k == 0:
            return "divisible"
        elif n % k == 1:
            return "remainder 1"
        else:
            return "other remainder"

    print("\nclassify(6, 3)  =", classify(6, 3))   # divisible
    print("classify(7, 3)  =", classify(7, 3))    # remainder 1
    print("classify(8, 3)  =", classify(8, 3))    # other remainder

    # ----------------------------------------------------------
    # Ternary (one-line if-else)
    # value = x if condition else y
    # ----------------------------------------------------------
    x = 5
    label = "positive" if x > 0 else "non-positive"
    print(f"\n{x} is {label}")

    # ----------------------------------------------------------
    # None - represents "no value yet"
    # Used to initialize variables before a loop
    # ----------------------------------------------------------
    best = None    # no best value yet

    for v in [5, 2, 8, 1, 9, 3]:
        if best is None or v < best:   # 'is None' checks for None
            best = v

    print("\nSmallest value found:", best)   # 1


# ============================================================
# SECTION 8: FUNCTIONS AND RETURN VALUES
# ============================================================

def section8_functions():
    print("\n" + "="*50)
    print("SECTION 8: FUNCTIONS")
    print("="*50)

    # ----------------------------------------------------------
    # Basic function
    # def function_name(parameters):
    #     body
    #     return result
    # ----------------------------------------------------------
    def add(a, b):
        return a + b

    print("add(3, 4) =", add(3, 4))   # 7

    # ----------------------------------------------------------
    # Function with default parameter values
    # If not provided, uses default
    # ----------------------------------------------------------
    def greet(name, greeting="Hello"):
        return f"{greeting}, {name}!"

    print(greet("Alice"))            # Hello, Alice!
    print(greet("Bob", "Hi"))        # Hi, Bob!

    # ----------------------------------------------------------
    # Returning multiple values
    # Python returns them as a TUPLE
    # Unpack with: a, b = function()
    # ----------------------------------------------------------
    def min_max(arr):
        return np.min(arr), np.max(arr)   # returns two values

    data = np.array([3, 1, 4, 1, 5, 9])
    lo, hi = min_max(data)    # unpack both values
    print(f"\nmin={lo}, max={hi}")   # min=1, max=9

    # Can also receive as tuple:
    result = min_max(data)
    print("as tuple:", result)        # (1, 9)
    print("first:",    result[0])     # 1
    print("second:",   result[1])     # 9

    # ----------------------------------------------------------
    # Type annotations (optional but helpful)
    # Shows what type each parameter should be
    # Does NOT enforce the types, just for readability
    # ----------------------------------------------------------
    def time_scale(
        t: np.ndarray,    # says t should be a numpy array
        k: int            # says k should be an integer
    ) -> np.ndarray:      # says function returns a numpy array
        return t / k

    t = np.array([0.0, 1.0, 2.0])
    print("\ntime_scale(t, 2) =", time_scale(t, 2))
    # Output: [0.  0.5 1. ]

    # ----------------------------------------------------------
    # raise NotImplementedError
    # Used in templates where you have to fill in the function
    # If you call the function without implementing it, it crashes
    # ----------------------------------------------------------
    def not_yet_done():
        raise NotImplementedError   # this crashes on purpose

    # ----------------------------------------------------------
    # f-strings (formatted strings)
    # f"text {variable}" inserts variable into string
    # f"text {variable:.2f}" shows 2 decimal places
    # f"text {variable:.4f}" shows 4 decimal places
    # f"text {variable:d}"   shows as integer
    # \n inside string means new line
    # ----------------------------------------------------------
    pi = 3.14159265
    n0 = 3
    mse_val = 0.000142

    print(f"\npi = {pi}")            # pi = 3.14159265
    print(f"pi = {pi:.2f}")         # pi = 3.14
    print(f"pi = {pi:.4f}")         # pi = 3.1416
    print(f"n0 = {n0:d}")           # n0 = 3
    print(f"mse = {mse_val:.6f}")   # mse = 0.000142
    print(f"Line1\nLine2")          # Line1 (newline) Line2


# ============================================================
# SECTION 9: DICTIONARIES
# ============================================================

def section9_dictionaries():
    print("\n" + "="*50)
    print("SECTION 9: DICTIONARIES")
    print("="*50)

    # ----------------------------------------------------------
    # Dictionary: stores key-value pairs
    # Useful for passing multiple signals to a plot function
    # Syntax: {key: value, key: value, ...}
    # ----------------------------------------------------------
    t = np.linspace(-4, 4, 100)
    signals = {
        "x(t)":  np.sin(t),
        "xe(t)": np.cos(t),
        "xo(t)": np.sin(t) * 0.5
    }

    # .items() gives (key, value) pairs
    print("Signal names:")
    for name, values in signals.items():
        print(f"  {name}: length={len(values)}")

    # Access by key
    print("\nsignals['x(t)'][0] =", round(float(signals["x(t)"][0]), 4))


# ============================================================
# SECTION 10: MATPLOTLIB - BASIC PLOTTING
# ============================================================

def section10_basic_plotting():
    print("\n" + "="*50)
    print("SECTION 10: BASIC MATPLOTLIB PLOTTING")
    print("="*50)

    t = np.linspace(-4, 4, 1000)
    x = np.sin(2 * np.pi * 0.5 * t)
    y = 0.5 * np.cos(2 * np.pi * 0.5 * t)

    # ----------------------------------------------------------
    # plt.figure(figsize=(width, height))
    # Creates a new figure window
    # figsize is in inches
    # ----------------------------------------------------------
    plt.figure(figsize=(10, 4))

    # ----------------------------------------------------------
    # plt.plot(x_values, y_values)
    # Draws a line connecting the points
    # ----------------------------------------------------------
    plt.plot(t, x)               # basic plot
    plt.plot(t, y)               # second line on same figure

    # ----------------------------------------------------------
    # plt.title, plt.xlabel, plt.ylabel
    # Add text labels to the figure
    # ----------------------------------------------------------
    plt.title("Basic Plot Example")
    plt.xlabel("Time t")
    plt.ylabel("Amplitude")

    # ----------------------------------------------------------
    # plt.grid(True)
    # Adds grid lines to help read values
    # alpha controls transparency (0=invisible, 1=solid)
    # ----------------------------------------------------------
    plt.grid(True)
    plt.grid(True, alpha=0.3)    # lighter grid

    # ----------------------------------------------------------
    # plt.legend()
    # Shows a box with line labels
    # Labels come from label="..." in plt.plot()
    # ----------------------------------------------------------
    plt.figure(figsize=(10, 4))
    plt.plot(t, x, label="sin(t)")    # label= names this line
    plt.plot(t, y, label="cos(t)")    # label= names this line
    plt.legend()                       # shows the label box
    plt.title("Plot with Legend")
    plt.xlabel("t")
    plt.ylabel("Amplitude")
    plt.grid(True)

    print("Basic plot created (see figure)")


# ============================================================
# SECTION 11: MATPLOTLIB - LINE STYLES AND COLORS
# ============================================================

def section11_line_styles():
    print("\n" + "="*50)
    print("SECTION 11: LINE STYLES AND COLORS")
    print("="*50)

    t = np.linspace(0, 4, 500)

    plt.figure(figsize=(10, 6))

    # ----------------------------------------------------------
    # linestyle (or ls) controls the line pattern
    # '-'   → solid line (default)
    # '--'  → dashed line
    # ':'   → dotted line
    # '-.'  → dash-dot line
    # ----------------------------------------------------------
    plt.plot(t, np.sin(t) + 3,  label="solid '-'",    linestyle='-')
    plt.plot(t, np.sin(t) + 2,  label="dashed '--'",  linestyle='--')
    plt.plot(t, np.sin(t) + 1,  label="dotted ':'",   linestyle=':')
    plt.plot(t, np.sin(t) + 0,  label="dashdot '-.'", linestyle='-.')

    # ----------------------------------------------------------
    # color controls line color
    # 'r' = red, 'g' = green, 'b' = blue
    # 'k' = black, 'orange', 'purple', etc.
    # ----------------------------------------------------------
    plt.figure(figsize=(10, 4))
    plt.plot(t, np.sin(t),       color='r',      label='red')
    plt.plot(t, np.sin(t) + 1,   color='b',      label='blue')
    plt.plot(t, np.sin(t) + 2,   color='green',  label='green')
    plt.plot(t, np.sin(t) + 3,   color='orange', label='orange')

    # ----------------------------------------------------------
    # linewidth (or lw) controls line thickness
    # ----------------------------------------------------------
    plt.figure(figsize=(10, 4))
    plt.plot(t, np.sin(t),     linewidth=0.5, label='thin 0.5')
    plt.plot(t, np.sin(t)+1,   linewidth=1.5, label='normal 1.5')
    plt.plot(t, np.sin(t)+2,   linewidth=3.0, label='thick 3.0')

    plt.legend()
    plt.title("Line Width Example")
    plt.grid(True)

    print("Line style plots created")


# ============================================================
# SECTION 12: MATPLOTLIB - STEM PLOTS (for discrete signals)
# ============================================================

def section12_stem_plots():
    print("\n" + "="*50)
    print("SECTION 12: STEM PLOTS")
    print("="*50)

    # Stem plots show vertical lines with dots
    # CORRECT way to show DISCRETE signals
    # plt.plot() shows continuous lines (wrong for discrete)

    n = np.arange(-8, 9)        # -8 to 8
    x = np.zeros(17)
    x[8]  = 1.0    # n=0
    x[9]  = 0.5    # n=1
    x[7]  = 2.0    # n=-1
    x[10] = 1.0    # n=2
    x[6]  = 0.5    # n=-2

    # ----------------------------------------------------------
    # METHOD 1: plt.stem (simple)
    # plt.stem(x_values, y_values)
    # ----------------------------------------------------------
    plt.figure(figsize=(10, 4))
    plt.stem(n, x)
    plt.title("Simple Stem Plot")
    plt.xlabel("n")
    plt.ylabel("x[n]")
    plt.grid(True)

    # ----------------------------------------------------------
    # METHOD 2: ax.stem with customization
    # ax.stem returns three objects:
    #   markerline → the dots at top of stems
    #   stemlines  → the vertical lines
    #   baseline   → the horizontal line at y=0
    # ----------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 4))
    markerline, stemlines, baseline = ax.stem(
        n, x, label="x[n]"
    )
    baseline.set_visible(False)    # hide horizontal baseline
    ax.set_title("Customized Stem Plot")
    ax.set_xlabel("n")
    ax.set_ylabel("x[n]")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ----------------------------------------------------------
    # MULTIPLE STEM PLOTS on same axes
    # Just call ax.stem multiple times
    # ----------------------------------------------------------
    y = x[::-1]   # reversed signal

    fig, ax = plt.subplots(figsize=(10, 4))

    ml1, sl1, bl1 = ax.stem(n, x, label="x[n]")
    bl1.set_visible(False)

    ml2, sl2, bl2 = ax.stem(n, y, label="x[-n]")
    bl2.set_visible(False)

    ax.legend()
    ax.set_title("Two Stem Plots Together")
    ax.set_xlabel("n")
    ax.set_ylabel("Amplitude")
    ax.grid(True, alpha=0.3)

    print("Stem plots created")


# ============================================================
# SECTION 13: MATPLOTLIB - SUBPLOTS (multiple graphs)
# ============================================================

def section13_subplots():
    print("\n" + "="*50)
    print("SECTION 13: SUBPLOTS")
    print("="*50)

    t = np.linspace(-4, 4, 1000)
    x = np.sin(t)
    y = np.cos(t)

    # ----------------------------------------------------------
    # METHOD 1: plt.subplots(rows, cols)
    # Creates a grid of subplots
    # Returns: fig (whole figure), ax (single axes or array of axes)
    # ----------------------------------------------------------

    # Single plot
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(t, x)
    ax.set_title("Single Plot")
    ax.set_xlabel("t")
    ax.grid(True)

    # ----------------------------------------------------------
    # Two plots side by side (1 row, 2 columns)
    # ax is now an array: ax[0] and ax[1]
    # ----------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    # axes[0] → left plot
    # axes[1] → right plot

    axes[0].plot(t, x, color='blue')
    axes[0].set_title("sin(t)")
    axes[0].set_xlabel("t")
    axes[0].grid(True)

    axes[1].plot(t, y, color='red')
    axes[1].set_title("cos(t)")
    axes[1].set_xlabel("t")
    axes[1].grid(True)

    plt.tight_layout()    # fix spacing between subplots

    # ----------------------------------------------------------
    # Two plots stacked (2 rows, 1 column)
    # ----------------------------------------------------------
    fig, axes = plt.subplots(2, 1, figsize=(10, 6))
    # axes[0] → top plot
    # axes[1] → bottom plot

    axes[0].plot(t, x, label="sin(t)")
    axes[0].set_title("Top Plot")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(t, y, color='red', label="cos(t)")
    axes[1].set_title("Bottom Plot")
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()

    # ----------------------------------------------------------
    # ax methods vs plt methods
    # When using fig, ax = plt.subplots():
    #   ax.plot()      instead of plt.plot()
    #   ax.set_title() instead of plt.title()
    #   ax.set_xlabel()instead of plt.xlabel()
    #   ax.set_ylabel()instead of plt.ylabel()
    #   ax.legend()    same
    #   ax.grid()      same
    # ----------------------------------------------------------
    print("Subplot examples created")


# ============================================================
# SECTION 14: MATPLOTLIB - AXIS CONTROL
# ============================================================

def section14_axis_control():
    print("\n" + "="*50)
    print("SECTION 14: AXIS CONTROL")
    print("="*50)

    t = np.linspace(-4, 4, 1000)
    x = np.sin(2 * np.pi * t)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # ----------------------------------------------------------
    # plt.xlim(min, max) or ax.set_xlim(min, max)
    # Sets the visible range of x axis
    # ----------------------------------------------------------
    axes[0, 0].plot(t, x)
    axes[0, 0].set_xlim(-2, 2)       # only show t from -2 to 2
    axes[0, 0].set_title("xlim(-2, 2)")
    axes[0, 0].grid(True)

    # ----------------------------------------------------------
    # plt.ylim(min, max) or ax.set_ylim(min, max)
    # Sets the visible range of y axis
    # ----------------------------------------------------------
    axes[0, 1].plot(t, x)
    axes[0, 1].set_ylim(-0.5, 0.5)   # only show amplitude -0.5 to 0.5
    axes[0, 1].set_title("ylim(-0.5, 0.5)")
    axes[0, 1].grid(True)

    # ----------------------------------------------------------
    # ax.set_xticks([list of values])
    # Controls WHERE the tick marks appear on x axis
    # ----------------------------------------------------------
    axes[1, 0].plot(t, x)
    axes[1, 0].set_xticks([-4, -2, 0, 2, 4])   # tick only at these values
    axes[1, 0].set_title("Custom x ticks")
    axes[1, 0].grid(True)

    # ----------------------------------------------------------
    # np.arange with set_xticks for integer ticks
    # Common for discrete signal plots
    # ----------------------------------------------------------
    n = np.arange(-8, 9)
    sig = np.zeros(17)
    sig[8] = 1.0

    axes[1, 1].stem(n, sig)
    axes[1, 1].set_xticks(np.arange(-8, 9, 1))  # tick at every integer
    axes[1, 1].set_title("Integer ticks (discrete)")
    axes[1, 1].grid(True)

    plt.tight_layout()
    print("Axis control examples created")


# ============================================================
# SECTION 15: MATPLOTLIB - SAVING AND SHOWING
# ============================================================

def section15_save_show():
    print("\n" + "="*50)
    print("SECTION 15: SAVING AND SHOWING")
    print("="*50)

    t = np.linspace(0, 4, 100)
    plt.figure(figsize=(8, 4))
    plt.plot(t, np.sin(t), label="sin(t)")
    plt.title("Example")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # ----------------------------------------------------------
    # plt.savefig("filename.png")
    # Saves figure to a file
    # Must call BEFORE plt.show() (show() clears the figure)
    # ----------------------------------------------------------
    # plt.savefig("my_plot.png")     # save as PNG
    # plt.savefig("my_plot.pdf")     # save as PDF

    # ----------------------------------------------------------
    # plt.tight_layout()
    # Automatically adjusts spacing so labels don't overlap
    # Always call this before show() or savefig()
    # ----------------------------------------------------------
    plt.tight_layout()

    # ----------------------------------------------------------
    # plt.show()
    # Displays ALL currently open figures
    # In exams, this goes at the END of main()
    # ----------------------------------------------------------
    # plt.show()   # commented out to not block script

    print("plt.show() would display all figures")
    print("plt.savefig('name.png') would save to file")

    # ----------------------------------------------------------
    # plt.close()        → close current figure
    # plt.close('all')   → close all figures
    # ----------------------------------------------------------
    plt.close('all')


# ============================================================
# SECTION 16: COMPLETE EXAM-STYLE EXAMPLE
# Puts everything together like a real exam question
# ============================================================

def section16_complete_example():
    print("\n" + "="*50)
    print("SECTION 16: COMPLETE EXAM EXAMPLE")
    print("="*50)

    # ---- CONTINUOUS SIGNAL ----
    T_MIN, T_MAX, N = -4.0, 4.0, 4001
    t = np.linspace(T_MIN, T_MAX, N)

    # Signal: x(t) = sin(2π*0.5*t) + 0.5*sin(2π*1.5*t)
    x = np.sin(2*np.pi*0.5*t) + 0.5*np.sin(2*np.pi*1.5*t)

    # Time reversal: x(-t) = flip the array
    x_rev = x[::-1]

    # Even part: xe(t) = [x(t) + x(-t)] / 2
    xe = (x + x_rev) / 2

    # Odd part: xo(t) = [x(t) - x(-t)] / 2
    xo = (x - x_rev) / 2

    # Verify reconstruction
    reconstruction_error = np.max(np.abs(xe + xo - x))
    print(f"Reconstruction error: {reconstruction_error:.2e}")

    # Time scaling: y(t) = x(t/2)
    k = 2
    t_query = t / k
    valid = (t_query >= T_MIN) & (t_query <= T_MAX)
    dt = t[1] - t[0]
    y_scaled = np.zeros_like(t)

    # Interpolation for valid points
    for i in np.where(valid)[0]:   # np.where returns indices where True
        pos = (t_query[i] - T_MIN) / dt
        left = int(np.floor(pos))
        right = min(left + 1, N - 1)    # min() prevents going out of bounds
        y_scaled[i] = 0.5 * (x[left] + x[right])

    # ---- PLOT EVERYTHING ----
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))

    # Plot 1: x(t) and x(-t)
    axes[0, 0].plot(t, x,     label="x(t)",    linewidth=1.5)
    axes[0, 0].plot(t, x_rev, label="x(-t)",   linewidth=1.5, linestyle='--')
    axes[0, 0].set_title("Time Reversal")
    axes[0, 0].set_xlabel("t")
    axes[0, 0].set_ylabel("Amplitude")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Plot 2: x(t), xe(t), xo(t)
    axes[0, 1].plot(t, x,  label="x(t)",  linewidth=1.5)
    axes[0, 1].plot(t, xe, label="xe(t)", linewidth=1.5, linestyle='--')
    axes[0, 1].plot(t, xo, label="xo(t)", linewidth=1.5, linestyle=':')
    axes[0, 1].set_title("Even-Odd Decomposition")
    axes[0, 1].set_xlabel("t")
    axes[0, 1].set_ylabel("Amplitude")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Plot 3: Time scaling
    axes[1, 0].plot(t, x,        label="x(t)",     linewidth=1.5)
    axes[1, 0].plot(t, y_scaled, label=f"x(t/{k})", linewidth=1.5, linestyle='--')
    axes[1, 0].set_title(f"Time Scaling: y(t) = x(t/{k})")
    axes[1, 0].set_xlabel("t")
    axes[1, 0].set_ylabel("Amplitude")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # ---- DISCRETE SIGNAL ----
    INF = 8
    n_axis = np.arange(-INF, INF + 1)
    sig = np.zeros(2 * INF + 1)
    sig[INF]     = 1.0    # n=0
    sig[INF + 1] = 0.5    # n=1
    sig[INF - 1] = 2.0    # n=-1
    sig[INF + 2] = 1.0    # n=2
    sig[INF - 2] = 0.5    # n=-2

    sig_rev = sig[::-1]
    sig_xe  = (sig + sig_rev) / 2
    sig_xo  = (sig - sig_rev) / 2

    # Plot 4: Discrete even-odd
    ml1, sl1, bl1 = axes[1, 1].stem(n_axis, sig,     label="x[n]")
    bl1.set_visible(False)
    ml2, sl2, bl2 = axes[1, 1].stem(n_axis, sig_xe,  label="xe[n]")
    bl2.set_visible(False)
    ml3, sl3, bl3 = axes[1, 1].stem(n_axis, sig_xo,  label="xo[n]")
    bl3.set_visible(False)
    axes[1, 1].set_title("Discrete Even-Odd")
    axes[1, 1].set_xlabel("n")
    axes[1, 1].set_ylabel("Amplitude")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_xticks(np.arange(-INF, INF+1, 2))

    plt.suptitle("Complete Signal Processing Reference", fontsize=14)
    plt.tight_layout()

    print("Complete example plot created")
    print(f"\nKey values:")
    print(f"  max |xe + xo - x| = {reconstruction_error:.2e}  (should be ~0)")
    print(f"  x at t=0: {x[N//2]:.4f}")
    print(f"  xe at t=0: {xe[N//2]:.4f}")
    print(f"  xo at t=0: {xo[N//2]:.4f}  (should be 0 for odd signal)")


# ============================================================
# MAIN: RUN ALL SECTIONS
# ============================================================

def main():
    section1_creating_arrays()
    section2_indexing()
    section3_operations()
    section4_math()
    section5_statistics()
    section6_manipulation()
    section7_loops()
    section8_functions()
    section9_dictionaries()
    section10_basic_plotting()
    section11_line_styles()
    section12_stem_plots()
    section13_subplots()
    section14_axis_control()
    section15_save_show()
    section16_complete_example()

    print("\n" + "="*50)
    print("ALL SECTIONS COMPLETE")
    print("Showing all figures...")
    print("="*50)

    plt.show()


if __name__ == "__main__":
    main()