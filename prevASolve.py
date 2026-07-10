import numpy as np
import matplotlib.pyplot as plt

T_MIN, T_MAX, N = -4.0, 4.0, 4001


def x_of_t(t):
    return (
        np.sin(2*np.pi*0.5*t)
        +0.5*np.sin(2*np.pi*1.5*t)
    )


def interpolate_signal(
    t_original,
    x_original,
    t_query
):

    y=np.zeros_like(t_query)

    for i,tq in enumerate(t_query):

        if tq<t_original[0] or tq>t_original[-1]:
            y[i]=0
            continue

        idx=np.where(np.isclose(t_original,tq))[0]

        if len(idx)>0:

            y[i]=x_original[idx[0]]

        else:

            right=np.searchsorted(
                t_original,
                tq
            )

            left=right-1

            y[i]=0.5*(
                x_original[left]
                +
                x_original[right]
            )

    return y


def time_scale(
    t,
    x,
    k
):

    scaled_time=t/k

    y=interpolate_signal(
        t,
        x,
        scaled_time
    )

    return y


def plot_pair(
    t,
    x,
    y,
    title
):

    plt.figure(figsize=(10,5))

    plt.plot(
        t,
        x,
        label="x(t)"
    )

    plt.plot(
        t,
        y,
        label="y(t)=x(t/k)"
    )

    plt.xlabel("Time")

    plt.ylabel("Amplitude")

    plt.title(title)

    plt.grid(True)

    plt.legend()


def main():

    t=np.linspace(
        T_MIN,
        T_MAX,
        N
    )

    x=x_of_t(t)

    k=2

    y=time_scale(
        t,
        x,
        k
    )

    plot_pair(
        t,
        x,
        y,
        f"Time Sub-scaling: y(t)=x(t/{k})"
    )

    plt.show()


if __name__=="__main__":
    main()