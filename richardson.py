import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from possion_1d import possion_1d



if __name__ == "__main__":

    start = 0
    end = np.pi
    m = 80
    boundary = [0, 0]

    def q(x):
        return np.ones_like(x)

    def f(x):
        return np.exp(x) * (np.sin(x) - 2 * np.cos(x))

    x, u_num1 = possion_1d(
        q=q, 
        start=start, 
        end=end, 
        m=m, 
        boundary=boundary, 
        f=f, 
        dtype=np.float64
    )


    t = m * 2

    _, u_num2 = possion_1d(
        q=q, 
        start=start, 
        end=end, 
        m=t, 
        boundary=boundary, 
        f=f, 
        dtype=np.float64
    )

    u_num = (4 * u_num2[::2] - u_num1) / 3

    u_exact = np.exp(x) * np.sin(x)

    print(f"{'x':<8} | {'numerical solution':<12}| {'abs error'}")

    for i in range(1, 5):
        idx = i * (m // 5) 
        val_num = u_num[idx]
        val_exact = u_exact[idx]
        error_abs = np.abs(val_num - val_exact)
        
        print(f"{f"{i}π/5":<8} | {val_num:<14.6f}| {error_abs:.6e}")


    sns.lineplot(x=x, y=u_num, label='Numerical Solution', color='gray', linewidth=2)
    sns.lineplot(x=x, y=u_exact, label='Analytical Solution', color='black', linestyle='--', linewidth=2)
    plt.title(f'Solution Comparison (m = {m})')
    plt.xlabel('x')
    plt.ylabel('u(x)')
    plt.legend()
    plt.show()

    error = np.abs(u_exact - u_num)
    print(f"max error: {np.max(error):.5e}")

    sns.lineplot(x=x, y=error, label='Error', color='black', linewidth=2)
    plt.title(f'Error Plot (m = {m})')
    plt.xlabel('x')
    plt.ylabel('Absolute Error')
    plt.legend()
    plt.show()