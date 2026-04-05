import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_theme(style="whitegrid")


def possion_1d(q, f, start, end, m, boundary, dtype=np.float64):
    h = (end - start) / m
    x = np.linspace(start, end, m + 1, dtype=dtype)
    n = m - 1        
    
    q_val = q(x)
    f_val = f(x)
    
    alpha = -1.0 + (h**2 / 12.0) * q_val[:-2]
    beta  =  2.0 + (10.0 * h**2 / 12.0) * q_val[1:-1]
    gamma = -1.0 + (h**2 / 12.0) * q_val[2:]
    
    d = np.zeros(n, dtype=dtype)
    for i in range(1, m):
        d[i-1] = (h**2 / 12.0) * (f_val[i-1] + 10 * f_val[i] + f_val[i+1])
    
    d[0] -= alpha[0] * boundary[0]
    d[-1] -= gamma[-1] * boundary[1]

    w = np.zeros(n, dtype=dtype)
    g = np.zeros(n, dtype=dtype)
    
    w[0] = gamma[0] / beta[0]
    g[0] = d[0] / beta[0]
    
    for i in range(1, n):
        denom = beta[i] - alpha[i] * w[i - 1]
        w[i] = gamma[i] / denom
        g[i] = (d[i] - alpha[i] * g[i - 1]) / denom

    u = np.zeros(m + 1, dtype=dtype)
    u[0] = boundary[0]
    u[-1] = boundary[1]
    
    u[m - 1] = g[-1]
    
    for i in range(n - 2, -1, -1):
        u[i + 1] = g[i] - w[i] * u[i + 2]
        
    return x, u


if __name__ == "__main__":

    start = 0
    end = np.pi
    m = 80
    boundary = [0, 0]

    def q(x):
        return np.ones_like(x)

    def f(x):
        return np.exp(x) * (np.sin(x) - 2 * np.cos(x))

    x, u_num = possion_1d(
        q=q, 
        start=start, 
        end=end, 
        m=m, 
        boundary=boundary, 
        f=f, 
        dtype=np.float64
    )

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