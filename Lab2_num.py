import numpy as np
import matplotlib.pyplot as plt

ksi = np.pi/4


def k1(x):
    return np.sin(x)*np.sqrt(2)

def k2(x):
    return np.cos(x)*np.cos(x)

def q1(x):
    return 1

def q2(x):
    return x * x

def f1(x):
    return np.sin(2*x)

def f2(x):
    return np.cos(x)

def k1_t(x):
    return np.sin(ksi)*np.sqrt(2)

def k2_t(x):
    return np.cos(ksi)*np.cos(ksi)

def q1_t(x):
    return 1

def q2_t(x):
    return ksi * ksi

def f1_t(x):
    return np.sin(2*ksi)

def f2_t(x):
    return np.cos(ksi)

def f1_th(x):
    C1 = (-0.339317603522784)
    C2 = 0.339317603522784
    return C1 * np.exp(x) +C2*np.exp(-x)+1

def f2_th(x):
    C3 = (-0.492041801231950)
    C4 = 1.05607826128678
    tmp=np.pi*x/(2*np.sqrt(2))
    return 8*np.sqrt(2)/(np.pi**2)+C3*np.exp(tmp)+C4*np.exp(-tmp)


def progonka(n, lower_diag, main_diag, upper_diag, f):
    res = [0] * (n + 1)
    a = [0] * (n + 1)
    b = [0] * (n + 1)

    a[1] = upper_diag[0]
    b[1] = f[0]

    for i in range(2, n+1):
        tmp = main_diag[i - 1] - a[i - 1] * lower_diag[i - 1]
        a[i] = upper_diag[i - 1] / tmp
        b[i] = (f[i - 1] + b[i - 1] * lower_diag[i - 1]) / tmp

    res[n] = (f[n] + lower_diag[n] * b[n]) / (1 - lower_diag[n] * a[n])

    for i in range(n-1, -1, -1):
        res[i] = a[i + 1] * res[i + 1] + b[i + 1]

    return res


def calc_a(n, k1, k2):
    res = [0] * (n + 1)
    x = 0
    h = 1. / n
    for i in range(1, n + 1):
        if (x + h) <= ksi:
            res[i] = k1(x + h / 2)
        elif x >= ksi:
            res[i] = k2(x + h / 2)
        else:
            tmp = n * ((ksi - x) / k1(0.5 * (x + ksi)) + (x + h - ksi) / k2(0.5 * (x + h + ksi)))
            res[i] = 1 / tmp
        x += h
    return res


def calc_d(n, q1, q2):
    res = [0] * (n + 1)
    h = 1. / n
    x = h / 2
    for i in range(1, n):
        if x + h <= ksi:
            res[i] = q1(x + h / 2)
        elif x >= ksi:
            res[i] = q2(x + h / 2)
        else:
            tmp = n * ((ksi - x) * q1(0.5 * (x + ksi)) + (x + h - ksi) * q2(0.5 * (x + h + ksi)))
            res[i] = tmp
        x += h
    return res


def calc_f(n, f1, f2):
    res = [0] * (n + 1)
    h = 1. / n
    x = h / 2
    for i in range(1, n):
        if x + h <= ksi:
            res[i] = f1(x + h / 2)
        elif x >= ksi:
            res[i] = f2(x + h / 2)
        else:
            tmp = n * ((ksi - x) * f1(0.5 * (x + ksi)) + (x + h - ksi) * f2(0.5 * (x + h + ksi)))
            res[i] = tmp
        x += h
    return res


def diag(n, k1, k2, q1, q2):
    lower_diag = calc_a(n, k1, k2)
    main_diag = calc_d(n, q1, q2)
    upper_diag = calc_a(n, k1, k2)
    lower_diag[1] *= n * n

    for i in range(2, n + 1):
        lower_diag[i] *= n * n
        upper_diag[i - 1] = lower_diag[i]
        main_diag[i - 1] += lower_diag[i] + lower_diag[i - 1]

    lower_diag[n] = 0
    main_diag[0] = 1
    main_diag[n] = 1
    upper_diag[n] = 0

    return lower_diag, main_diag, upper_diag


def free_comp(n, f1, f2):
    res = calc_f(n, f1, f2)
    res[0] = 1
    res[n] = 0
    return res


def th_func(n):
    res = [1]
    x = 0
    h = 1. / n
    for i in range(n):
        if x + h <= ksi:
            res.append(f1_th(x + h))
        else:
            res.append(f2_th(x + h))
        x += h
    res[n]=0
    return res


def data(is_test, n):
    e = []
    y = [float(i / n) for i in range(n + 1)]
    if is_test:
        res1 = th_func(n)
        lo, ma, up = diag(n, k1_t, k2_t, q1_t, q2_t)
        f = free_comp(n, f1_t, f2_t)
        res2 = progonka(n, lo, ma, up, f)
        for i, j in zip(res1, res2):
            e.append(abs(i-j))
    else:
        lo, ma, up = diag(n, k1, k2, q1, q2)
        f = free_comp(n, f1, f2)
        res1 = progonka(n, lo, ma, up, f)
        lo, ma, up = diag(2 * n, k1, k2, q1, q2)
        f = free_comp(2 * n, f1, f2)
        res2 = progonka(2 * n, lo, ma, up, f)
        res2 = [res2[2 * i] for i in range(n + 1)]
        for i, j in zip(res1, res2):
            e.append(abs(i - j))

    return y, res1, res2, e


def draw(x, res1, res2, is_test):
    fig, ax = plt.subplots()
    fig1, ax1 = plt.subplots()
    y = []
    if is_test:
        print(res1)
        print(res2)
        for i, j in zip(res1, res2):
            y.append(abs(i - j))
        ax.plot(x, res1, label='u(x)')
        ax.plot(x, res2, label='v(x)')
        ax.set_title("Графики аналитического и численного решений")
        ax1.plot(x, y, label='u(x)-v(x)')
        ax1.set_xlabel('X')
        ax1.set_ylabel("u(x)-v(x)")
        ax1.set_title("Разность графиков \nаналитического и численного решений")
        ax1.legend()
    else:
        for i, j in zip(res1, res2):
            y.append(abs(i - j))
        print(res1)
        print(res2)
        ax.plot(x, res1, label='v(x)')
        ax.plot(x, res2, '.', label='v2(x)')
        ax.set_title("Графики численного решения \nс обычным и половинным шагом")
        ax1.plot(x, y, label='v(x)-v2(x)')
        ax1.set_title("Разность графиков численного решения")
        ax1.set_xlabel('X')
        ax1.set_ylabel("v(x)-v2(x")
    ax.set_xlabel('X')
    ax.set_ylabel("V")
    ax.legend()
    ax1.legend()

    plt.show()

# x, res1, res2, e = data(0, 600)
# draw(x, res1, res2, 0)
# print(max(e))