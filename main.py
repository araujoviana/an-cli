import arguably
import numpy as np
from rich import print
from rich.console import Console


@arguably.command
def hello(name: str):
    print(f"Hello, {name}")


@arguably.command
def lsq(x, y, s):
    n = len(x)
    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)
    xy = []
    xsquared = []

    for i in range(n):
        xy.append(x[i] * y[i])
        xsquared.append(np.pow(x[i], 2))

    xy = np.array(xy, dtype=float)
    xsquared = np.array(xsquared, dtype=float)

    xsum = np.sum(x)
    ysum = np.sum(y)
    xysum = np.sum(xy)
    xsquaredsum = np.sum(xsquared)

    m = (n * xysum - xsum * ysum) / (n * xsquaredsum - np.pow(xsum, 2))

    b = (ysum - m * xsum) / n

    return m * s + b


if __name__ == "__main__":
    # arguably.run()
    console = Console()
    resultado = lsq([1, 2, 3, 4, 5, 6, 7], [1.5, 3.8, 6.7, 9.0, 11.2, 13.6, 16], 2)
    console.print(f"[bold magenta]Resultado: [/bold magenta]{resultado}")
