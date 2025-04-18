import arguably
import numpy as np
from rich.console import Console
from rich.pretty import pprint
from rich.table import Table


def error(message: str):
    error_console = Console(stderr=True, style="bold red")
    error_console.print(f"ERROR: {message}")
    exit(1)


@arguably.command
def lsm(*, x: list[float], y: list[float], r: float, v: bool = False):
    """
    Computes the linear least squares estimation for a given point `r`.

    Args:
      x: [-x/] list of x-values
      y: [-y/] list of y-values
      r: [-r/--result] x-value to estimate y using the least squares line
      v: [-v/--verbose] print steps in detail
    """

    if len(x) == 1 or len(y) == 1:
        error(f"List length is 1")  # TODO Melhorar mensagem...

    console = Console()

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

    result = m * r + b

    if v:
        console.rule("[bold white]DETAILED RESULTS[/bold white]")

        table = Table(show_header=True, header_style="bold white")
        table.add_column("", style="bold", justify="center")  # for row labels like ∑
        table.add_column("x", style="cyan", justify="right")
        table.add_column("y", style="green", justify="right")
        table.add_column("x * y", style="magenta", justify="right")
        table.add_column("x^2", style="yellow", justify="right")

        for i in range(len(x)):
            table.add_row(
                "",  # no label for data rows
                f"{x[i]:.9f}",
                f"{y[i]:.9f}",
                f"{xy[i]:.9f}",
                f"{xsquared[i]:.9f}",
            )

        table.add_section()
        table.add_row(
            "[bold blue]∑[/bold blue]",
            f"{xsum:.9f}",
            f"{ysum:.9f}",
            f"{xysum:.9f}",
            f"{xsquaredsum:.9f}",
        )
        console.print(table)

        console.print(f"\n[bold blue]m ->[/bold blue] {m}")
        console.print(f"[bold blue]b ->[/bold blue] {b}")
        console.print(f"\n[i]y = mx + b[/i] = {m} * {r} + {b} = [underline bold green]{result}[/underline bold green]")
    else:
        console.print(f"[bold magenta]Result: [/bold magenta]{result}")


if __name__ == "__main__":
    arguably.run()
