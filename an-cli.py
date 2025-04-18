# Logic
import numpy as np
import sympy as smp

# Formatting
import arguably
from rich.console import Console
from rich.pretty import pprint
from rich.table import Table
from rich.progress import Progress


def error(message: str):
    error_console = Console(stderr=True, style="bold red")
    error_console.print(f"ERROR: {message}")
    exit(1)


""" TODO 
2. Newton
3. Bissecção

4. Derivadas e taxas de variação
5. Aprox linear e quadrática

6. Método baseado em quadratura
7. Método baseado em trapézio
8. Método baseado em simpson
"""


@arguably.command
def lagrange(*, xs: list[float], y: list[float], r: float, v: bool = False):
    """
    Computes the Lagrange Interpolating estimation for a given point `r`.

    Args:
      xs: [-x/] list of x-values
      y: [-y/] list of y-values
      r: [-r/--result] x-value to estimate y using the lagrange interpolation function
      v: [-v/--verbose] print steps in detail
    """
    if len(xs) == 1 or len(y) == 1:
        error(f"List length is 1")  # TODO Melhorar mensagem...

    console = Console()

    n = len(xs)
    x = smp.symbols("x")
    y = [smp.Float(f) for f in y]
    px = []
    l_full = []
    l_rational = []

    with Progress() as progress:
        doing_px = progress.add_task(f"[cyan]Calculating P(x)...", total=n)

        for i in range(n):
            l = []
            for j in range(n):
                if xs[j] == xs[i]:
                    continue
                l_calc = (x - xs[j]) / (xs[i] - xs[j])
                l.append(l_calc)
                l_full.append(l_calc)
                l_rational.append(smp.nsimplify(l_calc, rational=True))
            term = y[i] * smp.prod(l)
            px.append(term)

            progress.update(doing_px, advance=1)

        px = smp.Add(*px)

    px_simplified = smp.simplify(px)

    result = px_simplified.subs(x, r)

    if v:
        console.rule("[bold white]DETAILED RESULTS[/bold white]")

        table = Table(show_header=True, header_style="bold white")
        table.add_column("x", style="cyan", justify="center")
        table.add_column("y", style="green", justify="center")
        table.add_column("L", style="magenta", justify="right")
        table.add_column("L (rational)", style="yellow", justify="right")

        for i in range(n):
            table.add_row(
                f"{xs[i]:.9f}",
                f"{y[i]:.9f}",
                f"{l_full[i]}",
                f"{l_rational[i]}",
            )

        console.print(table)

        console.print(f"[bold magenta]P(x)[/bold magenta] = ", end="")
        terms = [
            f"[bold white]({y[i]:.9f} * {l_rational[i]})[/bold white]" for i in range(n)
        ]
        console.print(" + ".join(terms))

        console.print(f"[bold magenta]P(x)[/bold magenta] = {px_simplified}")
        console.print(
            f"[bold magenta]P(x)[/bold magenta] = {smp.nsimplify(px_simplified, rational=True)}"
        )  # FIXME wrong place for logic
        console.print(f"[bold magenta]P({r})[/bold magenta] = {result}")

    else:
        console.print(f"[bold magenta]P(x)[/bold magenta] -> {px_simplified}")
        console.print(f"[bold magenta]P({r})[/bold magenta] -> {result}")


@arguably.command
def lsm(*, xs: list[float], y: list[float], r: float, v: bool = False):
    """
    Computes the linear least squares estimation for a given point `r`.

    Args:
      xs: [-x/] list of x-values
      y: [-y/] list of y-values
      r: [-r/--result] x-value to estimate y using the least squares line
      v: [-v/--verbose] print steps in detail
    """

    if len(xs) == 1 or len(y) == 1:
        error(f"List length is 1")  # TODO Melhorar mensagem...

    console = Console()

    n = len(xs)
    # Numpy arrays are immutable, but faster than a normal list
    xs = np.array(xs, dtype=float)
    y = np.array(y, dtype=float)
    xy, x_squared = [], []

    # Calculate intermediate values
    for i in range(n):
        xy.append(xs[i] * y[i])
        x_squared.append(np.pow(xs[i], 2))

    xy = np.array(xy, dtype=float)
    x_squared = np.array(x_squared, dtype=float)

    # Calculate sums for all arrays
    x_sum = np.sum(xs)
    y_sum = np.sum(y)
    xy_sum = np.sum(xy)
    x_squared_sum = np.sum(x_squared)

    # y = mx + b
    m = (n * xy_sum - x_sum * y_sum) / (n * x_squared_sum - np.pow(x_sum, 2))
    b = (y_sum - m * x_sum) / n
    result = m * r + b  # Estimate y for the given x-value (r)

    if v:
        # Detailed output

        console.rule("[bold white]DETAILED RESULTS[/bold white]")

        table = Table(show_header=True, header_style="bold white")
        table.add_column("", style="bold", justify="center")  # for row labels like ∑
        table.add_column("x", style="cyan", justify="center")
        table.add_column("y", style="green", justify="center")
        table.add_column("x * y", style="magenta", justify="center")
        table.add_column("x^2", style="yellow", justify="center")

        for i in range(len(xs)):
            table.add_row(
                "",  # no label for data rows
                f"{xs[i]:.9f}",
                f"{y[i]:.9f}",
                f"{xy[i]:.9f}",
                f"{x_squared[i]:.9f}",
            )

        table.add_section()
        table.add_row(
            "[bold blue]∑[/bold blue]",
            f"{x_sum:.9f}",
            f"{y_sum:.9f}",
            f"{xy_sum:.9f}",
            f"{x_squared_sum:.9f}",
        )
        console.print(table)

        # Print m, b, and result
        console.print(f"\n[bold blue]m ->[/bold blue] {m}")
        console.print(f"[bold blue]b ->[/bold blue] {b}")
        console.print(
            f"\n[i]y = mx + b[/i] = {m} * {r} + {b} = [underline bold green]{result}[/underline bold green]"
        )
    else:
        console.print(f"[bold magenta]Result: [/bold magenta]{result}")


if __name__ == "__main__":
    arguably.run()
