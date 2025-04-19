# Logic
import numpy as np
import sympy as smp

# Plotting
import matplotlib.pyplot as plt

# Formatting
import arguably
from rich.console import Console
from rich.table import Table


def error(message: str):
    error_console = Console(stderr=True, style="bold red")
    error_console.print(f"ERROR: {message}")
    exit(1)


""" TODO 
3. Bissecção

4. Derivadas e taxas de variação
5. Aprox linear e quadrática

6. Método baseado em quadratura
7. Método baseado em trapézio
8. Método baseado em simpson
"""


@arguably.command
def bisection(*, a: float, b: float, function: str, criterion: str = "function", tolerance: float = 0.001,
              max_iter: int = 100, v: bool = False, p: bool = False):
    """
    Approximates the root of the equation `f` using the Newton-Raphson method.

    Args:
      a: [-a/] starting estimate
      b: [-b/] ending estimate
      function: [-f/--function] function for finding the root
      criterion: [-c/--criteria] stopping criterion to be used for stopping, can only be: "absolute", "relative" or "function"
      tolerance: [-t/--tolerance] tolerance value that determines when to stop the iteration
      max_iter: [-m/--max] maximum number of iterations
      p: [-p/--plot] plot points in matplotlib graph
      v: [-v/--verbose] print calculation steps in detail
    """

    console = Console()

    x = smp.symbols('x')
    expr = smp.sympify(function)
    expr_f = smp.lambdify([x], expr)
    p_n = 0
    current_p_n = (a + b) / 2

    a_list, b_list, p_list, f_p_list = [], [], [], []

    n_iter = 0

    match criterion:
        case "absolute":
            while np.abs(current_p_n - p_n) > tolerance and n_iter <= max_iter:
                p_n = current_p_n
                f_a = expr_f(a)
                f_p = expr_f(p_n)

                a_list.append(a)
                b_list.append(b)
                p_list.append(current_p_n)
                f_p_list.append(f_p)

                if f_a * f_p < 0:
                    b = p_n
                else:
                    a = p_n

                current_p_n = (a + b) / 2
                n_iter += 1


        case "relative":
            while np.abs(current_p_n - p_n) / np.abs(current_p_n) > tolerance and n_iter <= max_iter:
                p_n = current_p_n
                f_a = expr_f(a)
                f_p = expr_f(p_n)

                a_list.append(a)
                b_list.append(b)
                p_list.append(current_p_n)
                f_p_list.append(f_p)

                if f_a * f_p < 0:
                    b = p_n
                else:
                    a = p_n

                current_p_n = (a + b) / 2
                n_iter += 1

        case "function":
            while np.abs(expr_f(current_p_n)) > tolerance and n_iter <= max_iter:
                p_n = current_p_n
                f_a = expr_f(a)
                f_p = expr_f(p_n)

                a_list.append(a)
                b_list.append(b)
                p_list.append(current_p_n)
                f_p_list.append(f_p)

                if f_a * f_p < 0:
                    b = p_n
                else:
                    a = p_n

                current_p_n = (a + b) / 2
                n_iter += 1

    # this row contains the result
    a_list.append(a)
    b_list.append(b)
    p_list.append(current_p_n)
    f_p_list.append(f_p)

    result = current_p_n # for readability

    if v:
        # Detailed output
        console.print("\n")
        console.rule("[bold white]DETAILED RESULTS[/bold white]")

        table = Table(show_header=True, header_style="bold white")
        table.add_column("n", style="cyan", justify="right")
        table.add_column("a_n", style="green", justify="right")
        table.add_column("b_n", style="magenta", justify="right")
        table.add_column("p_n", style="yellow", justify="right")
        table.add_column("f(p_n)", style="cyan", justify="right")

        for i in range(len(a_list)):
            table.add_row(
                f"{i+1}",
                f"{a_list[i]:.9f}",
                f"{b_list[i]:.9f}",
                f"{p_list[i]:.9f}",
                f"{f_p_list[i]:.9f}",
            )

        console.print(table)

        if n_iter > max_iter:
            console.print("[red]Maximum number of iterations reached![/red]")

        console.print(f"[bold green]f(x) = [/bold green]{expr}\n")
        console.print(f"[bold green]|f(p_n)| = [/bold green]{np.abs(f_p_list[-1])}\n")
        console.print(f"[bold green]Root = [/bold green]{result}")
    else:
        console.print(f"[bold magenta]Result: [/bold magenta]{result}")

    # plot
    if p:
        x_vals = np.linspace(result - 5, result + 5, 400)
        y_vals = expr_f(x_vals)

        plt.plot(x_vals, y_vals, label=f"f(x) = {expr}", color='blue')
        plt.axhline(0, color='black', linewidth=0.5)
        plt.scatter(p_list, f_p_list, color='red', label="f(p_n)")
        plt.scatter([result], [expr_f(result)], color='green', label=f"Root ≈ {result:.4f}", marker='x')
        plt.xlabel("x")
        plt.ylabel("f(x)")
        plt.title("Bisection Method")
        plt.legend()
        plt.grid(True)
        plt.show()


@arguably.command
def newton(*, estimate: float, function: str, criterion: str = "function", tolerance: float = 0.001, max_iter: int = 100, v: bool = False, p: bool = False):
    """
    Approximates the root of the equation `f` using the Newton-Raphson method.

    Args:
      estimate: [-e/--estimate] starting estimate for the root
      function: [-f/--function] function for finding the root
      criterion: [-c/--criteria] stopping criterion to be used for stopping, can only be: "absolute", "relative" or "function"
      tolerance: [-t/--tolerance] tolerance value that determines when to stop the iteration
      max_iter: [-m/--max] maximum number of iterations
      p: [-p/--plot] plot points in matplotlib graph
      v: [-v/--verbose] print calculation steps in detail
    """

    console = Console()

    x = smp.symbols('x')
    expr = smp.sympify(function)
    expr_derivative = expr.diff(x)
    expr_f = smp.lambdify([x], expr)
    expr_derivative_f = smp.lambdify([x], expr_derivative)

    # REVIEW variable names
    p_n_one, f_p_n_one, f_deriv_p_n_one, f_difference, p_n, f_p_n, epsilon = [], [], [], [], [], [], []

    # initial function calls
    f_e = expr_f(estimate)
    f_derivative_e = expr_derivative_f(estimate)

    current_estimate = estimate - (f_e / f_derivative_e)

    if v:
        p_n_one.append(estimate)
        f_p_n_one.append(f_e)
        f_deriv_p_n_one.append(f_derivative_e)
        f_difference.append(f_e / f_derivative_e)
        p_n.append(current_estimate)
        f_p_n.append(expr_f(current_estimate))

    n_iter = 0
    match criterion:
        case "absolute":
            while np.abs(current_estimate - estimate) > tolerance and n_iter <= max_iter:
                estimate = current_estimate # Previous estimate

                f_e = expr_f(estimate)
                f_derivative_e = expr_derivative_f(estimate)

                current_estimate = estimate - (f_e / f_derivative_e)

                n_iter += 1

                if v:
                    p_n_one.append(estimate)
                    f_p_n_one.append(f_e)
                    f_deriv_p_n_one.append(f_derivative_e)
                    f_difference.append(f_e / f_derivative_e)
                    p_n.append(current_estimate)
                    f_p_n.append(np.abs(current_estimate - estimate))

                    

        case "relative":
            while np.abs(current_estimate - estimate) / np.abs(current_estimate) > tolerance and n_iter <= max_iter:
                estimate = current_estimate # Previous estimate

                f_e = expr_f(estimate)
                f_derivative_e = expr_derivative_f(estimate)

                current_estimate = estimate - (f_e / f_derivative_e)

                n_iter += 1

                if v:
                    p_n_one.append(estimate)
                    f_p_n_one.append(f_e)
                    f_deriv_p_n_one.append(f_derivative_e)
                    f_difference.append(f_e / f_derivative_e)
                    p_n.append(current_estimate)
                    f_p_n.append(np.abs(current_estimate - estimate) / np.abs(current_estimate))

        case "function":
            while np.abs(expr_f(current_estimate)) > tolerance and n_iter <= max_iter:
                estimate = current_estimate  # Previous estimate

                f_e = expr_f(estimate)
                f_derivative_e = expr_derivative_f(estimate)

                current_estimate = estimate - (f_e / f_derivative_e)

                n_iter += 1

                if v:
                    p_n_one.append(estimate)
                    f_p_n_one.append(f_e)
                    f_deriv_p_n_one.append(f_derivative_e)
                    f_difference.append(f_e / f_derivative_e)
                    p_n.append(current_estimate)
                    f_p_n.append(expr_f(current_estimate))

    result = current_estimate # for readability

    if v:
        # Detailed output
        console.print("\n")
        console.rule("[bold white]DETAILED RESULTS[/bold white]")

        table = Table(show_header=True, header_style="bold white")
        table.add_column("n-1", style="cyan", justify="right")
        table.add_column("p_n-1", style="green", justify="right")
        table.add_column("f(p_n-1)", style="magenta", justify="right")
        table.add_column("f'(p_n-1)", style="yellow", justify="right")
        table.add_column("f(p_n-1) / f'(p_n-1)", style="cyan", justify="right")
        table.add_column("p(n)", style="green", justify="right")
        table.add_column("f(p_n) -> (ε)", style="magenta", justify="right")

        for i in range(len(p_n)):
            table.add_row(
                f"{i}",
                f"{p_n_one[i]:.9f}",
                f"{f_p_n_one[i]:.9f}",
                f"{f_deriv_p_n_one[i]:.9f}",
                f"{f_difference[i]:.9f}",
                f"{p_n[i]:.9f}",
                f"{f_p_n[i]:.9f}"
            )

        console.print(table)

        if n_iter > max_iter:
            console.print("[red]Maximum number of iterations reached![/red]")

        console.print(f"[bold green]f(x) = [/bold green]{expr}\n")

        # Decompose the expression into terms
        terms = smp.Add.make_args(expr)
        for term in terms:
            console.print(f"[bold green]f'({term}) = [/bold green]{smp.diff(term, x)}")

        console.print(f"[bold green]f'(x) = [/bold green]{expr_derivative}")

        pno = smp.symbols('p_n-1')
        console.print(f"\n[bold green]p_n = [/bold green] p_n-1 - ({expr.subs(x, pno)} / {expr_derivative.subs(x, pno)})")

        console.print(f"[bold magenta]ε: [/bold magenta]{f_p_n[-1]:.9f}")
        console.print(f"[bold magenta]Root: [/bold magenta]{result:.9f}")


    else:
        console.print(f"[bold magenta]Result: [/bold magenta]{result}")

    # plot
    if p:
        x_vals = np.linspace(current_estimate - 5, current_estimate + 5, 400)
        y_vals = expr_f(x_vals)

        plt.plot(x_vals, y_vals, label=f"f(x) = {expr}", color='blue')
        plt.axhline(0, color='black', linewidth=0.5)
        plt.scatter(p_n_one, f_p_n_one, color='red', label="Estimates")
        plt.scatter([result], [expr_f(result)], color='green', label=f"Root ≈ {result:.4f}", marker='x')
        plt.xlabel("x")
        plt.ylabel("f(x)")
        plt.title("Newton-Raphson Method")
        plt.legend()
        plt.grid(True)
        plt.show()



@arguably.command
def lagrange(*, xs: list[float], y: list[float], r: float, v: bool = False, p: bool = False):
    """
    Computes the Lagrange Interpolating estimation for a given point `r`.

    Args:
      xs: [-x/] list of x-values
      y: [-y/] list of y-values
      r: [-r/--result] x-value to estimate y using the lagrange interpolation function
      v: [-v/--verbose] print calculation steps in detail
      p: [-p/--plot] plot points in matplotlib graph
    """
    if len(xs) == 1 or len(y) == 1:
        error(f"List length is 1")  # TODO Improve error message...

    console = Console()

    n = len(xs)
    x = smp.symbols("x", real=True) # Symbolic variable for x
    y = [smp.Float(f) for f in y]  # Convert y-values to sympy floats
    px, l_full, l_rational = [], [], []  # Initialize lists for terms and Lagrange factors


    for i in range(n):
        l = [] # Holds Lagrange factors for each i

        if v:
            console.print(f"\n[bold cyan]Calculating Lagrange factor for i = {i}[/bold cyan]")

        # Calculate Lagrange factors (L_i) for each x[i]
        # \ell_i(x) = \prod_{\substack{0 \leq j \leq n \\ j \neq i}} \frac{x - x_j}{x_i - x_j}
        for j in range(n):
            if xs[j] == xs[i]:
                continue
            l_calc = (x - xs[j]) / (xs[i] - xs[j]) # Lagrange factor for each pair (i, j)

            l.append(l_calc)
            l_full.append(l_calc)
            l_rational.append(smp.nsimplify(l_calc, rational=True))

            if v:
                console.print(f"[bold green]L_{i}(x) (step {j}):[/bold green] "
                              f"(x - {xs[j]}) / ({xs[i]} - {xs[j]}) = {l_calc} -> {smp.nsimplify(l_calc, rational=True)}")

        term = y[i] * smp.prod(l)
        px.append(term)

        if v:
            console.print(
                f"[bold yellow]Term for i = {i}: [/bold yellow]y[{i}] * L_{i}(x) = {y[i]} * {smp.prod(l)}")


    # P(x) = \sum_{i=0}^{n} y_i \ell_i(x)
    px = smp.Add(*px) # Sum all terms to get P(x)

    px_simplified = smp.simplify(px)
    px_rational = smp.nsimplify(px_simplified, rational=True)

    # Evaluate the polynomial at the given x-value (r)
    result = px_simplified.subs(x, r)

    if v:
        # Detailed output
        console.print("\n")
        console.rule("[bold white]DETAILED RESULTS[/bold white]")

        table = Table(show_header=True, header_style="bold white")
        table.add_column("x", style="cyan", justify="center")
        table.add_column("f(x)", style="green", justify="center")
        table.add_column("L_n(x)", style="magenta", justify="right")
        table.add_column("L_n(x) (rational)", style="yellow", justify="right")

        for i in range(n):
            table.add_row(
                f"{xs[i]:.9f}",
                f"{y[i]:.9f}",
                f"{l_full[i]}",
                f"{l_rational[i]}",
            )

        console.print(table)

        # Show the expanded form of P(x)
        console.print(f"[bold magenta]P(x)[/bold magenta] = ", end="")
        terms = [
            f"[bold white]({y[i]:.9f} * {l_rational[i]})[/bold white]" for i in range(n)
        ]
        console.print(" + ".join(terms))

        # Print simplified polynomial and result for P(x) at r
        console.print(f"[bold magenta]P(x)[/bold magenta] = {px_simplified}")
        console.print(
            f"[bold magenta]P(x) rational[/bold magenta] = {px_rational}"
        )
        console.print(f"[bold magenta]P({r})[/bold magenta] = {result}")

    else:
        console.print(f"[bold magenta]P(x)[/bold magenta] -> {px_simplified}")
        console.print(f"[bold magenta]P(x) rational[/bold magenta] -> {px_rational}")
        console.print(f"[bold magenta]P({r})[/bold magenta] -> {result}")

    # Plot for data points and regression line
    if p:
        x_line = np.linspace(np.min(xs), np.max(xs), 100)
        px_f = smp.lambdify([x], px_rational, 'numpy')
        y_line = px_f(x_line)

        plt.plot(x_line, y_line, label=px_rational, color='blue')
        plt.scatter(xs, y, color='red', label="Data Points")
        plt.scatter([r], [result], color='green', label=f"Estimate at x={r}", marker='^')
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title("Lagrange Interpolation")
        plt.legend()
        plt.grid(True)
        plt.show()

@arguably.command
def lsm(*, xs: list[float], y: list[float], r: float, v: bool = False, p: bool = False):
    """
    Computes the linear least squares estimation for a given point `r`.

    Args:
      xs: [-x/] list of x-values
      y: [-y/] list of y-values
      r: [-r/--result] x-value to estimate y using the least squares line
      v: [-v/--verbose] print calculation steps in detail
      p: [-p/--plot] plot points in matplotlib graph
    """

    if len(xs) == 1 or len(y) == 1:
        error(f"List length is 1") # TODO Improve error message...

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

        console.print("\n")
        console.rule("[bold white]DETAILED RESULTS[/bold white]")

        table = Table(show_header=True, header_style="bold white")
        table.add_column("", style="bold", justify="center")  # for row labels like ∑
        table.add_column("x", style="cyan", justify="center")
        table.add_column("f(x)", style="green", justify="center")
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

    # Plot for data points and regression line
    if p:
        x_line = np.linspace(np.min(xs), np.max(xs), 100)
        y_line = m * x_line + b

        plt.plot(x_line, y_line, label="Regression Line", color='blue')
        plt.scatter(xs, y, color='red', label="Data Points")
        plt.scatter([r], [result], color='green', label=f"Estimate at x={r}", marker='^')
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title("Least Squares Regression")
        plt.legend()
        plt.grid(True)
        plt.show()


if __name__ == "__main__":
    arguably.run()
