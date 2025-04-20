import numpy as np
import sympy as smp
import matplotlib.pyplot as plt
import arguably
from rich.console import Console
from rich.table import Table


def error(message: str):
    error_console = Console(stderr=True, style="bold red")
    error_console.print(f"ERROR: {message}")
    exit(1)


""" TODO

4. Derivatives and rates of change
5. Linear and quadratic approximation

6. Quadrature-based method
7. Trapezoidal-based method
"""


@arguably.command
def simpson(
    *,
    a: float,
    b: float,
    function: str,
    v: bool = False,
):
    """
    Finds an approximation of the integral of the function `f`. WARNING: May have a small discrepancy due to floating-point precision errors.

    Args:
      a: [-a/] starting estimate
      b: [-b/] ending estimate
      function: [-f/--function] function to derive
      v: [-v/--verbose] print calculation steps in detail
    """

    console = Console()
    x = smp.symbols("x")

    expr = smp.sympify(function)
    expr_f = smp.lambdify([x], expr)

    f_a = expr_f(a)
    f_b = expr_f(b)
    f_mid = expr_f((a + b) / 2)
    simpson_rule = ((b - a) / 6) * (f_a + 4 * f_mid + f_b)

    result = simpson_rule

    if v:
        console.print("\n")
        console.rule("[bold white]DETAILED RESULTS[/bold white]")

        console.print(f"[bold green]f({a}) = [/bold green]{f_a}")
        console.print(f"[bold green]f(({a} + {b}) / 2) = [/bold green]{f_mid}")
        console.print(f"[bold green]f({b}) = [/bold green]{f_b}")

        console.print(f"[bold cyan]Applying the formula:[/bold cyan]")
        console.print(
            f"((b - a) / 6) * (f(a) + 4 * f((a + b) / 2) + f(b)) = {((b - a) / 6)} * ({f_a} + 4 * {f_mid} + {f_b})"
        )

        console.print(f"[bold magenta]Result: [/bold magenta]{result:.9g}")

    else:
        console.print(result)
        return result


# REVIEW bad grammar @ docstring
@arguably.command
def differentiate(
    *,
    X: float,
    h: float,
    function: str,
    method: str = "centered",
    v: bool = False,
):
    """
    Finds an approximation of the first derivative of a number `x`.

    Args:
      X: [-x/] value to find the derivative
      h: [-H/] increment between points in the function (UPPERCASE H!!)
      function: [-f/--function] function to derive
      method: [-m/--method] differencing method, can only be: "forward", "backward", "centered"
      v: [-v/--verbose] print calculation steps in detail
    """

    console = Console()
    x = smp.symbols("x")

    # HACK Placeholder values
    expr = x
    expr_f = lambda a: a
    expr_deriv_r = lambda a: a
    expr_deriv = None
    expr_deriv_f = None

    try:
        expr = smp.sympify(function)
    except (smp.SympifyError, TypeError) as e:
        error(f"Could not parse function '{function}': {e}")

    try:
        expr_f = smp.lambdify([x], expr, modules="numpy")  # Use numpy
    except Exception as e:
        error(f"Could not create numerical function from '{expr}': {e}")

    try:
        expr_deriv = smp.diff(expr, x)
        expr_deriv_r = smp.lambdify([x], expr_deriv, modules="numpy")  # Use numpy
    except Exception as e:
        error(f"Could not differentiate function '{expr}': {e}")

    expr_deriv_f = 0.0
    expr_deriv_s = smp.S(0)

    if h <= 0:
        error("h cannot be less or equal to zero.")

    try:
        match method:
            case "forward":
                expr_deriv_f = (expr_f(X + h) - expr_f(X)) / h
                expr_deriv_s = (expr.subs(x, x + h) - expr) / h
            case "backward":
                expr_deriv_f = (expr_f(X) - expr_f(X - h)) / h
                expr_deriv_s = (expr - expr.subs(x, x - h)) / h
            case "centered":
                expr_deriv_f = (expr_f(X + h) - expr_f(X - h)) / (2 * h)
                expr_deriv_s = (expr.subs(x, x + h) - expr.subs(x, x - h)) / (2 * h)
            case _:
                error("Invalid method.")
    except Exception as e:
        error(f"Error evaluating function/derivative near X={X} with h={h}: {e}")

    result = expr_deriv_f

    if v:
        console.print("\n")
        console.rule("[bold white]DETAILED RESULTS[/bold white]")

        console.print(
            f"[bold green]Symbolic Approx. f'({x}) = [/bold green]{smp.simplify(expr_deriv_s)}"
        )

        console.print(f"[bold magenta]f(x) = [/bold magenta]{expr}")
        console.print(f"[bold magenta]f'(x) from sympy = [/bold magenta]{expr_deriv}")
        try:
            actual_deriv_val = expr_deriv_r(X)
            console.print(
                f"[bold magenta]f'({X}) from sympy: [/bold magenta]{actual_deriv_val:.9g}"
            )
        except Exception as e:
            console.print(
                f"[bold red]Could not evaluate sympy derivative at {X}: {e}[/bold red]"
            )

        console.print(
            f"[bold magenta]f'({X}) from {method} differencing: [/bold magenta]{result:.9g}"
        )

    else:
        console.print(f"{result:.9g}")  # Format non-verbose output
        return result


@arguably.command
def bisection(
    *,
    a: float,
    b: float,
    function: str,
    criterion: str = "function",
    tolerance: float = 0.001,
    max_iter: int = 100,
    v: bool = False,
    p: bool = False,
):
    """
    Approximates the root of the equation `f` using the Bisection method.

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

    x = smp.symbols("x")
    try:
        expr = smp.sympify(function)
        expr_f = smp.lambdify([x], expr, modules="numpy")  # Use numpy
    except (smp.SympifyError, TypeError) as e:
        error(f"Could not parse function '{function}': {e}")
    except Exception as e:
        error(f"Could not create numerical function from '{expr}': {e}")

    # HACK Placeholder values
    expr = x
    expr_f = lambda a: a

    f_a_init = expr_f(a)
    f_b_init = expr_f(b)

    p_n = 0  # Keep track of the previous p_n for absolute/relative criteria
    f_p = 0  # Store f(p_n) for the table
    current_p_n = (a + b) / 2

    a_list, b_list, p_list, f_p_list = [], [], [], []

    n_iter = 0

    try:
        match criterion:
            case "absolute":
                while np.abs(current_p_n - p_n) > tolerance and n_iter < max_iter:
                    p_n = current_p_n
                    f_a = expr_f(a)
                    f_p = expr_f(p_n)

                    a_list.append(a)
                    b_list.append(b)
                    p_list.append(current_p_n)
                    f_p_list.append(f_p)

                    if np.sign(f_a) != np.sign(f_p):
                        b = p_n
                    else:
                        a = p_n

                    current_p_n = (a + b) / 2
                    n_iter += 1

            case "relative":
                # Initialize p_n differently for the first iteration relative check
                p_n = current_p_n + 2 * tolerance  # Ensure first iteration runs
                while (
                    np.abs(current_p_n)
                    > 1e-12  # Avoid division by zero if root is near 0
                    and np.abs(current_p_n - p_n) / np.abs(current_p_n) > tolerance
                    and n_iter < max_iter
                ):
                    p_n = current_p_n
                    f_a = expr_f(a)
                    f_p = expr_f(p_n)

                    a_list.append(a)
                    b_list.append(b)
                    p_list.append(current_p_n)
                    f_p_list.append(f_p)

                    if np.sign(f_a) != np.sign(f_p):
                        b = p_n
                    else:
                        a = p_n

                    current_p_n = (a + b) / 2
                    n_iter += 1

            case "function":
                while np.abs(expr_f(current_p_n)) > tolerance and n_iter < max_iter:
                    p_n = current_p_n  # For table consistency, store the p we are evaluating
                    f_a = expr_f(a)
                    f_p = expr_f(p_n)

                    a_list.append(a)
                    b_list.append(b)
                    p_list.append(current_p_n)
                    f_p_list.append(f_p)

                    if np.sign(f_a) != np.sign(f_p):
                        b = p_n
                    else:
                        a = p_n

                    current_p_n = (a + b) / 2
                    n_iter += 1
            case _:
                error(
                    f"Invalid criterion '{criterion}'. Use 'absolute', 'relative', or 'function'."
                )

        # Append the last iteration's state for the table/result
        # Need to calculate f(p) for the final current_p_n
        f_p_final = expr_f(current_p_n)
        a_list.append(a)
        b_list.append(b)
        p_list.append(current_p_n)
        f_p_list.append(f_p_final)  # Use the f(p) of the final result

    except Exception as e:
        error(f"Error during bisection iteration: {e}")

    result = current_p_n

    if v:
        console.print("\n")
        console.rule("[bold white]DETAILED RESULTS[/bold white]")

        table = Table(show_header=True, header_style="bold white")
        table.add_column("n", style="cyan", justify="right")
        table.add_column("a_n", style="green", justify="right")
        table.add_column("b_n", style="magenta", justify="right")
        table.add_column("p_n", style="yellow", justify="right")
        table.add_column("f(p_n)", style="cyan", justify="right")

        # Display n from 1 up to n_iter + 1 (for the final result row)
        for i in range(len(a_list)):
            table.add_row(
                f"{i+1}",
                f"{a_list[i]:.9f}",
                f"{b_list[i]:.9f}",
                f"{p_list[i]:.9f}",
                f"{f_p_list[i]:.9f}",
            )

        console.print(table)

        if n_iter >= max_iter:
            console.print(
                f"[yellow]Warning: Maximum number of iterations ({max_iter}) reached.[/yellow]"
            )

        console.print(f"\n[bold green]f(x) = [/bold green]{expr}")
        console.print(
            f"[bold green]|f(p_final)| = [/bold green]{np.abs(f_p_list[-1]):.9g}"
        )
        console.print(
            f"[bold green]Stopping Criterion Met: [/bold green]{criterion} <= {tolerance:.4g}"
        )
        console.print(f"[bold green]Root = [/bold green]{result:.9g}")
    else:
        console.print(result)
        return result

    if p:
        try:
            x_vals = np.linspace(
                min(a_list[0], b_list[0]) - 1, max(a_list[0], b_list[0]) + 1, 400
            )
            y_vals = expr_f(x_vals)

            plt.plot(x_vals, y_vals, label=f"f(x) = {expr}", color="blue")
            plt.axhline(0, color="black", linewidth=0.5)
            # Plot pn values from the table
            plt.scatter(
                p_list[:-1], f_p_list[:-1], color="red", label="f(p_n) Iterates", s=20
            )  # Smaller points for iterates
            plt.scatter(
                [result],
                [f_p_list[-1]],  # f(result)
                color="green",
                label=f"Final Root ≈ {result:.4f}",
                marker="x",
                s=100,  # Larger marker for result
            )
            plt.xlabel("x")
            plt.ylabel("f(x)")
            plt.title("Bisection Method")
            plt.legend()
            plt.grid(True)
            plt.ylim(
                min(f_p_list) - 1 if f_p_list else -1,
                max(f_p_list) + 1 if f_p_list else 1,
            )  # Adjust ylim based on points
            plt.show()
        except Exception as e:
            error(f"Error generating plot: {e}")


@arguably.command
def newton(
    *,
    estimate: float,
    function: str,
    criterion: str = "function",
    tolerance: float = 0.001,
    max_iter: int = 100,
    v: bool = False,
    p: bool = False,
):
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

    x = smp.symbols("x")
    expr = smp.sympify(function)
    expr_derivative = expr.diff(x)
    expr_f = smp.lambdify([x], expr)
    expr_derivative_f = smp.lambdify([x], expr_derivative)

    # REVIEW variable names
    (
        p_n_one,
        f_p_n_one,
        f_deriv_p_n_one,
        f_difference,
        p_n,
        f_p_n,
    ) = (
        [],
        [],
        [],
        [],
        [],
        [],
    )

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
            while (
                np.abs(current_estimate - estimate) > tolerance and n_iter <= max_iter
            ):
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
                    f_p_n.append(np.abs(current_estimate - estimate))

        case "relative":
            while (
                np.abs(current_estimate - estimate) / np.abs(current_estimate)
                > tolerance
                and n_iter <= max_iter
            ):
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
                    f_p_n.append(
                        np.abs(current_estimate - estimate) / np.abs(current_estimate)
                    )

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

    result = current_estimate  # for readability

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
                f"{f_p_n[i]:.9f}",
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

        pno = smp.symbols("p_n-1")
        console.print(
            f"\n[bold green]p_n = [/bold green] p_n-1 - ({expr.subs(x, pno)} / {expr_derivative.subs(x, pno)})"
        )

        console.print(f"[bold magenta]ε: [/bold magenta]{f_p_n[-1]:.9g}")
        console.print(f"[bold magenta]Root: [/bold magenta]{result:.9g}")

    else:
        console.print(result)
        return result

    # plot
    if p:
        x_vals = np.linspace(current_estimate - 5, current_estimate + 5, 400)
        y_vals = expr_f(x_vals)

        plt.plot(x_vals, y_vals, label=f"f(x) = {expr}", color="blue")
        plt.axhline(0, color="black", linewidth=0.5)
        plt.scatter(p_n_one, f_p_n_one, color="red", label="Estimates")
        plt.scatter(
            [result],
            [expr_f(result)],
            color="green",
            label=f"Root ≈ {result:.4f}",
            marker="x",
        )
        plt.xlabel("x")
        plt.ylabel("f(x)")
        plt.title("Newton-Raphson Method")
        plt.legend()
        plt.grid(True)
        plt.show()


@arguably.command
def lagrange(
    *, xs: list[float], y: list[float], r: float, v: bool = False, p: bool = False
):
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
    x = smp.symbols("x", real=True)  # Symbolic variable for x
    y = [smp.Float(f) for f in y]  # Convert y-values to sympy floats
    px, l_full, l_rational = (
        [],
        [],
        [],
    )  # Initialize lists for terms and Lagrange factors

    for i in range(n):
        l = []  # Holds Lagrange factors for each i

        if v:
            console.print(
                f"\n[bold cyan]Calculating Lagrange factor for i = {i}[/bold cyan]"
            )

        # Calculate Lagrange factors (L_i) for each x[i]
        # \ell_i(x) = \prod_{\substack{0 \leq j \leq n \\ j \neq i}} \frac{x - x_j}{x_i - x_j}
        for j in range(n):
            if xs[j] == xs[i]:
                continue
            l_calc = (x - xs[j]) / (
                xs[i] - xs[j]
            )  # Lagrange factor for each pair (i, j)

            l.append(l_calc)
            l_full.append(l_calc)
            l_rational.append(smp.nsimplify(l_calc, rational=True))

            if v:
                console.print(
                    f"[bold green]L_{i}(x) (step {j}):[/bold green] "
                    f"(x - {xs[j]}) / ({xs[i]} - {xs[j]}) = {l_calc} -> {smp.nsimplify(l_calc, rational=True)}"
                )

        term = y[i] * smp.prod(l)
        px.append(term)

        if v:
            console.print(
                f"[bold yellow]Term for i = {i}: [/bold yellow]y[{i}] * L_{i}(x) = {y[i]} * {smp.prod(l)}"
            )

    # P(x) = \sum_{i=0}^{n} y_i \ell_i(x)
    px = smp.Add(*px)  # Sum all terms to get P(x)

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
        console.print(f"[bold magenta]P(x) rational[/bold magenta] = {px_rational}")
        console.print(f"[bold magenta]P({r})[/bold magenta] = {result}")

    else:
        console.print(result)
        return result

    # Plot for data points and regression line
    if p:
        x_line = np.linspace(np.min(xs), np.max(xs), 100)
        px_f = smp.lambdify([x], px_rational, "numpy")
        y_line = px_f(x_line)

        plt.plot(x_line, y_line, label=px_rational, color="blue")
        plt.scatter(xs, y, color="red", label="Data Points")
        plt.scatter(
            [r], [result], color="green", label=f"Estimate at x={r}", marker="^"
        )
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title("Lagrange Interpolation")
        plt.legend()
        plt.grid(True)
        plt.show()


@arguably.command
def lsm(
    *,
    xs: list[float],
    y: list[float],
    r: float,
    v: bool = False,
    p: bool = False,
):
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
        error(f"List length is 1")  # TODO Improve error message...

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
        console.print(result)
        return result

    # Plot for data points and regression line
    if p:
        x_line = np.linspace(np.min(xs), np.max(xs), 100)
        y_line = m * x_line + b

        plt.plot(x_line, y_line, label="Regression Line", color="blue")
        plt.scatter(xs, y, color="red", label="Data Points")
        plt.scatter(
            [r], [result], color="green", label=f"Estimate at x={r}", marker="^"
        )
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title("Least Squares Regression")
        plt.legend()
        plt.grid(True)
        plt.show()


def main():
    arguably.run()


if __name__ == "__main__":
    main()
