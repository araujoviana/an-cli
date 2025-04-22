import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
import arguably
from rich.console import Console
from rich.table import Table


# Temporary™ error function
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


@arguably.command  # This decorator makes the function callable from the CLI
def simpson(
    *,
    a: float,
    b: float,
    function: str,
    verbose: bool = False,
):
    """
    Finds an approximation of the integral of the function `f`.

    Args:
      a: [-a/] starting estimate
      b: [-b/] ending estimate
      function: [-f/--function] function to derive
      verbose: [-v/--verbose] print calculation steps in detail
    """

    console = Console()
    x_symb = sp.symbols("x")

    symbolic_function = sp.sympify(function)
    numeric_function = sp.lambdify([x_symb], symbolic_function)

    # Evaluate function values at a, b, and midpoint
    f_at_a = numeric_function(a)
    f_at_b = numeric_function(b)
    f_at_mid = numeric_function((a + b) / 2)

    # Apply Simpson's Rule
    simpson_result = ((b - a) / 6) * (f_at_a + 4 * f_at_mid + f_at_b)

    result = simpson_result

    if verbose:
        console.print("\n")
        console.rule("[bold white]DETAILED RESULTS[/bold white]")

        console.print(
            f"\n[bold yellow]WARNING: May have a small discrepancy due to floating-point precision errors.[/Bold yellow]\n"
        )

        console.print(f"[bold green]f({a}) = [/bold green]{f_at_a}")
        console.print(f"[bold green]f(({a} + {b}) / 2) = [/bold green]{f_at_mid}")
        console.print(f"[bold green]f({b}) = [/bold green]{f_at_b}")

        console.print(f"[bold cyan]Applying the formula:[/bold cyan]")
        console.print(
            f"((b - a) / 6) * (f(a) + 4 * f((a + b) / 2) + f(b)) = {((b - a) / 6)} * ({f_at_a} + 4 * {f_at_mid} + {f_at_b})"
        )

        console.print(f"[bold magenta]Result: [/bold magenta]{result:.9g}")

    else:
        console.print(result)
        return result  # Returning the result makes testing easier


@arguably.command
def differentiate(
    *,
    x: float,
    h: float,
    function: str,
    method: str = "centered",
    v: bool = False,
):
    """
    Finds an approximation of the first derivative at a point `x`.

    Args:
      x: [-x/] point at which to approximate the derivative
      h: [-H/] increment between points in the function (uppercase H!)
      function: [-f/--function] function to differentiate
      method: [-m/--method] differencing method; must be one of: "forward", "backward", or "centered"
      v: [-v/--verbose] print calculation steps in detail
    """

    console = Console()  # rich console
    x_symb = sp.symbols("x")

    # HACK Placeholder values
    symbolic_function = x_symb
    numeric_function = lambda a: a
    numeric_derivative = lambda a: a
    symbolic_derivative = None
    numeric_approx_derivative = None

    try:
        symbolic_function = sp.sympify(function)
    except (sp.SympifyError, TypeError) as e:
        error(f"Could not parse function '{function}': {e}")

    try:
        numeric_function = sp.lambdify(
            [x_symb], symbolic_function, modules="numpy"
        )  # Use numpy
    except Exception as e:
        error(f"Could not create numerical function from '{symbolic_function}': {e}")

    try:
        symbolic_derivative = sp.diff(symbolic_function, x_symb)
        numeric_derivative = sp.lambdify(
            [x_symb], symbolic_derivative, modules="numpy"
        )  # Use numpy
    except Exception as e:
        error(f"Could not differentiate function '{symbolic_function}': {e}")

    numeric_approx_derivative = 0.0
    symbolic_approx_derivative = sp.S(0)  # Singleton registry (?)

    if h <= 0:
        error("h cannot be less or equal to zero.")

    try:
        match method:
            case "forward":
                numeric_approx_derivative = (
                    numeric_function(x + h) - numeric_function(x)
                ) / h
                symbolic_approx_derivative = (
                    symbolic_function.subs(x_symb, x_symb + h) - symbolic_function
                ) / h
            case "backward":
                numeric_approx_derivative = (
                    numeric_function(x) - numeric_function(x - h)
                ) / h
                symbolic_approx_derivative = (
                    symbolic_function - symbolic_function.subs(x_symb, x_symb - h)
                ) / h
            case "centered":
                numeric_approx_derivative = (
                    numeric_function(x + h) - numeric_function(x - h)
                ) / (2 * h)
                symbolic_approx_derivative = (
                    symbolic_function.subs(x_symb, x_symb + h)
                    - symbolic_function.subs(x_symb, x_symb - h)
                ) / (2 * h)
            case _:
                error("Invalid method.")
    except Exception as e:
        error(f"Error evaluating function/derivative near X={x} with h={h}: {e}")

    result = numeric_approx_derivative

    if v:
        console.print("\n")
        console.rule("[bold white]DETAILED RESULTS[/bold white]")

        console.print(
            f"[bold green]Symbolic Approx. f'({x_symb}) = [/bold green]{sp.simplify(symbolic_approx_derivative)}"
        )

        console.print(f"[bold magenta]f(x) = [/bold magenta]{symbolic_function}")
        console.print(
            f"[bold magenta]f'(x) from sympy = [/bold magenta]{symbolic_derivative}"
        )
        try:
            exact_derivative_val = numeric_derivative(x)
            console.print(
                f"[bold magenta]f'({x}) from sympy: [/bold magenta]{exact_derivative_val:.9g}"
            )
        except Exception as e:
            console.print(
                f"[bold red]Could not evaluate sympy derivative at {x}: {e}[/bold red]"
            )

        console.print(
            f"[bold magenta]f'({x}) from {method} differencing: [/bold magenta]{result:.9g}"
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
    verbose: bool = False,
    plot: bool = False,
):
    """
    Approximates the root of the equation `f` using the Bisection method.

    Args:
      a: [-a/] starting estimate
      b: [-b/] ending estimate
      function: [-f/--function] function for finding the root
      criterion: [-c/--criterion] stopping criterion to be used for stopping, can only be: "absolute", "relative" or "function"
      tolerance: [-t/--tolerance] tolerance value that determines when to stop the iteration
      max_iter: [-m/--max] maximum number of iterations
      plot: [-p/--plot] plot points in matplotlib graph
      verbose: [-v/--verbose] print calculation steps in detail
    """

    console = Console()

    x_symb = sp.symbols("x")
    try:
        symbolic_function = sp.sympify(function)
        numeric_function = sp.lambdify(
            [x_symb], symbolic_function, modules="numpy"
        )  # Use numpy
    except (sp.SympifyError, TypeError) as e:
        error(f"Could not parse function '{function}': {e}")
    except Exception as e:
        error(f"Could not create numerical function from '{symbolic_function}': {e}")

    previous_p = 0  # Keep track of the previous p_n for absolute/relative criteria
    f_at_p = 0  # Store f(p_n) for the table
    current_p = (a + b) / 2

    (
        a_values,
        b_values,
        f_a_values,
        p_values,
        f_p_values,
    ) = (
        [],
        [],
        [],
        [],
        [],
    )

    iter_count = 0

    if np.sign(numeric_function(a)) == np.sign(numeric_function(b)):
        error("f(a) and f(b) must have opposite signs.")

    try:
        match criterion:
            case "absolute":
                while (
                    np.abs(current_p - previous_p) > tolerance and iter_count < max_iter
                ):
                    previous_p = current_p
                    f_a = numeric_function(a)
                    f_at_p = numeric_function(previous_p)

                    a_values.append(a)
                    b_values.append(b)
                    f_a_values.append(f_a)
                    p_values.append(current_p)
                    f_p_values.append(f_at_p)

                    if np.sign(f_a) != np.sign(f_at_p):
                        b = previous_p
                    else:
                        a = previous_p

                    current_p = (a + b) / 2
                    iter_count += 1

            case "relative":
                # Initialize p_n differently for the first iteration relative check
                previous_p = current_p + 2 * tolerance  # Ensure first iteration runs
                while (
                    np.abs(current_p - previous_p) / np.abs(current_p) > tolerance
                    and iter_count < max_iter
                ):
                    previous_p = current_p
                    f_a = numeric_function(a)
                    f_at_p = numeric_function(previous_p)

                    a_values.append(a)
                    b_values.append(b)
                    f_a_values.append(f_a)
                    p_values.append(current_p)
                    f_p_values.append(f_at_p)

                    if np.sign(f_a) != np.sign(f_at_p):
                        b = previous_p
                    else:
                        a = previous_p

                    current_p = (a + b) / 2
                    iter_count += 1

            case "function":
                while (
                    np.abs(numeric_function(current_p)) > tolerance
                    and iter_count < max_iter
                ):
                    previous_p = current_p  # For table consistency, store the p we are evaluating
                    f_a = numeric_function(a)
                    f_at_p = numeric_function(previous_p)

                    a_values.append(a)
                    b_values.append(b)
                    f_a_values.append(f_a)
                    p_values.append(current_p)
                    f_p_values.append(f_at_p)

                    if np.sign(f_a) != np.sign(f_at_p):
                        b = previous_p
                    else:
                        a = previous_p

                    current_p = (a + b) / 2
                    iter_count += 1
            case _:
                error(
                    f"Invalid criterion '{criterion}'. Use 'absolute', 'relative', or 'function'."
                )

        # Append the last iteration's state for the table/result
        # Need to calculate f(p) for the final current_p_n
        f_at_final_p = numeric_function(current_p)
        a_values.append(a)
        b_values.append(b)
        f_a_values.append(f_a)
        p_values.append(current_p)
        f_p_values.append(f_at_final_p)  # Use the f(p) of the final result

    except Exception as e:  # Will catch anything i guess
        error(f"Error during bisection iteration: {e}")

    result = current_p

    if verbose:
        console.print("\n")
        console.rule("[bold white]DETAILED RESULTS[/bold white]")

        table = Table(show_header=True, header_style="bold white")
        table.add_column("n", style="cyan", justify="right")
        table.add_column("a_n", style="green", justify="right")
        table.add_column("b_n", style="magenta", justify="right")
        table.add_column("f(a_n)", style="blue", justify="right")
        table.add_column("p_n", style="yellow", justify="right")
        table.add_column("f(p_n)", style="cyan", justify="right")

        # Display n from 1 up to n_iter + 1 (for the final result row)
        for i in range(len(a_values)):
            table.add_row(
                f"{i+1}",
                f"{a_values[i]:.9f}",
                f"{b_values[i]:.9f}",
                f"{f_a_values[i]:.9f}",
                f"{p_values[i]:.9f}",
                f"{f_p_values[i]:.9f}",
            )

        console.print(table)

        if iter_count >= max_iter:
            console.print(
                f"[yellow]Warning: Maximum number of iterations ({max_iter}) reached.[/yellow]"
            )

        console.print(f"\n[bold green]f(x) = [/bold green]{symbolic_function}")
        console.print(
            f"[bold green]|f(p_final)| = [/bold green]{np.abs(f_p_values[-1]):.9g}"
        )
        console.print(
            f"[bold green]Stopping Criterion Met: [/bold green]{criterion} <= {tolerance:.4g}"
        )
        console.print(f"[bold green]Root = [/bold green]{result:.9g}")

    else:
        console.print(result)
        return result

    if plot:
        try:
            x_vals = np.linspace(
                min(a_values[0], b_values[0]) - 1,
                max(a_values[0], b_values[0]) + 1,
                400,
            )
            y_vals = numeric_function(x_vals)

            plt.plot(x_vals, y_vals, label=f"f(x) = {symbolic_function}", color="blue")
            plt.axhline(0, color="black", linewidth=0.5)

            # Plot pn values from the table
            plt.scatter(
                p_values[:-1],
                f_p_values[:-1],
                color="red",
                label="f(p_n) Iterates",
                s=20,
            )  # Smaller points for iterates

            plt.scatter(
                [result],
                [f_p_values[-1]],  # f(result)
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
                min(f_p_values) - 1 if f_p_values else -1,
                max(f_p_values) + 1 if f_p_values else 1,
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
    verbose: bool = False,
    plot: bool = False,
):
    """
    Approximates the root of the equation `f` using the Newton-Raphson method.

    Args:
      estimate: [-e/--estimate] starting estimate for the root
      function: [-f/--function] function for finding the root
      criterion: [-c/--criterion] stopping criterion to be used for stopping, can only be: "absolute", "relative" or "function"
      tolerance: [-t/--tolerance] tolerance value that determines when to stop the iteration
      max_iter: [-m/--max] maximum number of iterations
      plot: [-p/--plot] plot points in matplotlib graph
      verbose: [-v/--verbose] print calculation steps in detail
    """

    console = Console()  # rich console

    x_symb = sp.symbols("x")
    symbolic_function = sp.sympify(function)
    symbolic_derivative = symbolic_function.diff(x_symb)
    numeric_function = sp.lambdify([x_symb], symbolic_function)
    numeric_derivative = sp.lambdify([x_symb], symbolic_derivative)

    (
        previous_estimates,
        previous_function_values,
        previous_derivative_values,
        previous_differences,
        current_estimates,
        current_function_values,
    ) = (
        [],
        [],
        [],
        [],
        [],
        [],
    )

    # initial function calls, *f as in function*
    f_at_estimate = numeric_function(estimate)
    f_derivative_at_estimate = numeric_derivative(estimate)

    try:
        current_estimate = estimate - (f_at_estimate / f_derivative_at_estimate)

        if verbose:
            previous_estimates.append(estimate)
            previous_function_values.append(f_at_estimate)
            previous_derivative_values.append(f_derivative_at_estimate)
            previous_differences.append(f_at_estimate / f_derivative_at_estimate)
            current_estimates.append(current_estimate)
            current_function_values.append(numeric_function(current_estimate))

        current_iter = 0
        match criterion:
            case "absolute":
                while (
                    np.abs(current_estimate - estimate) > tolerance
                    and current_iter <= max_iter
                ):
                    estimate = current_estimate  # Previous estimate

                    f_at_estimate = numeric_function(estimate)
                    f_derivative_at_estimate = numeric_derivative(estimate)

                    current_estimate = estimate - (
                        f_at_estimate / f_derivative_at_estimate
                    )

                    current_iter += 1

                    if verbose:
                        previous_estimates.append(estimate)
                        previous_function_values.append(f_at_estimate)
                        previous_derivative_values.append(f_derivative_at_estimate)
                        previous_differences.append(
                            f_at_estimate / f_derivative_at_estimate
                        )
                        current_estimates.append(current_estimate)
                        current_function_values.append(
                            np.abs(current_estimate - estimate)
                        )

            case "relative":
                while (
                    np.abs(current_estimate - estimate) / np.abs(current_estimate)
                    > tolerance
                    and current_iter <= max_iter
                ):
                    estimate = current_estimate  # Previous estimate

                    f_at_estimate = numeric_function(estimate)
                    f_derivative_at_estimate = numeric_derivative(estimate)

                    current_estimate = estimate - (
                        f_at_estimate / f_derivative_at_estimate
                    )

                    current_iter += 1

                    if verbose:
                        previous_estimates.append(estimate)
                        previous_function_values.append(f_at_estimate)
                        previous_derivative_values.append(f_derivative_at_estimate)
                        previous_differences.append(
                            f_at_estimate / f_derivative_at_estimate
                        )
                        current_estimates.append(current_estimate)
                        current_function_values.append(
                            np.abs(current_estimate - estimate)
                            / np.abs(current_estimate)
                        )

            case "function":
                while (
                    np.abs(numeric_function(current_estimate)) > tolerance
                    and current_iter <= max_iter
                ):
                    estimate = current_estimate  # Previous estimate

                    f_at_estimate = numeric_function(estimate)
                    f_derivative_at_estimate = numeric_derivative(estimate)

                    current_estimate = estimate - (
                        f_at_estimate / f_derivative_at_estimate
                    )

                    current_iter += 1

                    if verbose:
                        previous_estimates.append(estimate)
                        previous_function_values.append(f_at_estimate)
                        previous_derivative_values.append(f_derivative_at_estimate)
                        previous_differences.append(
                            f_at_estimate / f_derivative_at_estimate
                        )
                        current_estimates.append(current_estimate)
                        current_function_values.append(
                            numeric_function(current_estimate)
                        )

    except ZeroDivisionError as e:
        error(f"{e}.")

    result = current_estimate  # for readability

    if verbose:
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

        for i in range(len(current_estimates)):
            table.add_row(
                f"{i}",
                f"{previous_estimates[i]:.9f}",
                f"{previous_function_values[i]:.9f}",
                f"{previous_derivative_values[i]:.9f}",
                f"{previous_differences[i]:.9f}",
                f"{current_estimates[i]:.9f}",
                f"{current_function_values[i]:.9f}",
            )

        console.print(table)

        if current_iter > max_iter:
            console.print("[red]Maximum number of iterations reached![/red]")

        console.print(f"[bold green]f(x) = [/bold green]{symbolic_function}\n")

        # Decompose the expression into terms
        terms = sp.Add.make_args(symbolic_function)
        for term in terms:
            console.print(
                f"[bold green]f'({term}) = [/bold green]{sp.diff(term, x_symb)}"
            )

        console.print(f"[bold green]f'(x) = [/bold green]{symbolic_derivative}")

        pno = sp.symbols("p_n-1")
        console.print(
            f"\n[bold green]p_n = [/bold green] p_n-1 - ({symbolic_function.subs(x_symb, pno)} / {symbolic_derivative.subs(x_symb, pno)})"
        )

        console.print(
            f"[bold magenta]ε: [/bold magenta]{current_function_values[-1]:.9g}"
        )
        console.print(f"[bold magenta]Root: [/bold magenta]{result:.9g}")

    else:
        console.print(result)
        return result

    # plot
    if plot:
        x_vals = np.linspace(current_estimate - 5, current_estimate + 5, 400)
        y_vals = numeric_function(x_vals)

        plt.axhline(0, color="black", linewidth=0.5)

        plt.plot(x_vals, y_vals, label=f"f(x) = {symbolic_function}", color="blue")
        plt.axhline(0, color="black", linewidth=0.5)
        plt.scatter(
            previous_estimates, previous_function_values, color="red", label="Estimates"
        )
        plt.scatter(
            [result],
            [numeric_function(result)],
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
    *,
    xs: list[float],
    y: list[float],
    r: float,
    verbose: bool = False,
    plot: bool = False,
):
    """
    Computes the Lagrange Interpolating estimation for a given point `r`.

    Args:
      xs: [-x/] list of x-values
      y: [-y/] list of y-values
      r: [-r/--result] x-value to estimate y using the lagrange interpolation function
      verbose: [-v/--verbose] print calculation steps in detail
      plot: [-p/--plot] plot points in matplotlib graph
    """
    if len(xs) <= 1 or len(y) <= 1:
        error(f"List length cannot less or equal than 1.")

    console = Console()  # rich console

    # xs represents a list of x-values, just like in functional languages
    xs_length = len(xs)
    x_symb = sp.symbols("x", real=True)  # Symbolic variable for x
    y = [sp.Float(f) for f in y]  # Convert y-values to sympy floats

    px, l_table_real, l_table_rational = (
        [],
        [],
        [],
    )  # Initialize lists for terms and Lagrange factors

    for i in range(xs_length):
        lagrange_factors = []  # Holds Lagrange factors for each i

        if verbose:
            console.print(
                f"\n[bold cyan]Calculating Lagrange factor for i = {i}[/bold cyan]"
            )

        # Calculate Lagrange factors (L_i) for each x[i]
        for j in range(xs_length):
            if xs[j] == xs[i]:  # Avoids subtracting itself
                continue

            l_calc = (x_symb - xs[j]) / (xs[i] - xs[j])

            lagrange_factors.append(l_calc)

            # Used for the verbose table
            l_table_real.append(l_calc)
            l_table_rational.append(sp.nsimplify(l_calc, rational=True))

            if verbose:
                console.print(
                    f"[bold green]L_{i}(x) (step {j}):[/bold green] "
                    f"(x - {xs[j]}) / ({xs[i]} - {xs[j]}) = {l_calc} -> {sp.nsimplify(l_calc, rational=True)}"
                )

        term = y[i] * sp.prod(lagrange_factors)
        px.append(term)

        if verbose:
            console.print(
                f"[bold yellow]Term for i = {i}: [/bold yellow]y[{i}] * L_{i}(x) = {y[i]} * {sp.prod(lagrange_factors)}"
            )

    px = sp.Add(*px)  # Add all terms to get P(x)

    px_simplified = sp.simplify(px)

    # Rational px may be easier to read
    px_rational = sp.nsimplify(px_simplified, rational=True)

    # Evaluate the polynomial at the given x-value (r)
    result = px_simplified.subs(x_symb, r)

    if verbose:
        # Detailed output
        console.print("\n")
        console.rule("[bold white]DETAILED RESULTS[/bold white]")

        table = Table(show_header=True, header_style="bold white")
        table.add_column("x", style="cyan", justify="center")
        table.add_column("f(x)", style="green", justify="center")
        table.add_column("L_n(x)", style="magenta", justify="right")
        table.add_column("L_n(x) (rational)", style="yellow", justify="right")

        for i in range(xs_length):
            table.add_row(
                f"{xs[i]:.9f}",
                f"{y[i]:.9f}",
                f"{l_table_real[i]}",
                f"{l_table_rational[i]}",
            )

        console.print(table)

        # Show the expanded form of P(x)
        console.print(f"[bold magenta]P(x)[/bold magenta] = ", end="")
        terms = [
            f"[bold white]({y[i]:.9f} * {l_table_rational[i]})[/bold white]"
            for i in range(xs_length)
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
    if plot:
        x_line = np.linspace(np.min(xs), np.max(xs), 100)
        px_function = sp.lambdify([x_symb], px_rational, "numpy")
        y_line = px_function(x_line)

        plt.axhline(0, color="black", linewidth=0.5)

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
    verbose: bool = False,
    plot: bool = False,
):
    """
    Computes the linear least squares estimation for a given point `r`.

    Args:
      xs: [-x/] list of x-values
      y: [-y/] list of y-values
      r: [-r/--result] x-value to estimate y using the least squares line
      verbose: [-v/--verbose] print calculation steps in detail
      plot: [-p/--plot] plot points in matplotlib graph
    """

    if len(xs) <= 1 or len(y) <= 1:
        error(f"List length cannot be less or equal than 1.")

    console = Console()

    xs_length = len(xs)
    # Numpy arrays are immutable, but faster than a normal list
    # REVIEW these conversions raise a warning but they work regardless?
    xs = np.array(xs, dtype=float)
    y = np.array(y, dtype=float)
    xy, x_squared = [], []  # x * y and x^2

    # Calculate intermediate values
    for i in range(xs_length):
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
    m = (xs_length * xy_sum - x_sum * y_sum) / (
        xs_length * x_squared_sum - np.pow(x_sum, 2)
    )
    b = (y_sum - m * x_sum) / xs_length

    result = m * r + b  # Estimate y for the given x-value (r)

    if verbose:
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
    if plot:
        x_line = np.linspace(np.min(xs), np.max(xs), 100)
        y_line = m * x_line + b

        plt.axhline(0, color="black", linewidth=0.5)

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
