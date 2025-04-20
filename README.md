# an-cli

A command-line tool providing implementations of common numerical analysis algorithms. Calculate roots, derivatives, integrals, interpolations, and regressions directly from your terminal.

*Non-verbose output is simplified in order to facilitate piping to other commands*

## Features

*   **Root Finding:**
    *   `bisection`: Find roots using the Bisection method.
    *   `newton`: Find roots using the Newton-Raphson method.
*   **Differentiation:**
    *   `differentiate`: Approximate the derivative using forward, backward, or centered difference methods.
*   **Integration:**
    *   `simpson`: Approximate definite integrals using Simpson's rule.
*   **Interpolation & Regression:**
    *   `lagrange`: Perform Lagrange polynomial interpolation.
    *   `lsm`: Perform Linear Least Squares regression.
*   **Output Options:**
    *   `-v`/`--verbose`: Display detailed calculation steps and intermediate results in tables.
    *   `-p`/`--plot`: Generate and display a `matplotlib` plot for relevant methods (Bisection, Newton, Lagrange, LSM).

## Installation

### Prerequisites

*   Python 3.10 or higher
*   `pip` (Python's package installer, usually included with Python)
*   `git` (optional, for cloning the source)


### Option 1: Install from Source

1.  Clone the repository:
    ```bash
    git clone https://github.com/araujoviana/an-cli 
    cd an-cli
    ```

2.  Install the package using pip:
    ```bash
    pip install .
    ```

### Option 2: Install using uv

1.  Clone the repository:
    ```bash
    git clone https://github.com/araujoviana/an-cli 
    cd an-cli
    ```

2.  Install the package using uv pip:
    ```bash
    uv pip install .
    ```
    
`uv` offers a faster and isolated virtual environment, which is ideal for development 🙂 

## Usage

The tool follows the structure `an-cli <command> [options]`.

You can get help for the tool or specific commands:

```bash
an-cli --help
an-cli bisection --help
```

### Examples

**Bisection Method:** Find the root of f(x) = x^3 - x - 1 between 0 and 2.

```bash
an-cli bisection -a 0 -b 2 -f "x**3 - x - 1"
```

Show details and plot:

```bash
an-cli bisection -a 0 -b 2 -f "x**3 - x - 1" -v -p --tolerance 0.0001
```

**Newton-Raphson Method:** Find the root of f(x) = cos(x) - x starting near 0.5.

```bash
an-cli newton -e 0.5 -f "cos(x) - x"
```

Show details using relative error criterion and plot:

```bash
an-cli newton -e 0.5 -f "cos(x) - x" -v -p -c relative -t 1e-6
```

**Differentiation:** Find the derivative of f(x) = sin(x) at x = pi/4 using centered difference with h=0.01.

```bash
an-cli differentiate -x 0.785398 -H 0.01 -f "x ** 2" -m centered -v
```

**Lagrange Interpolation:** Interpolate y at x=2.5 given points (1,1), (2,4), (3,9).

```bash
an-cli lagrange -x 1,2,3 -y 1,4,9 -r 2.5 -v -p
```

**Linear Least Squares:** Find the regression line for points (1, 2), (2, 4.1), (3, 5.9), (4, 8.2) and estimate y at x=2.5.

```bash
an-cli lsm -x 1,2,3,4 -y 2,4.1,5.9,8.2 -r 2.5 -v -p
```

**Simpson's Rule Integration:** Integrate f(x) = x**2 from 0 to 1.

```bash
an-cli simpson -a 0 -b 1 -f "x**2" -v
```

## License

This project is licensed under the MIT License. See the LICENSE file for details.
