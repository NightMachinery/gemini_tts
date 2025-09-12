# AGENT.md

This file provides guidance to coding agents when working with code in this repository.

- After editing files, run `black` on them to both format them correctly and to check for syntax errors.

------------------------------------------------------------------------

-   DRY.

    -   Find common patterns in the code that can refactored into shared code.

-   Use dependency injection to improve code flexibility - let components receive their dependencies from outside instead of hardcoding them. For example, pass configurations as arguments or inject service instances through constructors. However, never inconvenience the user. The dependencies must always be optional to provide.

-   Do NOT add comments about what you have changed, e.g., `newly added`. The user uses version control software to manually review the changes.

------------------------------------------------------------------------

# Functions

-   Have any non-obvious function arguments be keyword arguments. Have at most two positional arguments. Use `(pos_arg1, ..., *, kwarg,)` to enforce keyword argument usage.

# Dataclasses

Define dataclasses when you need to return multiple items, not tuples.

# Conditionals

For enum-like conditionals, use explicit matching conditions, and raise an exception on `else` (when it signifies an unknown value).

# `argparse`

## For Boolean flags, use new argparse BooleanOptionalAction:

```
parser.add_argument('--feature', action=argparse.BooleanOptionalAction)

```

## Interpolate the default in the help strings instead of hardcoding it.

------------------------------------------------------------------------
 

