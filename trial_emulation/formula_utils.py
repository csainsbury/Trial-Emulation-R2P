"""
Formula handling utilities.

Maps R formula-like interfaces to Python/statsmodels compatible formats.
"""

from typing import Union, List, Optional
import re


def parse_formula_like(
    x: Optional[Union[str, List[str]]], add: Optional[object] = None
) -> tuple[str, List[str]]:
    """
    Coerce to a formula string with RHS only.

    Maps R's as_formula() behavior:
    - None or "~1" -> intercept-only formula "1"
    - String with "~" -> extract RHS
    - List of variable names -> join with "+"

    Args:
        x: A formula string, list of variable names, or None
        add: Optional assertion collection (for compatibility with R checkmate)

    Returns:
        Tuple of (formula_string, list_of_variable_names)

    Examples:
        >>> parse_formula_like(None)
        ('1', [])
        >>> parse_formula_like("~ age + sex")
        ('age + sex', ['age', 'sex'])
        >>> parse_formula_like(["age", "sex", "bmi"])
        ('age + sex + bmi', ['age', 'sex', 'bmi'])
    """
    if x is None:
        return "1", []

    if isinstance(x, list):
        if len(x) == 0:
            return "1", []
        # Join variable names with "+"
        formula_str = " + ".join(str(v) for v in x)
        return formula_str, x

    if isinstance(x, str):
        # Check if it's a formula string with "~"
        if "~" in x:
            # Extract RHS (right-hand side)
            parts = x.split("~", 1)
            if len(parts) == 2:
                rhs = parts[1].strip()
                if rhs == "" or rhs == "1":
                    return "1", []
                # Extract variable names from RHS
                vars_list = _extract_vars_from_formula(rhs)
                return rhs, vars_list
            else:
                return "1", []
        else:
            # Treat strings without a leading ``~`` as a right-hand-side formula
            # fragment.  Splitting on whitespace corrupts valid formula strings
            # such as ``followup_time + I(followup_time**2)`` into ``+ + +``.
            rhs = x.strip()
            if rhs == "" or rhs == "1":
                return "1", []
            vars_list = _extract_vars_from_formula(rhs)
            return rhs, vars_list

    raise TypeError(f"Expected str, list, or None, got {type(x)}")


def _extract_vars_from_formula(formula_str: str) -> List[str]:
    """
    Extract variable names from a formula string.

    Handles basic cases like "age + sex", "age + I(sex^2)", etc.
    For complex formulas, this is a simplified parser.
    """
    # Remove whitespace
    formula_str = formula_str.strip()
    if formula_str == "1" or formula_str == "":
        return []

    # Split by "+" and clean up
    parts = re.split(r"\s*\+\s*", formula_str)
    vars_list = []
    for part in parts:
        part = part.strip()
        if part and part != "1":
            # Handle I() transformations - extract the inner variable
            if part.startswith("I(") and part.endswith(")"):
                inner = part[2:-1]
                # Try to extract variable name (before ^, *, etc.)
                var_match = re.match(r"([a-zA-Z_][a-zA-Z0-9_]*)", inner)
                if var_match:
                    vars_list.append(var_match.group(1))
            else:
                # Extract variable name
                var_match = re.match(r"([a-zA-Z_][a-zA-Z0-9_]*)", part)
                if var_match:
                    vars_list.append(var_match.group(1))

    return vars_list


def add_rhs(f1: str, f2: str) -> str:
    """
    Add RHS parts of formulas.

    Combines two formula strings by adding their RHS terms.

    Args:
        f1: First formula string (RHS)
        f2: Second formula string (RHS)

    Returns:
        Combined formula string

    Examples:
        >>> add_rhs("a + b", "c + log(d)")
        'a + b + c + log(d)'
    """
    rhs1 = f1.strip()
    rhs2 = f2.strip()

    if rhs1 == "1":
        return rhs2
    if rhs2 == "1":
        return rhs1

    return f"{rhs1} + {rhs2}"


def rhs_vars(formula_str: str) -> List[str]:
    """
    Get variables from RHS of formula.

    Args:
        formula_str: Formula string (RHS only)

    Returns:
        List of variable names
    """
    _, vars_list = parse_formula_like(formula_str)
    return vars_list
