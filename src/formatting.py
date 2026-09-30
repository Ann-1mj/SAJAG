"""Small display-formatting helpers shared across app.py and src/briefings.py."""


def format_inr(amount):
    """Format a rupee amount using Indian digit grouping, e.g. 1234567 -> ₹12,34,567."""
    n = int(round(amount))
    sign = "-" if n < 0 else ""
    digits = str(abs(n))

    if len(digits) <= 3:
        grouped = digits
    else:
        last3, rest = digits[-3:], digits[:-3]
        parts = []
        while len(rest) > 2:
            parts.insert(0, rest[-2:])
            rest = rest[:-2]
        if rest:
            parts.insert(0, rest)
        grouped = ",".join(parts) + "," + last3

    return f"{sign}₹{grouped}"


def format_inr_lakh_cr(amount):
    """Format a rupee amount as lakh or crore, auto-choosing the unit: >=1
    crore -> '₹X.XX Cr', otherwise -> '₹X.XX L'. For portfolio/branch-level
    aggregates, where lakh/crore reads more naturally than full digit
    grouping. Individual per-borrower amounts (a single loan/EMI) should
    keep using format_inr - lakh/crore only reads well at aggregate scale.
    """
    cr = amount / 1e7
    if abs(cr) >= 1:
        return f"₹{cr:.2f} Cr"
    lakh = amount / 1e5
    return f"₹{lakh:.2f} L"
