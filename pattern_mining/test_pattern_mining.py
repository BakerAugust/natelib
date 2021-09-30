from typing import Tuple, List
from pattern_mining.apriori import find_frequent, apriori

MIN_SUP = 3


def generate_test_data() -> Tuple[List[List[str]], dict]:
    # Make a list of items
    transactions = [
        ("A", "B", "C"),
        ("A", "B", "C", "D"),
        ("A", "D"),
        ("D"),
        ("B", "C", "D"),
        ("C", "D"),
    ]

    counts = {
        ("A",): 3,
        ("A", "B"): 2,
        ("A", "C"): 2,
        ("A", "B", "C"): 2,
        ("A", "B", "C", "D"): 1,
        ("B",): 3,
        ("B", "C"): 3,
        ("B", "D"): 2,
        ("B", "C", "D"): 2,
        ("C",): 4,
        ("C", "D"): 3,
        ("D",): 5,
    }

    return transactions, counts


def test_find_frequent():
    """
    Test find_frequent
    """
    transactions, counts = generate_test_data()
    candidates = [c for c in counts.keys() if len(c) == 1]
    output, _ = find_frequent(candidates, transactions, MIN_SUP)
    for k, v in output.items():
        if v > MIN_SUP:
            assert counts[k] == v


def test_apriori():
    """
    Test the full algorithm output to manually-determined
    counts above.
    """
    transactions, counts = generate_test_data()
    l = apriori(transactions, min_sup=MIN_SUP)
    for k, v in l.items():
        if v > MIN_SUP:
            assert counts[k] == v

    # Test for ratio MIN_SUP
    l = apriori(transactions, min_sup=(MIN_SUP / len(transactions)))
    for k, v in l.items():
        if v > MIN_SUP:
            assert counts[k] == v
