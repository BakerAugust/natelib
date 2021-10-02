from os import truncate
from typing import Tuple, List
from pattern_mining.apriori import find_frequent, apriori
from pattern_mining.FPGrowth import powerset, txsort, FPTree

MIN_SUPS = [2, 3, 4, 5]


def generate_test_data() -> Tuple[List[List[str]], dict]:
    """
    Returns tuple of transactions, counts
    """
    # Make a list of items
    transactions = [
        ("A", "B", "C"),
        ("A", "B", "C", "D"),
        ("A", "D"),
        ("F"),
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
        ("A", "D"): 2,
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
    for min_sup in MIN_SUPS:
        transactions, counts = generate_test_data()
        candidates = [c for c in counts.keys() if len(c) == 1]
        output, _ = find_frequent(candidates, transactions, min_sup)
        for k, v in output.items():
            if v > min_sup:
                assert counts[k] == v


def test_apriori():
    """
    Test the full algorithm output to manually-determined
    counts above.
    """
    for min_sup in MIN_SUPS:
        transactions, counts = generate_test_data()
        l = apriori(transactions, min_sup=min_sup)
        for k, v in l.items():
            if v > min_sup:
                assert counts[k] == v

        # Test for ratio MIN_SUP
        l = apriori(transactions, min_sup=(min_sup / len(transactions)))
        for k, v in l.items():
            if v > min_sup:
                assert counts[k] == v


def test_txsort():
    """ """
    # Test for str
    items = [("One",), ("Two",), ("Three",), ("Four",), ("Five",)]
    transaction = ("Five", "Two", "Four")
    assert txsort(transaction, items) == ("Two", "Four", "Five")

    # Test for ints
    items = [(4,), (9,), (8,), (7,), (5,)]
    transaction = (5, 7, 4)
    assert txsort(transaction, items) == (4, 7, 5)


def test_powerset():
    itemset = (
        "A",
        "B",
        "C",
    )
    true_combos = [
        ("A",),
        ("A", "B"),
        ("A", "C"),
        ("A", "B", "C"),
        ("B",),
        ("B", "C"),
        ("C",),
    ]
    combinations = list(powerset(itemset))
    for c in true_combos:
        assert c in combinations
    assert len(combinations) == len(true_combos)


def test_fpgrowth():
    transactions, counts = generate_test_data()
    fptree = FPTree()
    fptree.fit(transactions, min_sup=3)

    # Check top level
    level_1 = fptree.root.children
    assert len(level_1) == 2

    # Check a node in the top level
    D_node = fptree.header_table[("D",)]
    assert D_node.count == 5
    assert D_node.item == ("D",)
    assert D_node.level == 1
    assert len(D_node.children) == 2
    assert sum([c.count for c in D_node.children]) == 4

    for min_sup in MIN_SUPS:
        transactions, counts = generate_test_data()
        fpt1 = FPTree()
        fpt1.fit(transactions, min_sup)
        l = fpt1.mine()
        for k, v in l.items():
            if v > min_sup:
                assert counts[k] == v

        # Test for ratio MIN_SUP
        fpt2 = FPTree()
        fpt2.fit(transactions, min_sup=(min_sup / len(transactions)))
        l = fpt2.mine()
        for k, v in l.items():
            if v > min_sup:
                assert counts[k] == v


if __name__ == "__main__":
    # transactions, counts = generate_test_data()
    # transactions = [("D",), ("D",), ("D",), ("D",), ("D",)]
    transactions = [("C",), ("C", "D"), ("C", "D")]
    fpt1 = FPTree()
    fpt1.fit(transactions, 2)
    print(fpt1.header_table)
    # print(fpt1.header_table[("B",)].node_link)
    l = fpt1.mine()
    print(l)
