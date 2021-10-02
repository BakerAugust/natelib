import random
import matplotlib.pyplot as plt
import time
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
        ("W", "X", "Y", "Z"),
        ("W", "X", "Y", "Z"),
        ("W", "X", "Y", "Z"),
        ("W", "X", "Y", "Z"),
    ]

    counts = {
        ("A",): 3,
        ("A", "B"): 2,
        ("A", "C"): 2,
        ("A", "D"): 2,
        ("A", "B", "C"): 2,
        ("A", "B", "D"): 1,
        ("A", "C", "D"): 1,
        ("A", "B", "C", "D"): 1,
        ("B",): 3,
        ("B", "C"): 3,
        ("B", "D"): 2,
        ("B", "C", "D"): 2,
        ("C",): 4,
        ("C", "D"): 3,
        ("D",): 5,
        ("F",): 1,
        ("W",): 4,
        ("W", "X"): 4,
        ("W", "Y"): 4,
        ("W", "Z"): 4,
        ("W", "X", "Y"): 4,
        ("W", "X", "Z"): 4,
        ("W", "Y", "Z"): 4,
        ("W", "X", "Y", "Z"): 4,
        ("X",): 4,
        ("X", "Y"): 4,
        ("X", "Z"): 4,
        ("X", "Y", "Z"): 4,
        ("Y",): 4,
        ("Y", "Z"): 4,
        ("Z",): 4,
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
        assert len(l) == len([c for c in counts.values() if c > min_sup])
        for k, v in l.items():
            if v > min_sup:
                assert counts[tuple(sorted(k))] == v


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
    assert len(level_1) == 3

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
        assert len(l) == len([c for c in counts.values() if c >= min_sup])
        for k, v in l.items():
            if v > min_sup:
                assert counts[tuple(sorted(k))] == v

        # Test for ratio MIN_SUP
        # Multiply by 10 to increase number of tranactions
        fpt2 = FPTree()
        fpt2.fit(transactions * 10, min_sup=(min_sup / len(transactions) * 10))
        l = fpt2.mine()
        for k, v in l.items():
            if v > min_sup:
                assert counts[tuple(sorted(k))] * 10 == v


def compare_performance():
    SAMPLE_SIZES = [2 ** n for n in range(15)]
    MIN_SUP = 0.25
    transactions, _ = generate_test_data()
    apriori_times = []
    fpgrowth_times = []
    for sample_size in SAMPLE_SIZES:
        samples = random.choices(transactions, k=sample_size)

        start = time.time()
        a = apriori(samples, min_sup=MIN_SUP)
        end = time.time()
        apriori_times.append(end - start)

        start = time.time()
        fpt = FPTree()
        fpt.fit(samples, min_sup=MIN_SUP)
        b = fpt.mine()
        end = time.time()
        fpgrowth_times.append(end - start)

    plt.plot(SAMPLE_SIZES, apriori_times, label="apriori")
    plt.plot(SAMPLE_SIZES, fpgrowth_times, label="fpgrowth")
    plt.legend()
    plt.show()

    print(a)
    print(b)


if __name__ == "__main__":
    compare_performance()
