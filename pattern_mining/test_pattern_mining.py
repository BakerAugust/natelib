"""
Test cases for use with pytest and runtime comparison.
"""

import matplotlib.pyplot as plt
import time
from typing import Tuple, List
from pattern_mining.apriori import find_frequent, apriori
from pattern_mining.FPGrowth import powerset, txsort, FPTree
import pandas as pd

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
            if v >= min_sup:
                assert counts[k] == v

        # Test for ratio MIN_SUP
        l = apriori(transactions, min_sup=(min_sup / len(transactions)))
        assert len(l) == len([c for c in counts.values() if c >= min_sup])
        for k, v in l.items():
            if v >= min_sup:
                assert counts[tuple(sorted(k))] == v

    # Test for some strings with len>1
    transactions = [
        ("one",),
        ("one", "two"),
        ("one", "three"),
        ("three", "two"),
        ("two", "one"),
    ]
    l = apriori(transactions, min_sup=2)
    assert l == {("one",): 4, ("two",): 3, ("three",): 2, ("one", "two"): 2}


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
    assert len(D_node.children) == 2
    assert sum([c.count for c in D_node.children]) == 4

    for min_sup in MIN_SUPS:
        transactions, counts = generate_test_data()
        fpt1 = FPTree()
        fpt1.fit(transactions, min_sup)
        l = fpt1.mine()
        assert len(l) == len([c for c in counts.values() if c >= min_sup])
        for k, v in l.items():
            if v >= min_sup:
                assert counts[tuple(sorted(k))] == v

        # Test for ratio MIN_SUP
        # Multiply by 10 to increase number of tranactions
        fpt2 = FPTree()
        fpt2.fit(transactions * 10, min_sup=(min_sup / len(transactions) * 10))
        l = fpt2.mine()
        for k, v in l.items():
            if v >= min_sup:
                assert counts[tuple(sorted(k))] * 10 == v

    # Test for some strings with len>1
    transactions = [
        ("one",),
        ("one", "two"),
        ("one", "three"),
        ("three", "two"),
        ("two", "one"),
    ]
    counts = {("one",): 4, ("two",): 3, ("three",): 2, ("two", "one"): 2}
    fpt3 = FPTree()
    fpt3.fit(transactions, min_sup=2)
    l = fpt3.mine()
    assert len(l) == len(counts)
    for k, v in l.items():
        assert counts[k] == v


def fpgrowth_mine(transactions, min_sup) -> dict:
    """
    Wrapper to make FPGrowth api consistent.
    """
    fpt = FPTree()
    fpt.fit(transactions, min_sup)
    return fpt.mine()


def apriori_basic(transactions, min_sup) -> dict:
    """
    Wrapper to make apriori without tx pruning api consistent.
    """
    return apriori(transactions, min_sup, prune_infrequent=False)


def timetrial(fn, transactions, min_sup) -> float:
    start = time.time()
    fn(
        transactions=transactions,
        min_sup=min_sup,
    )
    end = time.time()
    return end - start


def compare_performance():
    """
    Compare runtime on adult census dataset.
    """
    # Percent of data to run against
    SAMPLE_SIZES = [0.10, 0.25, 0.5, 0.75, 1]

    # Limit the max number of items allowed per transation
    MAX_TX_LENGTH = [1, 3, 5, 7, 9]

    # Differing levels of support
    MIN_SUPS = [0.5, 0.25, 0.10, 0.05, 0.025, 0.01]

    FUNCTIONS = [
        ("apriori_basic", apriori_basic),
        ("apriori_tx_pruning", apriori),
        ("fpgrowth", fpgrowth_mine),
    ]

    # Our data
    adult = pd.read_csv("data/adult.data", header=None)
    adult.drop(
        labels=adult.dtypes[adult.dtypes == "int64"].index.to_list(),
        axis=1,
        inplace=True,
    )
    transactions = adult.values

    # Time as a function of N transactions
    size_trials = [[], [], []]
    sample_sizes_n = []
    for s in SAMPLE_SIZES:
        tx_subset = transactions[: round(len(transactions) * s), :]
        sample_sizes_n.append(len(tx_subset))
        for i, fn in enumerate(FUNCTIONS):
            size_trials[i].append(timetrial(fn[1], tx_subset, min_sup=0.05))

    # Time as a function of TX length
    tx_length_trials = [[], [], []]
    for n_cols in MAX_TX_LENGTH:
        tx_subset = adult[adult.columns[:n_cols]].values
        for i, fn in enumerate(FUNCTIONS):
            tx_length_trials[i].append(timetrial(fn[1], tx_subset, min_sup=0.05))

    # Time as a function of min_sup
    min_sup_trials = [[], [], []]
    tx_subset = transactions[: len(transactions) // 4, :]  # reduce for timeliness
    for min_sup in MIN_SUPS:
        for i, fn in enumerate(FUNCTIONS):
            min_sup_trials[i].append(timetrial(fn[1], tx_subset, min_sup=min_sup))

    # Plot everything
    fig, axs = plt.subplots(3, 1, figsize=(12, 8))
    for i, data in enumerate(size_trials):
        axs[0].plot(sample_sizes_n, data, label=FUNCTIONS[i][0])
    axs[0].set_xlabel("Number of transactions")
    axs[0].legend()
    axs[0].set_ylabel("Time (s)")

    for i, data in enumerate(tx_length_trials):
        axs[1].plot(MAX_TX_LENGTH, data, label=FUNCTIONS[i][0])
    axs[1].set_xlabel("Number of items per transaction")
    axs[1].set_ylabel("Time (s)")

    for i, data in enumerate(min_sup_trials):
        axs[2].plot(MIN_SUPS, data, label=FUNCTIONS[i][0])
    axs[2].set_xlabel("Minimum support (%)")
    axs[2].set_ylabel("Time (s)")

    axs[0].set_title(
        "Comparing selected pattern mining algorithms on Adult Census dataset"
    )
    plt.tight_layout()
    plt.savefig("pattern_mining/algorithm_performance.png")


if __name__ == "__main__":
    compare_performance()
