from typing import List, Union
from math import floor


def find_frequent(candidates: List[tuple], transactions, min_sup: int) -> dict:
    """
    Counts transactions that contain all items in candidate for
    every candidate.

    Includes an *improvement* to traditional Apriori. Transactions
    that contain no frequent itemsets are discarded for future
    consideration. This is acceptable because any transaction
    with no k length frequent itemsets cannot have any k+1 length
    frequent itemsets.

    Returns a tuple of (dict of {frequent tuples: counts}, frequent_transactions,)
    """
    counts = {}
    frequent_transaction_idx = set()
    for i, t in enumerate(transactions):
        for candidate in candidates:
            if all([c in t for c in candidate]):
                counts[candidate] = counts.get(candidate, 0) + 1
                frequent_transaction_idx.add(i)
    # Filter to only frequent
    output = {}
    for k, v in counts.items():
        if v >= min_sup:
            output[k] = v

    return output, [transactions[i] for i in frequent_transaction_idx]


def has_infrequent_subset(candidate: tuple, L_minus: List[tuple]):
    """
    Looks at all k-1 subsets of candidate and returns True if any
    are not keys of L_mins
    """
    for i in range(len(candidate)):
        s = candidate[:i] + candidate[i + 1 :]
        if s not in L_minus:
            return True
    return False


def find_candidates(k: int, L_minus: dict) -> List[str]:
    """
    Finds all length-k combinations of itemsets in L_minus and
    returns those that have only frequent subsets.
    """
    candidates = []
    for i1 in L_minus.keys():
        for i2 in L_minus.keys():
            if i1[: k - 1] == i2[: k - 1] and (i1[-1] > i2[-1]):
                c = (i2[-1],) + i1
                # Prune those with infrequent subset
                if not has_infrequent_subset(c, L_minus.keys()):
                    candidates.append(c)
    return candidates


def apriori(
    transactions: List[Union[List[str], tuple]], min_sup: Union[int, float]
) -> dict:
    """
    Apriori algorithm for mining frequent patterns as first proposed
    by [Agrawal and Srikant (1994)](http://www.vldb.org/conf/1994/P487.PDF).

    Includes an *improvement* to traditional Apriori. Transactions
    that contain no frequent itemsets are discarded for future
    consideration. This is acceptable because any transaction
    with no k length frequent itemsets cannot have any k+1 length
    frequent itemsets.
    """
    if min_sup < 1:
        min_sup = floor(min_sup * len(transactions))

    L: List[str] = []

    # Sort all transaction items
    transactions = [tuple(sorted(t)) for t in transactions]

    # Get unique items
    itemset = set()
    [itemset.update(set(t)) for t in transactions]
    items = list(tuple(i) for i in itemset)  # convert back to tuples

    L0, transactions = find_frequent(items, transactions, min_sup)
    L.append(L0)
    for k in range(1, max([len(t) for t in transactions])):
        candidates = find_candidates(k, L[k - 1])
        l, transactions = find_frequent(candidates, transactions, min_sup)
        L.append(l)

    output = {}
    for l in L:
        for k, v in l.items():
            output[k] = v

    return output
