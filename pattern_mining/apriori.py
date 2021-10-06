from typing import List, Union
from math import floor


def find_frequent(
    candidates: List[tuple], transactions, min_sup: int, prune_infrequent: bool = True
) -> dict:
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
    if prune_infrequent:
        frequent_transactions = []

    for t in transactions:
        added = False
        for candidate in candidates:
            if all([c in t for c in candidate]):
                counts[candidate] = counts.get(candidate, 0) + 1
                if prune_infrequent and not added:
                    frequent_transactions.append(t)
                    added = True
    # Filter to only frequent
    output = {}
    for k, v in counts.items():
        if v >= min_sup:
            output[k] = v
    if prune_infrequent:
        return output, list(frequent_transactions)

    else:
        return output, transactions


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
            if i1[:-1] == i2[:-1] and (i1[-1] < i2[-1]):
                c = i1 + (i2[-1],)
                # Prune those with infrequent subset
                if not has_infrequent_subset(c, L_minus.keys()):
                    candidates.append(c)
    return candidates


def apriori(
    transactions: List[Union[List[str], tuple]],
    min_sup: Union[int, float],
    prune_infrequent: bool = True,
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
    transactions = [tuple(sorted(t)) for t in transactions]

    # Get unique items
    itemset = set()
    [itemset.update(set(t)) for t in transactions]
    items = []
    for i in itemset:
        if isinstance(i, tuple):
            items.append(i)
        else:
            items.append((i,))
    L0, transactions = find_frequent(
        items, transactions, min_sup, prune_infrequent=prune_infrequent
    )
    L.append(L0)
    max_tx_length = max([len(t) for t in transactions])
    k = 1
    while len(transactions) > 0 and k <= (max_tx_length + 1):
        candidates = find_candidates(k, L[k - 1])
        l, transactions = find_frequent(
            candidates, transactions, min_sup, prune_infrequent=prune_infrequent
        )
        L.append(l)
        k += 1

    output = {}
    for l in L:
        for k, v in l.items():
            output[k] = v
    return output
