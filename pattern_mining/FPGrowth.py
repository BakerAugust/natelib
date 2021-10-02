from typing import List, Optional, Union, Dict
from itertools import chain, combinations
from collections import OrderedDict

from pattern_mining.apriori import find_frequent


def txsort(transaction: tuple, items: List[str]) -> tuple:
    """
    Sorts item in transaction based on their appearance in items.
    """
    return tuple(sorted(transaction, key=lambda x: items.index((x,))))


def powerset(itemset: tuple):
    """
    Returns all combinations of items in itemset of length 1 or more.

    Credit: https://docs.python.org/3/library/itertools.html#itertools-recipes
    """
    return chain.from_iterable(
        combinations(itemset, r) for r in range(1, len(itemset) + 1)
    )


def combine_dicts(dicts: List[dict]) -> dict:
    new_dict: dict = {}
    for d in dicts:
        for k, v in d.items():
            # Accumulate counts
            new_dict[k] = max(v, new_dict.get(k, 0))
    return new_dict


class FPNode:
    """
    Simple node class
    """

    def __init__(self, item: tuple, parent: Optional["FPNode"] = None):
        self.item: tuple = item
        self.count: int = 1
        self.children: List[FPNode] = []
        self.node_link: FPNode = None

        if parent:
            self.parent = parent
            self.level = parent.level + 1
        else:
            self.parent: FPNode = None
            self.level: int = 0

    def increment(self):
        """
        Add one to count
        """
        self.count += 1

    def has_child(self, item: tuple) -> Union[bool, "FPNode"]:
        """
        Checks if there is an immediate child node for item.
        """
        for child in self.children:
            if child.item == item:
                return child
        return False

    def add_child(self, item) -> "FPNode":
        child = FPNode(item, parent=self)
        self.children.append(child)
        return child

    def __repr__(self) -> str:
        s = f"<Class FPNode ({self.item}, {self.count}) "
        s = s + f"level {self.level} "
        if self.parent:
            s = s + f"parent ({self.parent.item}, {self.parent.count}) "
        s = s + f"children = [{[(c.item, c.count) for c in self.children]}] "
        s = s + f"has node_link: {self.node_link != None}>"
        return s


class FPTree:
    """
    Frequent-pattern trees are constructed based on a
    frequent 1-item sets and

    http://hanj.cs.illinois.edu/pdf/dami04_fptree.pdf
    """

    def __init__(self):
        self.root: FPNode = FPNode("null")
        self.frequent_items: OrderedDict = {}

        # shortcut to finding the top node for each item
        self.header_table: Dict[tuple, FPNode] = {}

    def _add_node(self, item: tuple, parent: FPNode) -> FPNode:
        """
        Adds new node, updates node_links appropriately and returns
        """
        child = parent.add_child(item)
        # Check header table
        current_header = self.header_table.get(item, None)

        # Set header if none
        if current_header is None:
            self.header_table[item] = child

        # Insert and displace previous header
        elif child.level < current_header.level:
            self.header_table[item] = child
            child.node_link = current_header

        # Append to end of node_link list
        else:
            while current_header.node_link is not None:
                current_header = current_header.node_link
            current_header.node_link = child

        return child

    def add_transaction(self, transaction: tuple):
        """
        Adds transaction to the FPTree
        """
        # filter infrequent items
        filtered_tx = [i for i in transaction if (i,) in self.items]
        t = txsort(filtered_tx, self.items)
        parent = self.root
        for i in t:
            child = parent.has_child(tuple(i))
            if child:
                child.increment()
            else:
                child = self._add_node((i,), parent)
            # Step down in the tree
            parent = child
        return

    def fit(self, transactions: List[tuple], min_sup: Union[int, float]):
        """
        Build up the FPTree
        """
        if min_sup >= 1:
            self.min_sup = min_sup
        elif min_sup > 0:
            self.min_sup = min_sup * len(transactions)

        # First pass through our data to get len-1 frequent items.
        itemset = set()
        [itemset.update(set(t)) for t in transactions]
        self.items = list(tuple(i) for i in itemset)  # convert back to tuples

        frequent_unsorted, transactions = find_frequent(
            self.items, transactions, min_sup
        )

        # Sort by counts and set items to sorted
        self.frequent_items = OrderedDict(
            sorted(frequent_unsorted.items(), key=lambda x: x[1], reverse=True)
        )
        self.items = list(self.frequent_items.keys())

        for transaction in transactions:
            self.add_transaction(transaction)

    def _mine_single_path(self, node: FPNode) -> dict:
        """
        Finds all items above node in path and returns a
        dictionary of frequent patterns.
        """
        frequent_items = {}
        path_items: tuple = ()
        count = node.count
        while node.item != "null":
            path_items = path_items + node.item
            node = node.parent
        all_combos = powerset(path_items)
        for itemset in all_combos:
            if count > self.min_sup:
                frequent_items[itemset] = count
        return frequent_items

    def mine(self) -> dict:
        """
        Mine the tree for all frequent patterns with recursion.
        """
        frequent_patterns = {}
        for i in reversed(self.items):
            current_node = self.header_table[i]
            # single path
            if not current_node.node_link:
                frequent_patterns = combine_dicts(
                    [self._mine_single_path(current_node), frequent_patterns]
                )

            # multipath
            else:
                # Could replace this with dict for performance. Would need to change .fit().
                pattern_base = []
                prefix = current_node.item
                prefix_count = 0  # Count instances of prefix

                # Traverse all node links
                while current_node is not None:
                    count = current_node.count
                    path = ()
                    node = current_node.parent
                    # Go up each path
                    while node.item != "null":
                        path = path + node.item
                        node = node.parent
                    pattern_base += count * [path]
                    prefix_count += count
                    current_node = current_node.node_link

                # Add in the prefix
                frequent_patterns[prefix] = prefix_count
                # Recursively mine the suffixes
                conditional_tree = FPTree()
                conditional_tree.fit(pattern_base, min_sup=self.min_sup)
                prefix_itemsets = conditional_tree.mine()
                for k, v in prefix_itemsets.items():
                    frequent_patterns[prefix + k] = v
        return frequent_patterns

    def __repr__(self) -> str:
        pass
