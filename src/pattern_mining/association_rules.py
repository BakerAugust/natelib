from dataclasses import dataclass
from pattern_mining.FPGrowth import powerset


@dataclass
class AssociationRule:
    """
    Association rules in the form of
    A --> B.
    """

    A: tuple
    B: tuple
    support: int
    confidence: float


def tuple_subtraction(tuple_a: tuple, tuple_b: tuple) -> tuple:
    """
    Returns a tuple with all elements in tuple_a that are not
    in tuple_b.
    """
    output = tuple()
    for t in tuple_a:
        if t not in tuple_b:
            output = output + (t,)
    return output


def mine_association_rules(frequent_patterns: dict, min_confidence: float) -> dict:
    """
    Finds associate rules
    """
    # Need to sort patterns to ensure correct lookup
    sorted_patterns = {}
    for k, v in frequent_patterns.items():
        sorted_patterns[tuple(sorted(k))] = v

    association_rules = []
    for fp, support in sorted_patterns.items():
        if len(fp) > 1:
            for sub_pattern in powerset(fp, max_size=len(fp) - 1):
                confidence = support / sorted_patterns[tuple(sorted(sub_pattern))]
                if confidence >= min_confidence:
                    rule = AssociationRule(
                        A=sub_pattern,
                        B=tuple_subtraction(fp, sub_pattern),
                        support=support,
                        confidence=confidence,
                    )
                    association_rules.append(rule)
    return association_rules
