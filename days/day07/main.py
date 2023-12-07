from collections import Counter

CARDS_P1 = ["A", "K", "Q", "J", "T", "9", "8", "7", "6", "5", "4", "3", "2"]
CARDS_P2 = ["A", "K", "Q", "T", "9", "8", "7", "6", "5", "4", "3", "2", "J"]

def hand_type(hand: str, handle_jokers: bool) -> int:
    counter = dict(Counter(hand))
    if handle_jokers:
        jokers = counter.get("J", 0)
        if 0 < jokers < 5:
            counter.pop("J")
            card = max(counter, key=counter.get)
            counter[card] += jokers
    counts = tuple(sorted(counter.values(), reverse=True))
    if counts == (5, ): # Five of a kind
        return 1 
    elif counts == (4, 1): # Four of a kind
        return 2 
    elif counts == (3, 2): # Full house
        return 3 
    elif counts == (3, 1, 1):  # Three of a kind
        return 4   
    elif counts == (2, 2, 1):  # Two pair
        return 5
    elif counts == (2, 1, 1, 1):  # One pair
        return 6
    elif counts == (1, 1, 1, 1, 1):  # High card
        return 7

def strength(line: str, handle_jokers: bool, cards: list[str]) -> int:
    hand, _ = line.split()
    return (hand_type(hand, handle_jokers), tuple(cards.index(c) for c in hand))

def strength_p1(line: str) -> int:
    return strength(line, False, CARDS_P1)

def strength_p2(line: str) -> int:
    return strength(line, True, CARDS_P2)

with open("days/day07/input", "r") as f:
    data = f.read()

result = 0
lines = sorted(data.splitlines(), key=strength_p1, reverse=True)
for i, line in enumerate(lines, start=1):
    _, bid = line.split()
    result += i * int(bid)

print(f"part1: {result}")

result = 0
lines = sorted(data.splitlines(), key=strength_p2, reverse=True)
for i, line in enumerate(lines, start=1):
    _, bid = line.split()
    result += i * int(bid)

print(f"part2: {result}")
