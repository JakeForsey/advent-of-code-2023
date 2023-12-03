from collections import defaultdict
import math

with open("days/day03/input", "r") as f:
    data = f.read()

def adjacent(x, y):
    yield x, y - 1
    yield x, y + 1
    yield x - 1, y
    yield x + 1, y
    yield x + 1, y + 1
    yield x - 1, y - 1
    yield x + 1, y - 1
    yield x - 1, y + 1

# Parse input
digits = defaultdict(lambda: None)
symbols = {}
for y, line in enumerate(data.splitlines()):
    for x, c in enumerate(line):
        if c == ".":
            continue
        elif c.isdigit():
            digits[(x, y)] = c
        else:
            symbols[(x, y)] = c

def build_number(visited, pos) -> int:
    """Accumulate digits to the left and right of this position."""
    adj_digits = digits[pos]
    lx, y = pos
    while True:
        lx = lx - 1
        visited.add((lx, y))
        ldigit = digits[(lx, y)]
        if ldigit is None:
            break
        adj_digits = ldigit + adj_digits
    rx, y = pos
    while True:
        rx = rx + 1
        visited.add((rx, y))
        rdigit = digits[(rx, y)]
        if rdigit is None:
            break
        adj_digits = adj_digits + rdigit
    return int(adj_digits)

def part1():
    result = 0
    visited = set()
    for pos in symbols:
        for adj_pos in adjacent(*pos):
            if adj_pos in visited:
                continue
            visited.add(adj_pos)
            digit = digits[adj_pos]
            if digit is None:
                continue
            result += build_number(visited, adj_pos)
            visited.add(adj_pos)

    print(f"part1: {result}")

def part2():
    result = 0
    visited = set()
    for pos, symbol in symbols.items():
        if symbol != "*":
            continue
        adj_numbers = []
        for adj_pos in adjacent(*pos):
            if adj_pos in visited:
                continue
            visited.add(adj_pos)
            digit = digits[adj_pos]
            if digit is None:
                continue
            adj_numbers.append(build_number(visited, adj_pos))
        if len(adj_numbers) == 2:
            result += math.prod(adj_numbers)

    print(f"part2: {result}")

part1()
part2()
