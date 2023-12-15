from collections import defaultdict

with open("days/day14/input", "r") as f:
    data = f.read()

total = 0
lines = data.splitlines()
height = len(lines)
width = len(lines[0])
distances = [len(lines) for _ in range(width)]
for y, line in enumerate(lines):
    for x, c in enumerate(line):
        if c == "#":
            distances[x] = height - y - 1
        elif c == "O":
            total += distances[x]
            distances[x] = distances[x] - 1

print(f"part1: {total}")

def printboard(rocks):
    xs = set([x for x, y in rocks])
    ys = set([y for x, y in rocks])
    for y in range(max(ys) + 1):
        for x in range(max(xs) + 1):
            c = rocks[(x, y)]
            if c is None:
                print(".", end="")
            else:
                print(c, end="")
        print()
    print()

def load(rocks, height):
    total = 0
    for (x, y), c in rocks.items():
        if c == "O":
            total += height - y
    return total


rocks = defaultdict(lambda: None)
for y, line in enumerate(lines):
    for x, c in enumerate(line):
        if c != ".":
            rocks[(x, y)] = c

history = []
for cycle in range(1, 1001):
    # NORTH
    fill = [0 for _ in range(width)]
    for (x, y), c in sorted(rocks.items(), key=lambda x: x[0][1]):
        if c == "#":
            fill[x] = y + 1
        elif c == "O":
            rocks.pop((x, y))
            next_y = fill[x]
            rocks[(x, next_y)] = c
            fill[x] = next_y + 1
        
    # WEST
    fill = [0 for _ in range(height)]
    for (x, y), c in sorted(rocks.items(), key=lambda x: x[0][0]):
        if c == "#":
            fill[y] = x + 1
        elif c == "O":
            rocks.pop((x, y))
            next_x = fill[y]
            rocks[(next_x, y)] = c
            fill[y] = next_x + 1

    # SOUTH
    fill = [height - 1 for _ in range(width)]
    for (x, y), c in sorted(rocks.items(), key=lambda x: -x[0][1]):
        if c == "#":
            fill[x] = y - 1
        elif c == "O":
            rocks.pop((x, y))
            next_y = fill[x]
            rocks[(x, next_y)] = c
            fill[x] = next_y - 1

    # EAST
    fill = [width - 1 for _ in range(height)]
    for (x, y), c in sorted(rocks.items(), key=lambda x: -x[0][0]):
        if c == "#":
            fill[y] = x - 1
        elif c == "O":
            rocks.pop((x, y))
            next_x = fill[y]
            rocks[(next_x, y)] = c
            fill[y] = next_x - 1

    history.append(load(rocks, height))

    # Find repeating pattern
    half_history = len(history) // 2
    for offset in range(min(100, half_history)):
        half_remaining = (len(history) - offset) // 2
        left, right = history[offset:offset + half_remaining], history[offset + half_remaining:]
        if tuple(left) == tuple(right):
            print(f"part2: {left[(1000000000 - offset - 1) % len(left)]}")
            exit(1)
