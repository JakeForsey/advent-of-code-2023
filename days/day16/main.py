from collections import defaultdict

NORTH = (0, -1)
SOUTH = (0, 1)
EAST = (1, 0)
WEST = (-1, 0)
DIRECTION_CHANGES = {
    ".": {NORTH: [NORTH], SOUTH: [SOUTH], EAST: [EAST], WEST: [WEST]},
    "|": {NORTH: [NORTH], SOUTH: [SOUTH], EAST: [NORTH, SOUTH], WEST: [NORTH, SOUTH]},
    "-": {NORTH: [EAST, WEST], SOUTH: [EAST, WEST], EAST: [EAST], WEST: [WEST]},
    "/": {NORTH: [EAST], SOUTH: [WEST], EAST: [NORTH], WEST: [SOUTH]},
    "\\": {NORTH: [WEST], SOUTH: [EAST], EAST: [SOUTH], WEST: [NORTH]},
}

with open("days/day16/input", "r") as f:
    data = f.read()

lines = data.splitlines()
height = len(lines)
width = len(lines[0])

grid = defaultdict(lambda: None)
for y, line in enumerate(lines):
    for x, c in enumerate(line):
        grid[(x, y)] = c

def count_energized(start_pos, start_direction):
    done = set()
    todo = [(start_pos, start_direction)]
    while todo:
        pos, d = todo.pop()
        c = grid[pos]
        if c is None:
            continue
        
        if (pos, d) in done:
            continue

        done.add((pos, d))

        for next_d in DIRECTION_CHANGES[c][d]:
            next_pos = pos[0] + next_d[0], pos[1] + next_d[1]
            todo.append((next_pos, next_d))

    energized = set([pos for pos, _ in done])
    return len(energized)

print(f"part1: {count_energized((0, 0), EAST)}")

result = -1
for x in range(width):
    result = max(count_energized((x, 0), SOUTH), result)
    result = max(count_energized((x, height), NORTH), result)

for y in range(height):
    result = max(count_energized((0, y), EAST), result)
    result = max(count_energized((width, y), WEST), result)

print(f"part2: {result}")
