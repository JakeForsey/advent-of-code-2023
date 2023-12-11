from collections import defaultdict

def adjacent(x, y):
    yield x, y - 1
    yield x, y + 1
    yield x - 1, y
    yield x + 1, y

VALID_DIRECTIONS = {
    #     (x, y)  (x, y)
    "|": {(0, 1), (0, -1)},
    "-": {(1, 0), (-1, 0)},

    "L": {(0, -1), (1, 0)},
    "J": {(0, -1), (-1, 0)},

    "7": {(0, 1), (-1, 0)},
    "F": {(0, 1), (1, 0)},
    ".": {},
    None: {},
    "S": set(adjacent(0, 0))
}

with open("input", "r") as f:
    data = f.read()

grid = {}
for y, line in enumerate(data.splitlines()):
    for x, c in enumerate(line):
        if c == "S": start = (x, y)
        grid[(x, y)] = c

loops = []
todo = [(start, [start])]
while todo:
    (x, y), path = todo.pop()
    pipe = grid.get((x, y), None)
    for dx, dy in VALID_DIRECTIONS[pipe]:
        cx, cy = x + dx, y + dy
        if (cx, cy) == start: loops.append(path)
        if (cx, cy) in path: continue
        todo.append(((cx, cy), path + [(cx,  cy)]))

longest_loop = max(loops, key=len)
print(f"part1: {len(longest_loop)//2}")

EXPANDED_TILES = {
    "S": [["S", "S", "S"],
          ["S", "S", "S"],
          ["S", "S", "S"]],
    ".": [[".", ".", "."],
          [".", ".", "."],
          [".", ".", "."]],
    "|": [["x", "|", "x"],
          ["x", "|", "x"], 
          ["x", "|", "x"]],
    "-": [["x", "x", "x"],
          ["-", "-", "-"],
          ["x", "x", "x"]],
    "7": [["x", "x", "x"],
          ["-", "7", "x"],
          ["x", "|", "x"]],
    "F": [["x", "x", "x"],
          ["x", "F", "-"],
          ["x", "|", "x"]],
    "L": [["x", "|", "x"],
          ["x", "L", "-"],
          ["x", "x", "x"]],
    "J": [["x", "|", "x"],
          ["-", "J", "x"], 
          ["x", "x", "x"]],
}

# Expand the grid (and mask out un-used pipe)
expanded_grid = {}
for (x, y), c in grid.items():
    c = grid[(x, y)]
    if (x, y) not in longest_loop:
        c = "."
    for dx in range(3):
        for dy in range(3):
            expanded_grid[((x * 3) + dx, (y * 3) + dy)] = EXPANDED_TILES[c][dy][dx]

class Zone:
    def __init__(self, pos: tuple[int, int], grid) -> None:
        self.grid = grid
        self.todo = [pos]
        self.filled = set()
        self.enclosed = True
        while True:
            if not self.todo:
                break

            pos = self.todo.pop()
            self.filled.add(pos)

            for next_pos in adjacent(*pos):
                if next_pos not in grid:
                    self.enclosed = False
                    continue
                if next_pos in self.filled:
                    continue
                if self.grid[next_pos] in ".x":
                    self.todo.append(next_pos)

    def count_dots(self):
        return len([pos for pos in self.filled if self.grid[pos] == "."])

zones = []
for pos, c in expanded_grid.items():
    if c != ".": continue
    if any(pos in zone.filled for zone in zones): continue
    zones.append(Zone(pos, expanded_grid))

print(f"part2: {sum(zone.count_dots() // 9 for zone in zones if zone.enclosed)}")
