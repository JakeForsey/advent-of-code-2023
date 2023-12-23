from collections import defaultdict
from heapq import heappop, heappush
from typing import NamedTuple

DIRECTIONS = {(-1, 0), (1, 0), (0, 1), (0, -1)}
CHAR_TO_DIRECTIONS = {
    ".": DIRECTIONS,
    ">": {(1, 0)},
    "<": {(-1, 0)},
    "^": {(0, -1)},
    "v": {(0, 1)},
}

class Segment(NamedTuple):
    positions: tuple[tuple[int, int]]

    @property
    def start(self):
        return self.positions[0]
    
    @property
    def end(self):
        return self.positions[-1]

with open("days/day23/input", "r") as f:
    lines = f.read().splitlines()

coords = defaultdict(lambda: None)
for y, line in enumerate(lines):
    for x, c in enumerate(line):
        if c == "#":
            continue
        coords[(x, y)] = c

start = min(coords, key=lambda coord: coord[1])
end = max(coords, key=lambda coord: coord[1])

def part1():
    done = set()
    todo = {(start, )}
    while todo:
        path = todo.pop()
        x, y = path[-1]
        if (x, y) == end:
            done.add(path)
            continue
        for dx, dy in DIRECTIONS:
            next_pos = x + dx, y + dy
            if next_pos in path:
                continue
            if next_pos not in coords:
                continue
            if (dx, dy) not in CHAR_TO_DIRECTIONS[coords[next_pos]]:
                continue
            todo.add(tuple(list(path) + [next_pos]))

    return max(len(path) for path in done) - 1

def part2():

    # Summarise nodes that don't branch
    done = set()
    todo = [Segment((start, ))]
    segments = set()
    while todo:
        segment = todo.pop(0)
        if segment in segments:
            continue
        neighbors = []
        for dx, dy in DIRECTIONS:
            next_pos = segment.end[0] + dx, segment.end[1] + dy
            if next_pos in segment.positions:
                continue
            if next_pos in coords:
                done.add(next_pos)
                neighbors.append(next_pos)

        if len(neighbors) == 1:
            todo.append(Segment(segment.positions + (neighbors[0], )))
        else:
            if not any(set(segment.positions) == set(other.positions) for other in segments):
                segments.add(segment)
            for neighbor in neighbors:
                todo.append(Segment((segment.end, neighbor, )))

    edges = defaultdict(set)
    for segment in segments:
        edges[segment.start].add((segment.end, len(segment.positions)))
        edges[segment.end].add((segment.start, len(segment.positions)))

    max_weight = -1
    todo = []
    heappush(todo, (0, (start, )))
    while todo:
        weight, path = heappop(todo)
        if path[-1] == end:
            if weight > max_weight:
                max_weight = weight
            continue
        for next_node, next_weight in edges[path[-1]]:
            if next_node in path:
                continue
            heappush(todo, (weight + next_weight - 1, path + (next_node, )))

    return max_weight

print(f"part1: {part1()}")
print(f"part2: {part2()}")
