from collections import defaultdict
from typing import NamedTuple
from queue import PriorityQueue

class Node(NamedTuple):
    position: tuple[int, int]
    direction: tuple[int, int]
    in_a_row: int

with open("days/day17/input", "r") as f:
    data = f.read()
lines = data.splitlines()

grid = defaultdict(lambda: None)
for y, line in enumerate(lines):
    for x, c in enumerate(line):
        grid[(x, y)] = int(c)

start_position = (0, 0)
end_position = (len(lines[0]) - 1, len(lines) - 1)

def run(min_in_a_row, max_in_a_row):
    best_score = float("inf")
    start = Node(start_position, (0, 0), 0)
    scores = defaultdict(lambda: float("inf"))
    scores[start] = 0
    todo = PriorityQueue()
    todo.put((0, start))
    while not todo.empty():
        _, node = todo.get()
        
        if node.position == end_position:
            if node.in_a_row >= min_in_a_row:
                best_score = min(best_score, scores[node])
            continue

        for next_direction in [(0, -1), (0, 1), (1, 0), (-1, 0)]:
            if next_direction == (node.direction[0] * -1, node.direction[1] * -1):
                continue  # Don't reverse

            next_position = node.position[0] + next_direction[0], node.position[1] + next_direction[1]
            cost = grid[next_position]
            if cost is None:
                continue  # Don't go off the grid

            next_in_a_row = node.in_a_row + 1 if next_direction == node.direction else 1
            if next_in_a_row > max_in_a_row:
                continue  # Don't go in the same direction for too long
            if node.in_a_row < min_in_a_row and node.direction != next_direction and node.direction != (0, 0):
                continue  # Go in the same direction for long enough

            next_score = scores[node] + cost
            if next_score > best_score:
                continue  # Prune paths early
            
            next_node = Node(next_position, next_direction, next_in_a_row)
            if next_score < scores[next_node]:
                scores[next_node] = next_score
                todo.put((next_position[0] + next_position[1], next_node))
    
    return best_score

print(f"part1: {run(0, 3)}")
print(f"part2: {run(4, 10)}")
