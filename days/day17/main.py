from collections import defaultdict
from typing import NamedTuple
from queue import PriorityQueue

with open("days/day17/input", "r") as f:
    data = f.read()
lines = data.splitlines()

grid = defaultdict(lambda: None)
for y, line in enumerate(lines):
    for x, c in enumerate(line):
        grid[(x, y)] = int(c)

NORTH = (0, -1)
SOUTH = (0, 1)
EAST = (1, 0)
WEST = (-1, 0)

class Node(NamedTuple):
    position: tuple[int, int]
    direction: tuple[int, int]
    in_a_row: int

def priority(position, end_position, score):
    remaining_distance = abs(position[0] - end_position[0]) + abs(position[1] - end_position[1])
    distance = position[0] + position[1]
    avg_score = next_score / (distance + 0.00001)
    return score + (remaining_distance * avg_score)

best_score = float("inf")
end_position = (len(lines[0]) - 1, len(lines)-1)
start = Node((0,0), None, 0)
tree: dict[Node, None] = {}
scores: dict[Node, int] = defaultdict(lambda: float("inf"))
scores[start] = 0
todo = PriorityQueue()
todo.put((end_position[0] + end_position[1], start))
while not todo.empty():
    _, node = todo.get()
    if node.position == end_position:
        best_score = min(best_score, scores[node])
        continue
    for next_direction in [NORTH, SOUTH, EAST, WEST]:
        next_position = node.position[0] + next_direction[0], node.position[1] + next_direction[1]
        next_in_a_row = node.in_a_row + 1 if next_direction == node.direction else 1

        previous_position = tree.get(node, None)
        if previous_position is not None:
            previous_position = previous_position.position

        tile = grid[next_position]
        if tile is None:
            continue
        next_score = scores[node] + tile
        if next_position == previous_position:
            continue
        if next_in_a_row > 3:
            continue
        if next_score > best_score:
            continue
        
        next_node = Node(next_position, next_direction, next_in_a_row)
        if next_score < scores[next_node]:
            scores[next_node] = next_score
            tree[next_node] = node
            todo.put((
                priority(next_position, end_position, next_score),
                next_node
            ))

print(f"part1: {best_score}")
    

best_score = float("inf")
end_position = (len(lines[0]) - 1, len(lines)-1)
start = Node((0,0), None, 0)
tree: dict[Node, None] = {}
scores: dict[Node, int] = defaultdict(lambda: float("inf"))
scores[start] = 0
todo = PriorityQueue()
todo.put((end_position[0] + end_position[1], start))
while not todo.empty():
    _, node = todo.get()

    if node.position == end_position:
        if node.in_a_row >= 4:
            best_score = min(best_score, scores[node])
        continue

    for next_direction in [NORTH, SOUTH, EAST, WEST]:
        if node.direction is not None and next_direction != node.direction and node.in_a_row < 4:
            continue

        next_position = node.position[0] + next_direction[0], node.position[1] + next_direction[1]
        next_in_a_row = node.in_a_row + 1 if next_direction == node.direction else 1

        previous_position = tree.get(node, None)
        if previous_position is not None:
            previous_position = previous_position.position

        tile = grid[next_position]
        if tile is None:
            continue
        next_score = scores[node] + tile
        if next_position == previous_position:
            continue
        if next_in_a_row > 10:
            continue
        if next_score > best_score:
            continue
        
        next_node = Node(next_position, next_direction, next_in_a_row)
        if next_score < scores[next_node]:
            scores[next_node] = next_score
            tree[next_node] = node
            todo.put((
                priority(next_position, end_position, next_score),
                next_node
            ))

print(f"part2: {best_score}")
