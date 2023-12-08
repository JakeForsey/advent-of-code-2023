from math import lcm
from itertools import cycle

def to_node(line: str) -> str:
    return line.split(" = ")[0]

def to_children(line: str) -> tuple[str, ...]:
    children = line.split(" = ")[1]
    children = children.replace(")", "").replace("(", "")
    return tuple(children.split(", "))

with open("days/day08/input", "r") as f:
    data = f.read()

lines = data.splitlines()
directions = [0 if c == "L" else 1 for c in lines[0]]
graph = {to_node(line): to_children(line) for line in data.splitlines()[2:]}

pos = "AAA"
for distance, direction in enumerate(cycle(directions), start=1):
    pos = graph[pos][direction]
    if pos == "ZZZ":
        break

print(f"part1: {distance}")

distances = set()
for pos in [pos for pos in graph if pos.endswith("A")]:
    for distance, direction in enumerate(cycle(directions), start=1):
        pos = graph[pos][direction]
        if pos.endswith("Z"):
            distances.add(distance)
            break

print(f"part2: {lcm(*distances)}")
