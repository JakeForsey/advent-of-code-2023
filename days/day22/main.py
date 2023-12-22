from collections import defaultdict
from typing import NamedTuple

with open("days/day22/input", "r") as f:
    lines = f.read().splitlines()

class Position(NamedTuple):
    x: int
    y: int
    z: int
    def __repr__(self):
        return f"(x={self.x}, y={self.y}, z={self.z})"

class Cube(NamedTuple):
    start: Position
    end: Position

def touches(a: Cube, b: Cube) -> bool:
    # https://stackoverflow.com/a/53488289
    return (a.end.x > b.start.x
        and a.start.x < b.end.x
        
        and a.end.y > b.start.y
        and a.start.y < b.end.y

        and a.end.z >= b.start.z
        and a.start.z <= b.end.z
    )

def print_x(cubes: list[Cube], labels: list[str]) -> None:
    min_x = min(cube.start.x for cube in cubes)
    max_x = max(cube.end.x for cube in cubes)
    min_z = min(cube.start.z for cube in cubes)
    max_z = max(cube.end.z for cube in cubes)

    print(" x")
    print("".join(map(str, range(min_x, max_x))))
    for z in range(max_z, min_z - 2, -1):
        for x in range(min_x, max_x):
            matches = [label for cube, label in zip(cubes, labels) if cube.start.x <= x < cube.end.x and cube.start.z <= z < cube.end.z]
            if len(matches) == 1:
                print(matches[0], end="")
            elif len(matches) > 1:
                print("?", end="")
            elif z == 0:
                print("-", end="")
            else:
                print(".", end="")
        print(f" {z}")
    print()

def drop(a: Cube) -> Cube:
    return Cube(Position(a.start.x, a.start.y, a.start.z -1), Position(a.end.x, a.end.y, a.end.z - 1))

cubes = []
for line in lines:
    start, end = line.split("~")
    sx, sy, sz = map(int, start.split(","))
    ex, ey, ez = map(int, end.split(","))
    cube = Cube(Position(sx, sy, sz), Position(ex + 1, ey + 1, ez + 1))
    cubes.append(cube)

labels = ["A", "B", "C", "D", "E", "F", "G"]
if len(cubes) == len(labels):
    print_x(cubes, labels)

supported_by = defaultdict(set)
supporting = defaultdict(set)
cubes = sorted(cubes, key=lambda cube: cube.start.z)
for i, cube in enumerate(cubes):
    cubes_below = [other for other in cubes if other != cube and other.end.z <= cube.start.z]    
    while True:
        supports = [other for other in cubes_below if touches(cube, other)]
        if len(supports) > 0:
            for support in supports:
                supported_by[cube].add(support)
                supporting[support].add(cube)
            break
        if cube.start.z <= 1:
            break
        cube = drop(cube)
    cubes[i] = cube

labels = ["A", "B", "C", "D", "E", "F", "G"]
if len(cubes) == len(labels):
    print_x(cubes, labels)

result = 0
for cube in cubes:
    cant_remove = False
    for above in supporting[cube]:
        if len(supported_by[above]) == 1:
            cant_remove = True
    if not cant_remove:
        result += 1

print(f"part1: {result}")
