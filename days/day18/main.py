from collections import defaultdict
from typing import Callable, NamedTuple

DIRECTIONS_LIST = [(1, 0), (0, 1), (-1, 0), (0, -1)]
DIRECTIONS_MAP = {"R": (1, 0), "L": (-1, 0), "D": (0, 1), "U": (0,-1)}

class Point(NamedTuple):
    x: int
    y: int

class Line(NamedTuple):
    start: Point
    end: Point

def ray_intersects(ray: Line, line: Line) -> bool:
    if line.start.y == line.end.y:
        return False  # line is horizontal (so cant intersect the horizontal ray)
    if line.start.y > ray.start.y and line.end.y > ray.start.y:
        return False  # line is below the ray
    if line.start.y < ray.start.y and line.end.y < ray.start.y:
        return False # line is above the ray
    if line.end.x < ray.start.x:
        return False
    return True

def run(decode: Callable[[str], tuple[tuple[int, int], int]]) -> int:
    polygon_len = 0
    polygon = []
    last_x, last_y = (0, 0)
    for line in lines:
        (dx, dy), length = decode(line)
        polygon.append((last_x, last_y))
        dxl = dx * length
        dyl = dy * length
        polygon_len += abs(dxl) + abs(dyl)
        last_x, last_y = (last_x + dxl, last_y + dyl)
    polygon.append((0, 0))

    xs = sorted(set([x for x, _ in polygon]))
    ys = sorted(set([y for _, y in polygon]))
    max_x = xs[-1] + 1

    coords = defaultdict(lambda: None)
    for y1, y2 in zip(ys[::1], ys[1::1]):
        for x1, x2 in zip(xs[::1], xs[1::1]):
            width = x2 - x1
            height = y2 - y1
            coords[(x1 + (width / 2), y1 + (height / 2))] = height * width


    result = 0
    for (x, y), count in coords.items():
        n = 0
        ray = Line(Point(x, y), Point(max_x, y))
        for p1, p2 in zip(polygon[::1], polygon[1::1]):
            if ray_intersects(ray, Line(Point(*p1), Point(*p2))):
                n += 1
        if n % 2 != 0:
            result += count
    
    return result + (polygon_len //2) + 1

def part2_decode(line):
    _, __, colour = line.split(" ")
    hex_str = colour.replace("(#", "").strip(")")
    length = int("0x" + hex_str[:5], 16)
    dx, dy = DIRECTIONS_LIST[int("0x" + hex_str[5], 16)]
    return (dx, dy), length

def part1_decode(line):
    direction, length, _ = line.split(" ")
    dx, dy = DIRECTIONS_MAP[direction]
    return (dx, dy), int(length)

with open("days/day18/input", "r") as f:
    lines = f.read().splitlines()

print(f"part1: {run(part1_decode)}")
print(f"part2: {run(part2_decode)}")
