from collections import defaultdict

with open("days/day18/input", "r") as f:
    lines = f.read().splitlines()

DIRECTIONS = {
    "R": (1, 0),
    "L": (-1, 0),
    "D": (0, 1),
    "U": (0,-1),
}
position = (0, 0)
coords = {position}
for line in lines:
    direction, length, colour = line.split(" ")
    dx, dy = DIRECTIONS[direction]
    for i in range(int(length)):
        position = position[0] + dx, position[1] + dy
        coords.add(position)

min_x = min((x for x, _ in coords))
min_y = min((y for _, y in coords))
max_x = max((x for x, _ in coords))
max_y = max((y for _, y in coords))
areas = []
for y in range(min_y, max_y):
    for x in range(min_x, max_x):
        if (x, y) in coords:
            continue

        if any((x, y) in area for area, _ in areas):
            continue
        
        enclosed = True
        todo = [(x, y)]
        area = set()
        while todo:
            pos = todo.pop()
            area.add(pos)
            for direction in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                next_pos = pos[0] + direction[0], pos[1] + direction[1]
                
                if next_pos in area:
                    continue
                
                if next_pos in coords:
                    continue
                    
                if next_pos[0] < min_x or next_pos[0] >= max_x + 1:
                    enclosed = False
                    continue

                if next_pos[1] < min_y or next_pos[1] >= max_y + 1:
                    enclosed = False
                    continue
                
                todo.append(next_pos)
        areas.append((area, enclosed))

area = [area for area, enclosed in areas if enclosed][0]
print(f"part1: {len(area) + len(coords)}")
