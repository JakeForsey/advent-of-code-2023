with open("days/day11/input", "r") as f:
    data = f.read()

expanded_rows = {i: True for i in range(len(data.splitlines()))}
expanded_cols = {i: True for i in range(len(data.splitlines()[0]))}
coords = {}
i = 1
for y, line in enumerate(data.splitlines()):
    for x, c in enumerate(line):
        if c == "#":
            expanded_cols[x] = False
            expanded_rows[y] = False
            coords[(x, y)] = i
            i += 1

result = 0
for (x1, y1), i1 in coords.items():
    for (x2, y2), i2 in coords.items():
        if i1 > i2: continue
        extra_rows = sum(
            1 for row, expanded in expanded_rows.items() 
            if expanded and (y1 < row < y2 or y2 < row < y1)
        )
        extra_cols = sum(
            1 for col, expanded in expanded_cols.items() 
            if expanded and (x1 < col < x2 or x2 < col < x1)
        )
        dx = x1 - x2
        dy = y1 - y2
        distance = abs(dx) + abs(dy)
        result += distance + extra_cols + extra_rows

print(f"part1: {result}")

result = 0
for (x1, y1), i1 in coords.items():
    for (x2, y2), i2 in coords.items():
        if i1 > i2: continue
        extra_rows = sum(
            1 for row, expanded in expanded_rows.items() 
            if expanded and (y1 < row < y2 or y2 < row < y1)
        ) 
        extra_cols = sum(
            1 for col, expanded in expanded_cols.items() 
            if expanded and (x1 < col < x2 or x2 < col < x1)
        )
        dx = x1 - x2
        dy = y1 - y2
        distance = abs(dx) + abs(dy)
        result += distance + (extra_cols + extra_rows) * 999_999

print(f"part2: {result}")
