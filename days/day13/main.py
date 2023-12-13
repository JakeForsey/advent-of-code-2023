with open("days/day13/input", "r") as f:
    data = f.read()

result = 0
for pattern in data.split("\n\n"):
    rows = pattern.splitlines()
    cols = [[row[i] for row in rows] for i in range(len(rows[0]))]

    n_rows = len(rows)
    for y in range(1, n_rows):
        size = min(y, n_rows - y)
        left = rows[y - size:y]
        right = rows[y:y + size][::-1]
        if all(l == r for l, r in zip(left, right)):
            result += (100 * y)
    
    n_cols = len(cols)
    for x in range(1, n_cols):
        size = min(x, n_cols - x)
        top = cols[x - size: x]
        bottom = cols[x:x + size][::-1]
        if all(t == b for t, b in zip(top, bottom)):
            result += x

print(f"part1: {result}")

def flatten(data: list[str]) -> list[str]:
    ret = []
    for x in data:
        ret.extend([c for c in x])
    return ret

result = 0
for pattern in data.split("\n\n"):
    rows = pattern.splitlines()
    cols = [[row[i] for row in rows] for i in range(len(rows[0]))]

    n_rows = len(rows)
    for y in range(1, n_rows):
        size = min(y, n_rows - y)
        left = rows[y - size:y]
        right = rows[y:y + size][::-1]
        mismatches = sum(l != r for l, r in zip(flatten(left), flatten(right)))
        if mismatches == 1:
            result += (100 * y)

    n_cols = len(cols)
    for x in range(1, n_cols):
        size = min(x, n_cols - x)
        top = cols[x - size: x]
        bottom = cols[x:x + size][::-1]
        mismatches = sum(t != b for t, b in zip(flatten(top), flatten(bottom)))
        if mismatches == 1:
            result += x

print(f"part2: {result}")
