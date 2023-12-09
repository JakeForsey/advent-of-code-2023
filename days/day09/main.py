with open("days/day09/input", "r") as f:
    data = f.read()

def get_diffs(line: str) -> list[list[int]]:
    diffs = [list(map(int, line.split()))]
    while not all(x == 0 for x in diffs[-1]):
        series = diffs[-1]
        diffs.append([b - a for a, b in zip(series[::1], series[1::1])])
    diffs.reverse()
    return diffs

result = 0
for line in data.splitlines():
    diffs = get_diffs(line)
    for i in range(len(diffs)):
        if i == 0:
            continue
        diffs[i].append(diffs[i][-1] + diffs[i - 1][-1])
    
    result += diffs[-1][-1]

print(f"part1: {result}")

result = 0
for line in data.splitlines():
    diffs = get_diffs(line)
    for i in range(len(diffs)):
        if i == 0:
            continue
        diffs[i].insert(0, diffs[i][0] - diffs[i - 1][0])
    
    result += diffs[-1][0]

print(f"part2: {result}")
