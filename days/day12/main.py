from itertools import combinations, takewhile

with open("days/day12/input", "r") as f:
    data = f.read()

def valid(pattern, counts):
    contiguous = 0
    for c in pattern:
        if c == ".":
            if contiguous > 0:
                if not counts:
                    return False
                count = counts.pop(0)
                if count != contiguous:
                    return False
            contiguous = 0
        elif c == "#":
            contiguous += 1
        elif c == "?":
            raise AssertionError("Unreachable")
    
    if contiguous > 0:
        if not counts:
            return False
        count = counts.pop()
        if count != contiguous:
            return False
    if counts:
        return False
    return True

result = 0
for li, line in enumerate(data.splitlines()):

    pattern, counts = line.split(" ")
    pattern = list(pattern)
    counts = list(map(int, counts.split(",")))
    
    total = sum(counts)
    hashes = sum(1 for c in pattern if c == "#")
    for choice in set(combinations([i for i, c in enumerate(pattern) if c == "?"], total - hashes)):
        tmp = []
        for i, c in enumerate(pattern):
            if c == "?":
                tmp.append("." if i not in choice else "#")
            else:
                tmp.append(c)

        v = valid(tmp, counts.copy())
        if v:
            result += 1

print(f"part1: {result}")
