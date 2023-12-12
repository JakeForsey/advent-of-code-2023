from collections import defaultdict
from functools import lru_cache

with open("days/day12/input", "r") as f:
    data = f.read()

def process_line(line, folds=5):
    pattern, counts = line.split(" ")
    pattern = list("?".join(pattern for _ in range(folds)))
    counts = list(map(int, counts.split(","))) * folds

    options = defaultdict(list)
    for count in set(counts):
        for start in range(len(pattern)):
            end = start + count
            book_start = start - 1
            if end > len(pattern):
                continue
            if book_start >= 0 and pattern[book_start] not in "?.":
                continue
            if end < len(pattern) and pattern[end] not in "?.":
                continue
            if "." in pattern[start: end]:
                continue
            options[count].append((start, end))

    @lru_cache(maxsize=1_000_000)
    def recursive(last: int, counts: tuple[counts]) -> int:
        if not counts:
            return 0 if "#" in pattern[last:] else 1

        try:
            next_hash = pattern.index("#", last + 1)
        except ValueError:
            next_hash = len(pattern)

        total = 0
        for (start, end) in options[counts[0]]:
            if start > next_hash:
                break
            if start > last:
                total += recursive(end, counts[1:])
        return total

    return recursive(-1, tuple(counts))

result = 0
for line in data.splitlines():
    result += process_line(line, folds=1)

print(f"part1: {result}")

result = 0
for line in data.splitlines():
    result += process_line(line, folds=5)

print(f"part2: {result}")
