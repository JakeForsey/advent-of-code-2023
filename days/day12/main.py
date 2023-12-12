from functools import lru_cache
from itertools import combinations, takewhile
from collections import defaultdict

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

# result = 0
# for line in enumerate(data.splitlines()):

#     pattern, counts = line.split(" ")
#     pattern = list(pattern)
#     counts = list(map(int, counts.split(",")))
    
#     total = sum(counts)
#     hashes = sum(1 for c in pattern if c == "#")
#     for choice in set(combinations([i for i, c in enumerate(pattern) if c == "?"], total - hashes)):
#         tmp = []
#         for i, c in enumerate(pattern):
#             if c == "?":
#                 tmp.append("." if i not in choice else "#")
#             else:
#                 tmp.append(c)

#         v = valid(tmp, counts.copy())
#         if v:
#             result += 1

# print(f"part1: {result}")

result = 0
for li, line in enumerate(data.splitlines()):
    print(line, end="")

    pattern, counts = line.split(" ")
    pattern = list("?".join(pattern for _ in range(5)))
    counts = ",".join(counts for _ in range(5))
    counts = list(map(int, counts.split(",")))

    options = defaultdict(list)
    for count in set(counts):
        for start in range(len(pattern)):
            end = start + (count - 1)
            book_start = start - 1
            book_end = end + 1
            if end >= len(pattern):
                continue
            if book_start >= 0 and pattern[book_start] not in "?.":
                continue
            if book_end < len(pattern) and pattern[book_end] not in "?.":
                continue
            if "." in pattern[start: end + 1]:
                continue
            options[count].append((start, end))

    # print("options:")
    # for count, option in options.items():
    #     print(f"{count}: {option}")

    @lru_cache(maxsize=1_000_000)
    def recursive(last, counts):
        if not counts:
            return 1

        try:
            next_hash = pattern.index("#", last + 1)
        except ValueError:
            next_hash = len(pattern)

        total = 0
        for (start, end) in options[counts[0]]:
            if start > next_hash:
                break
            if start > last:
                total += recursive(end + 1, counts[1:])
        return total

    line_result = recursive(-1, tuple(counts))
    print(f" - {line_result} arrangement")
    result += line_result

print(f"part2: {result}")

# too low: 9377
# too high: 20655489691690,
#           34191380712181
#           35024564475996