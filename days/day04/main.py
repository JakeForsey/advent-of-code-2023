from functools import lru_cache

with open("days/day04/input", "r") as f:
    data = f.read()

@lru_cache(maxsize=100_000)
def calculate_wins(line: str) -> int:
    _, line = line.split(":")
    win, nums = line.split(" | ")
    nums = set(map(int, nums.split()))
    win = set(map(int, win.split()))
    wins = nums & win
    return len(wins)

@lru_cache(maxsize=100_000)
def calculate_score(line: str) -> int:
    wins = calculate_wins(line)
    score = 0
    if wins > 0:
        score = 1
        for _ in range(wins - 1):
            score = score * 2
    return score

result = 0
for line in data.splitlines():
    result += calculate_score(line)

print(f"part1: {result}")

result = 0
lines = data.splitlines()
todo = list(range(len(lines)))
while todo:
    li = todo.pop()
    result += 1
    for i in range(li + 1, li + 1 + calculate_wins(lines[li])):
        todo.append(i)

print(f"part 2: {result}")
