import math

with open("days/day04/input", "r") as f:
    data = f.read()

result = 0
for line in data.splitlines():
    _, line = line.split(":")
    win, nums = line.split(" | ")
    nums = set(map(int, nums.split()))
    win = set(map(int, win.split()))
    wins = nums & win
    score = 0
    if wins:
        score = 1
        for i in range(len(wins) - 1):
            score = score * 2
    result += score

print(f"part1: {result}")