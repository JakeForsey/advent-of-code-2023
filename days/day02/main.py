from collections import defaultdict
import math

with open("days/day02/input", "r") as f:
    data = f.read()

result = 0
target = {
    "red": 12, 
    "green": 13,
    "blue": 14
}

for line in data.splitlines():
    game, handfuls = line.split(":")
    game_number = int(game.split(" ")[1])

    def viable() -> bool:
        for handful in handfuls.split(";"):
            for count_colour in handful.strip().split(","):
                count, colour = count_colour.strip().split(" ")
                if int(count) > target[colour]:
                    return False
        return True

    if viable():
        result += game_number

print(f"part1: {result}")

result = 0
for line in data.splitlines():
    game, handfuls = line.split(":")
    game_number = int(game.split(" ")[1])
    maxs = defaultdict(int)
    for handful in handfuls.split(";"):
        for count_colour in handful.strip().split(","):
            count, colour = count_colour.strip().split(" ")
            maxs[colour] = max(maxs[colour], int(count))
    result += math.prod(maxs.values())

print(f"part2: {result}")
