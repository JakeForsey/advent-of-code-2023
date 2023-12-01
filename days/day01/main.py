import string

with open("days/day01/input", "r") as f:
    data = f.read()

result = 0
for line in data.splitlines():
    numbers = [int(c) for c in line if c in string.digits]
    result += numbers[0] * 10
    result += numbers[-1]

print("part1: ", result)

word_numbers = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    **{d: int(d) for d in string.digits}
}
result = 0
for line in data.splitlines():
    first, last = None, None
    for i in range(len(line)):
        for word, num in word_numbers.items():
            if line[i:].startswith(word) and not first:
                first = num
            if line[-(i + 1):].startswith(word) and not last:
                last = num

    result += first * 10
    result += last

print("part2: ", result)
