from collections import defaultdict

with open("days/day15/input", "r") as f:
    data = f.read()

def custom_hash(string: str) -> int:
    value = 0
    for c in string:
        value += ord(c)
        value *= 17
        value = value % 256
    return value

result = 0
for string in data.split(","):
    result += custom_hash(string)

print(f"part1: {result}")

boxes = defaultdict(list)
for string in data.split(","):

    if "=" in string:
        label, focal_length = string.split("=")
        h = custom_hash(label)
        inserted = False
        for i, (other, _) in enumerate(boxes[h]):
            if label == other:
                boxes[h][i] = (label, focal_length)
                inserted = True
                break
        
        if not inserted:
            boxes[h].append((label, focal_length))

    elif "-" in string:
        label = string.strip("-")
        h = custom_hash(label)
        for i, (other, focal_length) in enumerate(boxes[h]):
            if label == other:
                boxes[h].remove((other, focal_length))

result = 0
for h, box in boxes.items():
    for slot, (label, focal_length) in enumerate(box, start=1):
        result += (h + 1) * slot * int(focal_length)

print(f"part2: {result}")
