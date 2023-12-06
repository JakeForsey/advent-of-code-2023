with open("days/day06/input", "r") as f:
    data = f.read()

times, distances = data.splitlines()
times = list(map(int, times.split()[1:]))
distances = list(map(int, distances.split()[1:]))

result = 1
for time, record_distance in zip(times, distances):
    n = 0
    for i in range(time):
        remaining_time = time - i
        distance = i * remaining_time
        if distance > record_distance:
            n += 1
    result *= n

print(f"part1: {result}")

times, distances = data.splitlines()
time = int("".join(times.split()[1:]))
record_distance = int("".join(distances.split()[1:]))

result = 0
for i in range(time):
    remaining_time = time - i
    distance = i * remaining_time
    if distance > record_distance:
        result += 1

print(f"part2: {result}")