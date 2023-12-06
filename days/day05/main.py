with open("days/day05/input", "r") as f:
    data = f.read()

seeds = list(map(int, data.splitlines()[0].split(": ")[1].split()))
convertions = data.split("\n\n")[1:]

min_location = float("inf")
for seed in seeds:
    for convertion in convertions:
        for row in convertion.splitlines()[1:]:
            dest, convertion_start, size  = map(int, row.split())
            convertion_start, convertion_end = convertion_start, convertion_start + size
            if (convertion_start <= seed and seed < convertion_end):
                seed = dest + (seed - convertion_start)
                break
    if (seed < min_location):
        min_location = seed

print(f"part1: {min_location}")

todo = []
for seed_start, size in zip(seeds[::2], seeds[1::2]):
    todo.append((seed_start, seed_start + size, convertions.copy()))

locations = []
while todo:
    seed_start, seed_end, convertions = todo.pop()
    if not convertions:
        locations.append(seed_start)
        continue

    convertion = convertions.pop(0)
    match = False
    for row in convertion.splitlines()[1:]:
        dest, convertion_start, size  = map(int, row.split())
        convertion_end = convertion_start + size

        if (convertion_start <= seed_start and seed_start < convertion_end) and (convertion_start <= seed_end and seed_end < convertion_end):
            seed_start = dest + (seed_start - convertion_start)
            seed_end = dest + (seed_end - convertion_start)
            todo.append((seed_start, seed_end, convertions))
            match = True
            break

        if (convertion_start <= seed_start and seed_start < convertion_end) or (convertion_start <= seed_end and seed_end < convertion_end):
            if (seed_start < convertion_start):
                todo.append((seed_start, convertion_start - 1, [convertion] + convertions))
                todo.append((convertion_start, seed_end, [convertion] + convertions))
            else:
                todo.append((seed_start, convertion_end - 1 , [convertion] + convertions))
                todo.append((convertion_end, seed_end, [convertion] + convertions))
            match = True
            break

    if not match:
        todo.append((seed_start, seed_end, convertions))

print(f"part1: {min(locations)}")
