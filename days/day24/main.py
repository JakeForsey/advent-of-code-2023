with open("days/day24/input", "r") as f:
    lines = f.read().splitlines()

formulas = []
for line in lines:
    position_str, velocity_str = line.split(" @ ")
    px, py, pz = map(int, position_str.split(", "))
    vx, vy, vz = map(int, velocity_str.split(", "))
    x1, x2 = px, px + vx
    y1, y2 = py, py + vy
    m = (y2 - y1) / (x2 - x1)
    c = -m * x1 + y1
    formulas.append((px, py, vx, vy, m, c))

min_x = min_y = 200000000000000
max_x = max_y = 400000000000000

result = 0
for i, formula_1 in enumerate(formulas):
    for formula_2 in formulas[i:]:
        px1, py1, vx1, vy1, m1, c1 = formula_1
        px2, py2, vx2, vy2, m2, c2 = formula_2
        if m1 == m2:
            # Parallel
            continue
        x = (c2 - c1) / (m1 - m2)
        y = (m1 * c2 - c1 * m2) / (m1 - m2)

        if vx1 > 0 and x < px1:
            continue
        if vx1 < 0 and x > px1:
            continue

        if vx2 > 0 and x < px2:
            continue
        if vx2 < 0 and x > px2:
            continue

        if (min_x < x < max_x and min_y < y < max_y):
            result += 1

print(f"part1: {result}")
