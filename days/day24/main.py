from z3 import Solver, Ints, Int, sat

with open("input", "r") as f:
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
    formulas.append((px, py, pz, vx, vy, vz, m, c))

min_x = min_y = 200000000000000
max_x = max_y = 400000000000000

result = 0
for i, formula_1 in enumerate(formulas):
    for formula_2 in formulas[i:]:
        px1, py1, _, vx1, vy1, _, m1, c1 = formula_1
        px2, py2, _, vx2, vy2, _, m2, c2 = formula_2
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

solver = Solver()

px0, py0, pz0 = Ints("px0 py0 pz0")
vx0, vy0, vz0 = Ints("vx0 vy0 vz0")

for i, (px1, py1, pz1, vx1, vy1, vz1, _, __) in enumerate(formulas[:3]):
    t = Int(f"t{i}")
    solver.add(t > 0)
    solver.add(px1 + t * vx1 == px0 + t * vx0)
    solver.add(py1 + t * vy1 == py0 + t * vy0)
    solver.add(pz1 + t * vz1 == pz0 + t * vz0)

assert solver.check() == sat
model = solver.model()
print(f"part1: {model.eval(px0).as_long() + model.eval(py0).as_long() + model.eval(pz0).as_long()}")
