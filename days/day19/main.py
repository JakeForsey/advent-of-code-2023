from collections import defaultdict
import math

with open("days/day19/input", "r") as f:
    data = f.read()

rules_str, parts_str = data.split("\n\n")

rules = {}
for rule_str in rules_str.splitlines():
    rule_id, matchers_str = rule_str.split("{")
    matchers = matchers_str.split(",")[:-1]
    default = matchers_str.split(",")[-1].strip("}")
    rules[rule_id] = (matchers, default)

parts = []
for part_str in parts_str.splitlines():
    part_str = part_str.strip("}").strip("{")
    parts.append({item_str.split("=")[0]: int(item_str.split("=")[1]) for item_str in part_str.split(",")})

def apply(matcher, part):
    test, next_rule_id = matcher.split(":")
    key = test[0]
    operator = test[1]
    target = int(test[2:])
    if operator == ">":
        return next_rule_id if part.get(key, float("-inf")) > target else None
    elif operator == "<":
        return next_rule_id if part.get(key, float("inf")) < target else None
    else:
        raise AssertionError("Unreachable")

result = 0
for part in parts:
    rule_id = "in"
    while rule_id not in "RA":
        matchers, default = rules[rule_id]
        matches = [apply(matcher, part) for matcher in matchers if apply(matcher, part)]
        if matches:
            rule_id = matches[0]
        else:
            rule_id = default

        if rule_id == "A":
            result += sum(part.values())

print(f"part1: {result}")

edges = defaultdict(list)
for rule_id, (matchers, default) in rules.items():
    for i in range(len(matchers)):
        next_rule_id = matchers[i].split(":")[1]
        edges[next_rule_id].append((rule_id, matchers[:i], [matchers[i]]))
    edges[default].append((rule_id, matchers, []))

initial_bound = (0, 4000)
todo = [(["A"], {"x": initial_bound, "m": initial_bound, "a": initial_bound, "s": initial_bound})]
done = []
while todo:
    rule_ids, bounds = todo.pop()

    if rule_ids[-1] == "in":
        done.append((rule_ids, bounds))
        continue

    for next_rule_id, negative_matchers, positive_matchers in edges[rule_ids[-1]]:
        tmp = bounds.copy()

        for matcher in negative_matchers:
            test, _ = matcher.split(":")
            key = test[0]
            operator = test[1]
            target = int(test[2:])
            bound = tmp[key]
            if operator == "<":
                tmp[key] = (max(bound[0], target - 1), bound[1])
            elif operator == ">":
                tmp[key] = (bound[0], min(bound[1], target))
        
        for matcher in positive_matchers:
            test, _ = matcher.split(":")
            key = test[0]
            operator = test[1]
            target = int(test[2:])
            bound = tmp[key]
            if operator == ">":
                tmp[key] = (max(bound[0], target), bound[1])
            elif operator == "<":
                tmp[key] = (bound[0], min(bound[1], target - 1))

        if all(bound[0] < bound[1] for bound in bounds.values()):
            todo.append((rule_ids.copy() + [next_rule_id], tmp))

result = sum([math.prod([max_ - min_ for min_, max_ in bounds.values()]) for _, bounds in done])
print(f"part2: {result}")
