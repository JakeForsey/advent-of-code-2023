from collections import defaultdict
import math
from typing import NamedTuple

NodeId = str

class Beam(NamedTuple):
    source: str
    state: bool

    @property
    def high(self):
        return self.state
    
    @property
    def low(self):
        return not self.state

class FlipFlop:
    """Flip-flop modules (prefix %) are either on or off;
    they are initially off. If a flip-flop module receives a 
    high pulse, it is ignored and nothing happens. However,
    if a flip-flop module receives a low pulse, it flips
    between on and off. If it was off, it turns on and sends
    a high pulse. If it was on, it turns off and sends a low
    pulse."""
    def __init__(self, node_id):
        self.node_id = node_id
        self.on = False
    
    def step(self, beam: Beam) -> Beam:
        if beam.high:
            # If it receives a high pulse, it is ignored and nothing happens
            return None
        if not self.on:
            # If it was off, it turns on and sends a high pulse
            self.on = True
            return Beam(self.node_id, True)
        else:
            # If it was on, it turns off and sends a low pulse
            self.on = False 
            return Beam(self.node_id, False)

class Conjunction:
    """Conjunction modules (prefix &) remember the type of the 
    most recent pulse received from each of their connected
    input modules; they initially default to remembering a low
    pulse for each input. When a pulse is received, the
    conjunction module first updates its memory for that input. 
    Then, if it remembers high pulses for all inputs, it sends a 
    low pulse; otherwise, it sends a high pulse."""
    def __init__(self, node_id, downstream_ids):
        self.node_id = node_id
        self.states = {did: False for did in downstream_ids}

    def step(self, beam: Beam):
        self.states[beam.source] = beam.state
        beam_state = not all(state for state in self.states.values())
        return Beam(self.node_id, beam_state)

class Broadcast:
    def __init__(self, node_id):
        self.node_id = node_id

    def step(self, beam):
        return Beam(self.node_id, beam.state)

class NoOp:
    def step(self, beam):
        return None

with open("days/day20/input", "r") as f:
    data = f.read()

edges = defaultdict(list)
for line in data.splitlines():
    source_str, rest_str = line.split(" -> ")
    source_type, source_id = source_str[0], source_str[1:]
    for dest in rest_str.split(", "):
        edges[source_id].append(dest)

nodes = {}
for line in data.splitlines():
    source_str, rest_str = line.split(" -> ")
    source_type, source_id = source_str[0], source_str[1:]
    if source_type == "b":
        node = Broadcast(source_id)
    elif source_type == "%":
        node = FlipFlop(source_id)
    elif source_type == "&":
        node = Conjunction(source_id, [n for n, ns in edges.items() if source_id in ns])
    else:
        raise AssertionError("unreachable")
    nodes[source_id] = node

periods = {}
total_low, total_high = 0, 0
for step in range(1, 10000):
    if step < 1001:
        total_low += 1  # button pushes count as low beams
    beams = [("roadcaster", Beam("roadcaster", False))]
    while beams:
        next_beams = []
        for node_id, beam in beams:
            next_beam = nodes.get(node_id, NoOp()).step(beam)
            if next_beam is not None:
                for other_id in edges[node_id]:                    
                    if step < 1001:
                        if next_beam.high:
                            total_high += 1
                        else:
                            total_low += 1
                    next_beams.append((other_id, next_beam))
        beams = next_beams

    # Hardcoded the central nodes in each subgraph 
    for node_id in ["cs", "ck", "jh", "dx"]:
        if node_id in periods:
            continue
        states = []
        parent_ids = {n for n, ns in edges.items() if node_id in ns}
        parent_ids |= set(edges[node_id])
        
        for parent_id in parent_ids:
            parent = nodes[parent_id]
            if isinstance(parent, FlipFlop):    
                states.append(int(parent.on))
        if all([s == 0 for s in states]):
            periods[node_id] = step

print(f"part1: {total_high * total_low}")
print(f"part2: {math.lcm(*periods.values())}")
