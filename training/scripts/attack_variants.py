"""Geometry-only hypotheses; no arena physics or controllers are changed.

An empty universal safe band breaks this particular defensive construction. It
does not prove a winning attack, balanced gameplay, or human playability.
"""
import json
from math import ceil, floor
from pathlib import Path


def reflect(y):
    z = (y - 4) % 304
    return 4 + (z if z <= 152 else 304 - z)


def analyze(cap, deflection):
    shots = {}
    for y in range(4, 157):
        shots[y] = []
        for offset in range(-15, 16):
            center = y - offset
            if not 15 <= center <= 145:
                continue
            for motion in ([-1, 0, 1] if deflection else [0]):
                # Actual movement on the contact tick. A clamped, stationary
                # paddle contributes zero; impossible pre-contact centers do not count.
                if not 15 <= center - 2 * motion <= 145:
                    continue
                vx = 5 + floor((cap - 5) * abs(offset) / 15 + .5)
                vy = floor(offset / 3 + .5) + deflection * motion
                ticks = ceil(330 / vx)
                shots[y].append((center, ticks, reflect(y + ticks * vy)))
    bands = {y: (max([15] + [landing - 15 - 2 * ticks for _, ticks, landing in row]),
                 min([145] + [landing + 15 + 2 * ticks for _, ticks, landing in row]))
             for y, row in shots.items()}
    empty = [y for y, (low, high) in bands.items() if low > high]
    failures = sum(max(center - 2 * ticks, bands[landing][0], 15)
                   > min(center + 2 * ticks, bands[landing][1], 145)
                   for row in shots.values() for center, ticks, landing in row)
    return dict(horizontal_cap=cap, contact_motion_deflection=deflection,
                vertical_cap=5 + deflection, shortest_flight_ticks=ceil(330 / cap),
                contact_heights=153, empty_safe_band_heights=empty,
                minimum_band_width=min(high - low for low, high in bands.values()),
                failed_interval_recovery_cases=failures,
                legal_shots=sum(map(len, shots.values())))


if __name__ == '__main__':
    rows = [analyze(cap, amount) for cap, amount in
            [(13, 0), (14, 0), (15, 0), (16, 0), (13, 1), (13, 2), (13, 3)]]
    current = rows[0]
    assert current['legal_shots'] == 4041
    assert current['empty_safe_band_heights'] == []
    assert current['failed_interval_recovery_cases'] == 0
    assert len(rows[5]['empty_safe_band_heights']) == 27
    output = Path(__file__).resolve().parents[1] / 'results/attack-audit/variants.json'
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(dict(
        scope='Geometry-only candidate hypotheses; not simulated matches or changed rules.',
        deflection='outgoing vy = round(offset/3) + k * actual paddle movement direction on contact; maximum absolute vy is 5+k; no persistent spin state.',
        limitation='Breaking the universal-band construction does not establish forced wins, interesting play or human accessibility. Requires adaptive opponents, retraining and human testing.',
        rows=rows), indent=2) + '\n')
    print(output)
