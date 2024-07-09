import json

import matplotlib.pyplot as plt
import numpy as np

with open("timings.json", "r") as fh:
    timings = json.load(fh)

fig, ax = plt.subplots(figsize=(15, 10))
x, y = zip(*sorted(timings.items(), key=lambda timing: int(timing[0])))
print(x)
cpp = np.array([item["cpp"] for item in y])
cpu = np.array([item["cpu"] for item in y])
gpu = np.array([item["gpu"] for item in y])

bw = 0.25

br1 = np.arange(len(cpu))
br2 = [x + bw for x in br1]
br3 = [x + bw for x in br2]

ax.bar(br1, cpp.mean(axis=1), label="CORESI C++", width=bw)
ax.bar(br2, cpu.mean(axis=1), label="CORESI Python CPU", width=bw)
ax.bar(br3, gpu.mean(axis=1), label="CORESI Python GPU", width=bw)
ax.set_xticks([bw, 1 + bw, 2 + bw])
ax.set_yscale("log")
ax.set_ylabel("time (seconds)")
ax.set_xlabel("Number of events")
ax.set_xticklabels(x)
ax.set_title(
    "Comparison of CORESI C++, Coresi Python CPU and GPU for 4 iterations and 500x500x1 volume"
)
ax.grid(axis="y", linestyle="dashed", linewidth=0.5)

plt.legend()
plt.tight_layout()
plt.savefig("timings_comparison_events.png", dpi=300)
plt.show()
