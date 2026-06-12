import re, matplotlib.pyplot as plt
from collections import defaultdict

history = defaultdict(list)
steps = []

with open("wandb/run-20260612_004941-twr49rbo/files/output.log") as f:
    step = None
    for line in f:
        m = re.search(r"Iter:\s+(\d+)", line)
        if m:
            step = int(m.group(1))
            steps.append(step)
        m = re.search(r"(\w+_(?:loss|error))\s+([\d.e+\-]+)", line)
        if m and step is not None:
            history[m.group(1)].append(float(m.group(2)))

plt.figure()
for key in list(filter(lambda kk: kk.count('_loss') * kk.count('r_'),history.keys())):
    if key in history:
        plt.semilogy(steps[:len(history[key])], history[key], label=key)
plt.legend(); 
plt.xlabel("step"); 
plt.savefig("interior_loss_history.png")
plt.show()

plt.figure()
for key in list(filter(lambda kk: kk.count('_loss') * (not kk.count('r_')),history.keys())):
    if key in history:
        plt.semilogy(steps[:len(history[key])], history[key], label=key)
plt.legend(); 
plt.xlabel("step"); 
plt.savefig("bc_loss_history.png")
plt.show()

plt.figure()
for key in list(filter(lambda kk: kk.count('_error'),history.keys())):
    if key in history:
        plt.semilogy(steps[:len(history[key])], history[key], label=key)
plt.legend(); plt.xlabel("step"); plt.savefig("error_history.png")
plt.show()

# for key in ["rc_loss", "ru_loss", "rv_loss", "u_error", "v_error"]:
#     if key in history:
#         plt.semilogy(steps[:len(history[key])], history[key], label=key)
# plt.legend(); plt.xlabel("step"); plt.savefig("loss_history.png")
