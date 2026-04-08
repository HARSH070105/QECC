import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from itertools import combinations

from gb_254_28 import HX as HX_A1, HZ as HZ_A1
from ghgp_882_24 import HX as HX_B1, HZ as HZ_B1
# from hgp_7938_578 import HX as HX_C1, HZ as HZ_C1


# config
p_list = np.linspace(0.01, 0.12, 8)
num_trials = 100
max_iter = 15
osd_order = 0   # -1 = BP only

# error model - depol
def generate_pauli_error(n, p):
    r = np.random.rand(n)

    X = np.zeros(n, dtype=np.uint8)
    Z = np.zeros(n, dtype=np.uint8)

    for i in range(n):
        if r[i] < p:
            t = np.random.randint(3)
            if t == 0:   # X
                X[i] = 1
            elif t == 1: # Z
                Z[i] = 1
            else:        # Y
                X[i] = 1
                Z[i] = 1

    return X, Z

# BP
class BPDecoder:
    def __init__(self, H, p, max_iter=20):
        self.H = H
        self.m, self.n = H.shape
        self.max_iter = max_iter
        self.p = p

        self.check_to_var = [np.where(H[i])[0] for i in range(self.m)]
        self.var_to_check = [np.where(H[:, j])[0] for j in range(self.n)]

    def decode(self, syndrome):
        p_eff = 2 * self.p / 3
        llr = np.full(self.n, np.log((1 - p_eff) / p_eff))
        
        msg_vc = np.zeros((self.m, self.n))
        msg_cv = np.zeros((self.m, self.n))

        # init
        for i in range(self.m):
            for j in self.check_to_var[i]:
                msg_vc[i, j] = llr[j]

        for _ in range(self.max_iter):

            # check → var
            for i in range(self.m):
                idxs = self.check_to_var[i]

                for j in idxs:
                    sign = -1 if int(syndrome[i]) else 1
                    min_val = np.inf

                    for k in idxs:
                        if k != j:
                            val = msg_vc[i, k]
                            sign *= -1 if val < 0 else 1
                            min_val = min(min_val, abs(val))

                    msg_cv[i, j] = sign * min_val

            # var → check
            posterior = llr.copy()

            for j in range(self.n):
                checks = self.var_to_check[j]
                posterior[j] += np.sum(msg_cv[checks, j])

                for i in checks:
                    msg_vc[i, j] = posterior[j] - msg_cv[i, j]

            guess = (posterior < 0).astype(np.uint8)

            if np.all((self.H @ guess) % 2 == syndrome):
                return guess, posterior

        return guess, posterior

# OSD
def gf2_solve(H, s):
    H = H.copy()
    s = s.copy()
    m, n = H.shape

    pivots = []
    row = 0

    for col in range(n):
        if row >= m:
            break

        pivot = None
        for r in range(row, m):
            if H[r, col]:
                pivot = r
                break

        if pivot is None:
            continue

        H[[row, pivot]] = H[[pivot, row]]
        s[[row, pivot]] = s[[pivot, row]]

        for r in range(m):
            if r != row and H[r, col]:
                H[r] ^= H[row]
                s[r] ^= s[row]

        pivots.append(col)
        row += 1

    return pivots, s[:row]

def osd_decode(H, syndrome, llr, order):
    n = H.shape[1]
    perm = np.argsort(np.abs(llr))

    Hs = H[:, perm]
    pivots, s_red = gf2_solve(Hs, syndrome)

    best = None
    min_w = np.inf

    for w in range(order + 1):
        for pattern in combinations(range(len(pivots)), w):
            e = np.zeros(n, dtype=np.uint8)
            s_tmp = s_red.copy()

            for idx in pattern:
                if idx < len(s_tmp):
                    s_tmp[idx] ^= 1

            for i, col in enumerate(pivots):
                if i < len(s_tmp):
                    e[col] = s_tmp[i]

            weight = np.sum(e)
            if weight < min_w:
                min_w = weight
                best = e.copy()

    out = np.zeros(n, dtype=np.uint8)
    out[perm] = best
    return out

# decoding
def decode_css(HX, HZ, X_err, Z_err, p, osd_order):

    # Z errors
    sX = (HX @ Z_err) % 2
    bpZ = BPDecoder(HX, p, max_iter)
    z_hat, llr_z = bpZ.decode(sX)

    if osd_order != -1:
        z_hat = osd_decode(HX, sX, llr_z, osd_order)

    # X errors
    sZ = (HZ @ X_err) % 2
    bpX = BPDecoder(HZ, p, max_iter)
    x_hat, llr_x = bpX.decode(sZ)

    if osd_order != -1:
        x_hat = osd_decode(HZ, sZ, llr_x, osd_order)

    # success?
    resZ = Z_err ^ z_hat
    resX = X_err ^ x_hat

    okZ = np.all((HX @ resZ) % 2 == 0)
    okX = np.all((HZ @ resX) % 2 == 0)

    return okZ and okX


# MC sims
def simulate(HX, HZ, osd_order):
    n = HX.shape[1]
    wer = []

    for p in p_list:
        fails = 0

        for _ in tqdm(range(num_trials), desc=f"p={p:.3f}, OSD-{osd_order}"):
            X, Z = generate_pauli_error(n, p)

            if not decode_css(HX, HZ, X, Z, p, osd_order):
                fails += 1

        wer.append(fails / num_trials)

    return np.array(wer)


codes = {
    "A1 (GB)": (np.array(HX_A1, dtype=np.uint8), np.array(HZ_A1, dtype=np.uint8)),
    "B1 (GHP)": (np.array(HX_B1, dtype=np.uint8), np.array(HZ_B1, dtype=np.uint8)),
    # "C1 (HGP)": (np.array(HX_C1, dtype=np.uint8), np.array(HZ_C1, dtype=np.uint8)),
}

results_bp = {}
results_osd = {}

for name, (HX, HZ) in codes.items():
    print(f"\n=== {name} : BP ===")
    results_bp[name] = simulate(HX, HZ, osd_order=-1)

    print(f"\n=== {name} : BP+OSD-{osd_order} ===")
    results_osd[name] = simulate(HX, HZ, osd_order=osd_order)


plt.figure(figsize=(7,6))

for name in results_bp:
    plt.semilogy(p_list, results_bp[name], 'o-', label=f"{name}, BP")

for name in results_osd:
    plt.semilogy(p_list, results_osd[name], 'o--', label=f"{name}, BP+OSD")

plt.xlabel("Physical error rate (p)")
plt.ylabel("WER")
plt.title("QLDPC BP vs BP+OSD")
plt.grid(True, which="both", linestyle="--", alpha=0.6)
plt.legend()

plt.savefig("final_results.png", dpi=300)
plt.show()