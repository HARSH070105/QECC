import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from itertools import combinations
from concurrent.futures import ProcessPoolExecutor
from scipy import sparse

# Import your matrices (assumed to be sparse NPZ loads as per previous turns)
from gb_254_28 import HX as HX_A1, HZ as HZ_A1
from ghgp_882_24 import HX as HX_B1, HZ as HZ_B1
from hgp_7938_578 import HX as HX_C1, HZ as HZ_C1

# Configuration matching paper [cite: 357, 358]
p_list = np.linspace(0.01, 0.12, 8)
num_trials = 100
max_iter = 32
osd_order = 0 
nms_factor = 0.625  # Normalized Min-Sum factor 

def generate_pauli_error(n, p):
    r = np.random.rand(n)
    X, Z = np.zeros(n, dtype=np.uint8), np.zeros(n, dtype=np.uint8)
    for i in range(n):
        if r[i] < p:
            t = np.random.randint(3)
            if t == 0: X[i] = 1        # X error
            elif t == 1: Z[i] = 1      # Z error
            else: X[i], Z[i] = 1, 1    # Y error
    return X, Z

class BPDecoder:
    def __init__(self, H, p, max_iter=32, nms_factor=0.625):
        # Ensure H is CSR for efficient row slicing
        self.H = H.tocsr() if sparse.issparse(H) else sparse.csr_matrix(H)
        self.m, self.n = self.H.shape
        self.max_iter = max_iter
        self.p = p
        self.nms_factor = nms_factor
        
        # Pre-calculate neighbors for speed [cite: 141, 142]
        self.check_to_var = [self.H.getrow(i).indices for i in range(self.m)]
        self.var_to_check = [self.H.getcol(j).indices for j in range(self.n)]

    def decode(self, syndrome):
        p_eff = 2 * self.p / 3 # [cite: 251, 307]
        llr = np.full(self.n, np.log((1 - p_eff) / p_eff))
        msg_vc = np.zeros((self.m, self.n))
        msg_cv = np.zeros((self.m, self.n))

        # Initialization
        for i in range(self.m):
            for j in self.check_to_var[i]:
                msg_vc[i, j] = llr[j]

        for _ in range(self.max_iter):
            # Check-to-Variable Update with NMS 
            for i in range(self.m):
                idxs = self.check_to_var[i]
                for j in idxs:
                    sign = -1 if int(syndrome[i]) else 1
                    min_val = np.inf
                    for k in idxs:
                        if k != j:
                            val = msg_vc[i, k]
                            sign *= np.sign(val) if val != 0 else 1
                            min_val = min(min_val, abs(val))
                    # Apply Normalized Min-Sum scaling 
                    msg_cv[i, j] = sign * min_val * self.nms_factor

            # Variable-to-Check Update
            posterior = llr.copy()
            for j in range(self.n):
                checks = self.var_to_check[j]
                posterior[j] += np.sum(msg_cv[checks, j])
                for i in checks:
                    msg_vc[i, j] = posterior[j] - msg_cv[i, j]

            guess = (posterior < 0).astype(np.uint8)
            if np.all((self.H @ guess) % 2 == syndrome):
                return guess, posterior

        return (posterior < 0).astype(np.uint8), posterior

def gf2_solve(H, s):
    # Gaussian elimination over GF(2) [cite: 215]
    H, s = H.copy(), s.copy()
    m, n = H.shape
    pivots, row = [], 0
    for col in range(n):
        if row >= m: break
        pivot = np.where(H[row:, col])[0]
        if len(pivot) == 0: continue
        pivot = pivot[0] + row
        H[[row, pivot]], s[[row, pivot]] = H[[pivot, row]], s[[pivot, row]]
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
    # Standard OSD uses a dense solver on the reliable basis [cite: 212, 214]
    Hs = H.toarray()[:, perm] if sparse.issparse(H) else H[:, perm]
    pivots, s_red = gf2_solve(Hs, syndrome)
    
    best, min_w = None, np.inf
    for w in range(order + 1):
        for pattern in combinations(range(min(len(pivots), 20)), w): # Limit search space
            e = np.zeros(n, dtype=np.uint8)
            s_tmp = s_red.copy()
            for idx in pattern:
                if idx < len(s_tmp): s_tmp[idx] ^= 1
            for i, col in enumerate(pivots):
                if i < len(s_tmp): e[col] = s_tmp[i]
            
            # The paper selects recovery operator with minimum weight [cite: 43]
            weight = np.sum(e)
            if weight < min_w:
                min_w, best = weight, e.copy()

    out = np.zeros(n, dtype=np.uint8)
    if best is not None: out[perm] = best
    return out

def run_single_trial(args):
    HX, HZ, p, osd_order = args
    X_err, Z_err = generate_pauli_error(HX.shape[1], p)
    
    # Decode Z-type errors using HX [cite: 310, 311]
    sX = (HX @ Z_err) % 2
    bpZ = BPDecoder(HX, p, max_iter, nms_factor)
    z_hat, llr_z = bpZ.decode(sX)
    
    # OSD post-processing triggers only if BP fails 
    if osd_order != -1 and not np.all((HX @ z_hat) % 2 == sX):
        z_hat = osd_decode(HX, sX, llr_z, osd_order)

    # Decode X-type errors using HZ
    sZ = (HZ @ X_err) % 2
    bpX = BPDecoder(HZ, p, max_iter, nms_factor)
    x_hat, llr_x = bpX.decode(sZ)
    
    if osd_order != -1 and not np.all((HZ @ x_hat) % 2 == sZ):
        x_hat = osd_decode(HZ, sZ, llr_x, osd_order)

    resZ, resX = Z_err ^ z_hat, X_err ^ x_hat
    return np.all((HX @ resZ) % 2 == 0) and np.all((HZ @ resX) % 2 == 0)

def simulate_parallel(HX, HZ, osd_order):
    results = []
    for p in p_list:
        with ProcessPoolExecutor() as executor:
            args = [(HX, HZ, p, osd_order) for _ in range(num_trials)]
            trial_results = list(tqdm(executor.map(run_single_trial, args), total=num_trials, desc=f"p={p:.3f}"))
            results.append(1.0 - np.mean(trial_results))
    return np.array(results)

# Main Execution
if __name__ == "__main__":
    codes = {"A1 (GB)": (HX_A1, HZ_A1), "B1 (GHP)": (HX_B1, HZ_B1)}
    
    plt.figure(figsize=(7,6))
    for name, (HX, HZ) in codes.items():
        print(f"\nSimulating {name}...")
        res_bp = simulate_parallel(HX, HZ, osd_order=-1)
        res_osd = simulate_parallel(HX, HZ, osd_order=0)
        
        plt.semilogy(p_list, res_bp, 'o-', label=f"{name}, BP")
        plt.semilogy(p_list, res_osd, 'o--', label=f"{name}, BP+OSD")

    plt.xlabel("Physical error rate (p)")
    plt.ylabel("WER")
    plt.legend()
    plt.grid(True, which="both", linestyle="--")
    plt.show()