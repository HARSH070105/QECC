import numpy as np
from itertools import combinations
import BP_OSD_TILE.tile_code as config

class TileCodeMatrixBuilder:
    def __init__(self, L, B, x_tile):
        # Loading from config file
        self.L = L
        self.B = B
        self.x_tile = x_tile
        self.z_tile = self._derive_z_tile()
        self.num_edges = 2 * L * L
        
    def _derive_z_tile(self):
        # Make the X tile from the equations of paper
        z_tile = []
        for e_type, x, y in self.x_tile:
            z_tile.append((1 - e_type, self.B - 1 - x, self.B - 1 - y))
        return z_tile

    # indexing the edges (h -> 0-100, v -> 100-200)
    def _edge_to_idx(self, e_type, x, y):
        return e_type * (self.L * self.L) + y * self.L + x


    ############################################################
    ########          FIXXX THISSSSSSSSSSSS        #############
    ############################################################
    def build_matrices(self):
        # Tiling and makingh the parity chk matrices
        num_stabs = (self.L - self.B + 1)**2
        hx = np.zeros((num_stabs, self.num_edges), dtype=int)
        hz = np.zeros((num_stabs, self.num_edges), dtype=int)
        
        row = 0
        for dy in range(self.L - self.B + 1):
            for dx in range(self.L - self.B + 1):
                for e_type, x, y in self.x_tile:
                    hx[row, self._edge_to_idx(e_type, x + dx, y + dy)] = 1
                for e_type, x, y in self.z_tile:
                    hz[row, self._edge_to_idx(e_type, x + dx, y + dy)] = 1
                row += 1
        return hx, hz

class BPOSD:
    def __init__(self, H, max_iter=20, channel_prob=0.05, osd_order=1):
        self.H = H
        self.m, self.n = H.shape
        self.max_iter = max_iter
        self.p = channel_prob
        self.osd_order = osd_order

    def gf2_elimination(self, H, syndrome):
        M = H.copy()
        s = syndrome.copy()
        rows, cols = M.shape
        pivot_row = 0
        basis_cols = []

        print("Starting row reduction...")
        for c in range(cols):
            if pivot_row >= rows:
                break
            
            # Find pivot
            pivot = -1
            for r in range(pivot_row, rows):
                if M[r, c] == 1:
                    pivot = r
                    break
            
            if pivot != -1:
                basis_cols.append(c)
                # Swap
                M[[pivot_row, pivot]] = M[[pivot, pivot_row]]
                s[[pivot_row, pivot]] = s[[pivot, pivot_row]]
                # Eliminate
                for r in range(rows):
                    if r != pivot_row and M[r, c] == 1:
                        M[r] = (M[r] + M[pivot_row]) % 2
                        s[r] = (s[r] + s[pivot_row]) % 2
                pivot_row += 1

        print(f"Found basis of size {len(basis_cols)}.")
        return basis_cols, s[:pivot_row]

    def decode(self, syndrome):
        print("\n=== STARTING BELIEF PROPAGATION ===")
        llrs = np.full(self.n, np.log((1 - self.p) / self.p))
        msg_cv = np.zeros((self.m, self.n))
        msg_vc = np.zeros((self.m, self.n))
        
        # Init step
        for i in range(self.m): 
            for j in range(self.n):
                if self.H[i, j]:
                    msg_vc[i, j] = llrs[j]

        posteriori_llrs = llrs.copy()

        for iteration in range(self.max_iter):
            print(f"\n  [BP Iteration {iteration + 1}]")
            
            # Check-to-Variable
            for i in range(self.m):
                connected_vars = np.where(self.H[i, :] == 1)[0]
                for j in connected_vars:
                    sign = (-1) ** syndrome[i]
                    min_mag = np.inf
                    for k in connected_vars:
                        if k != j:
                            sign *= np.sign(msg_vc[i, k]) if msg_vc[i, k] != 0 else 1
                            min_mag = min(min_mag, abs(msg_vc[i, k]))
                    msg_cv[i, j] = sign * min_mag

            # Variable-to-Check
            posteriori_llrs = llrs.copy()
            for j in range(self.n):
                connected_checks = np.where(self.H[:, j] == 1)[0]
                posteriori_llrs[j] += np.sum(msg_cv[connected_checks, j])
                
                for i in connected_checks:
                    msg_vc[i, j] = llrs[j] + np.sum(msg_cv[connected_checks, j]) - msg_cv[i, j]

            # decision
            guess = (posteriori_llrs < 0).astype(int)
            current_syndrome = (self.H @ guess) % 2
            mismatches = np.sum(current_syndrome != syndrome)
            
            print(f"Syndrome mismatches remaining: {mismatches}")
            if mismatches == 0:
                print(f"BP Converged successfully in {iteration + 1} iterations.")
                return guess

        print(f"\nBP FAILED TO CONVERGE. TRIGGERING OSD-{self.osd_order} ===")
        
        # OSD Step 1: Sort by Reliability
        print("  [OSD Step 1] Sorting columns by posteriori LLR reliabilities...")
        reliabilities = np.abs(posteriori_llrs)
        sorted_indices = np.argsort(reliabilities) 
        H_sorted = self.H[:, sorted_indices]
        
        # OSD Step 2: GF(2) Elimination
        print("  [OSD Step 2] Performing GF(2) Gaussian Elimination...")
        basis_cols, reduced_syndrome = self.gf2_elimination(H_sorted, syndrome)
        
        # OSD Step 3: Bit-flip search
        print(f"  [OSD Step 3] Testing bit-flip combinations up to weight {self.osd_order}...")
        best_error_guess = None
        min_weight = np.inf
        
        flip_patterns = [()] 
        for flip_weight in range(1, self.osd_order + 1):
            flip_patterns.extend(combinations(range(len(basis_cols)), flip_weight))
            
        for pattern in flip_patterns:
            e_test = np.zeros(self.n, dtype=int)
            s_test = reduced_syndrome.copy()
            
            for bit_idx in pattern:
                s_test[bit_idx] = (s_test[bit_idx] + 1) % 2
                
            for idx, col_idx in enumerate(basis_cols):
                if idx < len(s_test):
                    e_test[col_idx] = s_test[idx]
                    
            current_weight = np.sum(e_test)
            if current_weight < min_weight:
                min_weight = current_weight
                best_error_guess = e_test.copy()

        print(f"Best recovery weight found by OSD: {min_weight}")

        # Un-sort the error vector
        final_guess = np.zeros(self.n, dtype=int)
        final_guess[sorted_indices] = best_error_guess
        
        print("=== DECODING COMPLETE ===\n")
        return final_guess


if __name__ == "__main__":
    print("Building Tile Code Matrices...")
    builder = TileCodeMatrixBuilder(config.L, config.B, config.X_TILE)
    hx, hz = builder.build_matrices()
    
    # Verify commutativity (H_X * H_Z^T = 0 mod 2)
    commute_check = np.max((hx @ hz.T) % 2)
    if commute_check != 0:
        print("WARNING: H_X and H_Z do not commute! Check edge mappings.")
    
    print(f"Matrix H_X shape: {hx.shape}")
    
    error_z = np.zeros(builder.num_edges, dtype=int)
    for e_type, x, y in config.ERRORED_EDGES_Z:
        idx = builder._edge_to_idx(e_type, x, y)
        if idx < builder.num_edges:
            error_z[idx] = 1

    syndrome_x = (hx @ error_z) % 2
    print(f"Injected Z-Error Weight: {np.sum(error_z)}")
    print(f"Generated Syndrome Weight: {np.sum(syndrome_x)}")

    decoder = BPOSD(hx, max_iter=10, channel_prob=config.CHANNEL_PROB, osd_order=1)
    recovery_z = decoder.decode(syndrome_x)
    
    final_syndrome = (hx @ recovery_z) % 2
    success = np.array_equal(final_syndrome, syndrome_x)
    # print(f"Final Verification - Did recovery clear the syndrome?: {success}")
    print(f"Final Recovery Weight: {np.sum(recovery_z)}")