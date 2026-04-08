import itertools

def generate_lattice(L, M):

    horizontal = []
    vertical = []

    for x in range(L):
        for y in range(M + 1):
            horizontal.append(("h", x, y))

    for x in range(L + 1):
        for y in range(M):
            vertical.append(("v", x, y))

    return horizontal, vertical


def tile_edges(D):

    edges = []

    for x in range(D):
        for y in range(D):

            if y < D - 1:
                edges.append(("h", x, y))

            if x < D - 1:
                edges.append(("v", x, y))

    return edges


def derive_z_tile(x_tile, D):

    z_tile = []

    for t, x, y in x_tile:

        if t == "h":
            z_tile.append(("v", D - 1 - x, D - 1 - y))

        else:
            z_tile.append(("h", D - 1 - x, D - 1 - y))

    return z_tile


def generate_tiles(D, w):

    edges = tile_edges(D)

    tiles = []

    for combo in itertools.combinations(edges, w):

        x_tile = list(combo)
        z_tile = derive_z_tile(x_tile, D)

        tiles.append((x_tile, z_tile))

    return tiles


def compute_n(L, M):

    h, v = generate_lattice(L, M)
    return len(h) + len(v)


def compute_k(D):

    return 2 * D * D


def compute_distance(tile, L, M, D):
    return min(L, M) #needs fixing



def efficiency(n, k, d):

    return (k * d * d) / n


def search_tiles(L, M, D, w, filename):

    tiles = generate_tiles(D, w)

    n = compute_n(L, M)
    k = compute_k(D)

    results = []

    for tile in tiles:

        d = compute_distance(tile, L, M, D)

        eta = efficiency(n, k, d)

        results.append((eta, tile, n, k, d))

    # sort by efficiency
    results.sort(reverse=True, key=lambda x: x[0])

    with open(filename, "w") as f:

        for eta, tile, n, k, d in results:

            f.write("Efficiency kd^2/n: {:.4f}\n".format(eta))
            f.write("n = {}, k = {}, d = {}\n".format(n, k, d))

            f.write("X tile:\n")
            f.write(str(tile[0]) + "\n")

            f.write("Z tile:\n")
            f.write(str(tile[1]) + "\n")

            f.write("-" * 40 + "\n")


if __name__ == "__main__":

    L = 12
    M = 12
    D = 3
    w = 6

    search_tiles(L, M, D, w, "tile_results.txt")