L = 10 # Big lattice
B = 3  # The Tile

# 0 = horizontal, 1 = vertical
# tile starts at 0,0

X_TILE = [
    (1, 0, 2), (1, 1, 2), (1, 2, 0),
    (0, 0, 0), (0, 2, 1), (0, 2, 2)
]

ERRORED_EDGES_Z = [
    (0, 2, 2), (1, 4, 5), (0, 6, 7), 
    (1, 1, 1), (0, 3, 4), (1, 5, 5), 
    (0, 7, 7)
]

ERRORED_EDGES_X = []

CHANNEL_PROB = 0.05