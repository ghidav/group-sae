import pprint

# --- JSON from disk, represented here as a Python dict ---
data = {
    "1": {
        "labels": ["0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0"],
        "amd": 0.45899999141693115,
    },
    "2": {
        "labels": ["0", "0", "0", "0", "0", "0", "0", "0", "0", "1", "1", "1", "1", "1", "1"],
        "amd": 0.39399999380111694,
    },
    "3": {
        "labels": ["1", "1", "1", "1", "1", "2", "2", "2", "2", "0", "0", "0", "0", "0", "0"],
        "amd": 0.35600000619888306,
    },
    "4": {
        "labels": ["0", "0", "0", "0", "0", "2", "2", "2", "2", "1", "1", "1", "1", "3", "3"],
        "amd": 0.35600000619888306,
    },
    "5": {
        "labels": ["4", "4", "1", "1", "1", "2", "2", "2", "2", "0", "0", "0", "0", "3", "3"],
        "amd": 0.26899999380111694,
    },
    "6": {
        "labels": ["4", "4", "1", "1", "1", "0", "0", "0", "0", "2", "2", "5", "5", "3", "3"],
        "amd": 0.26899999380111694,
    },
    "7": {
        "labels": ["1", "1", "0", "0", "0", "4", "4", "2", "2", "6", "6", "5", "5", "3", "3"],
        "amd": 0.26899999380111694,
    },
    "8": {
        "labels": ["0", "0", "7", "3", "3", "4", "4", "2", "2", "6", "6", "5", "5", "1", "1"],
        "amd": 0.20999999344348907,
    },
    "9": {
        "labels": ["8", "4", "7", "3", "3", "1", "1", "2", "2", "6", "6", "5", "5", "0", "0"],
        "amd": 0.16200000047683716,
    },
    "10": {
        "labels": ["8", "9", "7", "1", "1", "0", "0", "2", "2", "6", "6", "5", "5", "4", "3"],
        "amd": 0.16200000047683716,
    },
    "11": {
        "labels": ["8", "9", "7", "0", "0", "10", "3", "2", "2", "6", "6", "5", "5", "4", "1"],
        "amd": 0.1589999943971634,
    },
    "12": {
        "labels": ["8", "9", "7", "11", "6", "10", "3", "0", "0", "2", "2", "5", "5", "4", "1"],
        "amd": 0.14800000190734863,
    },
    "13": {
        "labels": ["8", "9", "7", "11", "6", "10", "3", "12", "5", "0", "0", "2", "2", "4", "1"],
        "amd": 0.14300000667572021,
    },
    "14": {
        "labels": ["8", "9", "7", "11", "13", "10", "3", "12", "5", "6", "2", "0", "0", "4", "1"],
        "amd": 0.13699999451637268,
    },
    "training_clusters": {
        "k1-c0": ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14"],
        "k2-c0": ["0", "1", "2", "3", "4", "5", "6", "7", "8"],
        "k2-c1": ["9", "10", "11", "12", "13", "14"],
        "k3-c0": ["0", "1", "2", "3", "4"],
        "k3-c1": ["5", "6", "7", "8"],
        "k4-c0": ["9", "10", "11", "12"],
        "k4-c1": ["13", "14"],
        "k5-c0": ["2", "3", "4"],
        "k5-c1": ["0", "1"],
        "k6-c0": ["9", "10"],
        "k6-c1": ["11", "12"],
        "k7-c0": ["7", "8"],
        "k7-c1": ["5", "6"],
        "k8-c0": ["3", "4"],
    },
}

# --- Our computed rule ---
# For each k-group (1 to 8) and for each layer 0..14:
# * Among all training clusters with k value <= current k-group
#   that include the layer (as a string),
#   choose the one with the highest k value.
# * If none are found, fall back to "layers.<i>"


def get_cluster_for_layer(k_group, layer, training_clusters):
    layer_str = str(layer)
    candidates = []
    for cluster_key, layers in training_clusters.items():
        # Extract the k value from a key like "k2-c1"
        k_val = int(cluster_key.split("-")[0][1:])
        if k_val <= k_group and layer_str in layers:
            candidates.append((k_val, cluster_key))
    if candidates:
        # Choose the candidate with the maximum k value.
        best = max(candidates, key=lambda x: x[0])
        return best[1]
    else:
        return f"layers.{layer}"


# Build the 8 k–groups (each a list of 15 entries for layers 0 to 14)
training_clusters = data["training_clusters"]
pythia_1b_groups = {}
for k in range(1, 9):
    group = []
    for layer in range(15):
        group.append(get_cluster_for_layer(k, layer, training_clusters))
    pythia_1b_groups[str(k)] = group

# Print out the computed groups.
print("Computed pythia_1b k–groups (each list has 15 entries):")
pprint.pprint(pythia_1b_groups)

# --- Expected output (as described in the problem statement) ---
expected = {
    "1": ["k1-c0"] * 15,
    "2": (["k2-c0"] * 9) + (["k2-c1"] * 6),
    "3": (["k3-c0"] * 5) + (["k3-c1"] * 4) + (["k2-c1"] * 6),
    "4": (["k3-c0"] * 5) + (["k3-c1"] * 4) + (["k4-c0"] * 4) + (["k4-c1"] * 2),
    "5": (["k5-c1"] * 2) + (["k5-c0"] * 3) + (["k3-c1"] * 4) + (["k4-c0"] * 4) + (["k4-c1"] * 2),
    "6": (["k5-c1"] * 2)
    + (["k5-c0"] * 3)
    + (["k3-c1"] * 4)
    + (["k6-c0"] * 2)
    + (["k6-c1"] * 2)
    + (["k4-c1"] * 2),
    "7": (["k5-c1"] * 2)
    + (["k5-c0"] * 3)
    + (["k7-c1"] * 2)
    + (["k7-c0"] * 2)
    + (["k6-c0"] * 2)
    + (["k6-c1"] * 2)
    + (["k4-c1"] * 2),
    "8": (["k5-c1"] * 2)
    + (["k5-c0"] * 1)
    + (["k8-c0"] * 2)
    + (["k7-c1"] * 2)
    + (["k7-c0"] * 2)
    + (["k6-c0"] * 2)
    + (["k6-c1"] * 2)
    + (["k4-c1"] * 2),
}

print("\nExpected pythia_1b k–groups:")
pprint.pprint(expected)

# --- Verify that the computed groups match the expected ones ---
for k in expected:
    comp = pythia_1b_groups[k]
    exp = expected[k]
    if comp == exp:
        print(f"k–group {k}: PASS")
    else:
        print(f"k–group {k}: FAIL")
        print("Computed:", comp)
        print("Expected:", exp)
