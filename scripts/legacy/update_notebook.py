import json
from pathlib import Path

nb_path = Path("notebooks/colab_training.ipynb")

with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

# --- Update Cell 7 (Config) ---
# Index 7 corresponds to the code cell after "2️⃣ Configuration" markdown?
# Let's find it by content to be safe.
config_cell = None
graph_cell = None
train_cell = None

for cell in nb["cells"]:
    src = "".join(cell["source"])
    if "MAX_EPOCHS =" in src and "SAMPLE_FRAC =" in src:
        config_cell = cell
    if "def build_edges_in_chunks" in src:
        graph_cell = cell
    if "criterion = FocalLoss" in src:
        train_cell = cell

# 1. Update Config
if config_cell:
    new_source = []
    for line in config_cell["source"]:
        if "MAX_EPOCHS =" in line:
            new_source.append("MAX_EPOCHS = 30\n")
        else:
            new_source.append(line)
    config_cell["source"] = new_source
    print("Updated Config Cell (MAX_EPOCHS=30)")

# 2. Update Graph Building (Use GraphBuilder)
if graph_cell:
    # Replace manual implementation with GraphBuilder class usage
    new_source = [
        "import torch\n",
        "import gc\n",
        "from pathlib import Path\n",
        "from src.data.graph_builder import GraphBuilder\n",
        "from src.utils.config import load_model_config\n",
        "\n",
        'GRAPH_CACHE = f"{MODELS_DIR}/edges_full.pt"\n',
        "\n",
        "if os.path.exists(GRAPH_CACHE):\n",
        '    print(f"Loading cached graph from {GRAPH_CACHE}...")\n',
        "    edge_index = torch.load(GRAPH_CACHE)\n",
        '    print(f"Loaded {edge_index.shape[1]:,} edges")\n',
        "    edge_index = edge_index.to(device)\n",
        "    X_full = X_full.to(device)\n",
        "else:\n",
        '    print("🚀 Starting Memory-Optimized Graph Build (Directed)...")\n',
        "    \n",
        "    # Configure GraphBuilder\n",
        "    model_cfg = load_model_config()\n",
        "    model_cfg.graph.similarity_threshold = 0.90\n",
        "    model_cfg.graph.max_neighbors = 50\n",
        "    model_cfg.graph.batch_size = 50000\n",
        "    \n",
        "    builder = GraphBuilder(config=model_cfg)\n",
        "    \n",
        "    # 1. Recover original splits from X_full (since we merged them)\n",
        "    # We need to act as if we have X_train, X_val, X_test\n",
        "    # Note: GraphBuilder expects tensors or numpy arrays\n",
        "    \n",
        "    # In this notebook, we lost the split indices in cell 13's context if we ran cell 10's cleanup.\n",
        "    # But Cell 10 defines n_train, n_val, n_test and keeps X_full.\n",
        "    # We can slice X_full.\n",
        "    \n",
        "    X_train = X_full[:n_train]\n",
        "    X_val = X_full[n_train:n_train+n_val]\n",
        "    X_test = X_full[n_train+n_val:]\n",
        "    \n",
        "    # 2. Build edges using Builder\n",
        "    # Train -> Train\n",
        '    print("  Phase 1: Train -> Train...")\n',
        "    builder.fit(X_train)\n",
        "    \n",
        "    # Val/Test -> Train\n",
        '    print("  Phase 2: Val/Test -> Train...")\n',
        "    edge_index = builder.transform(torch.cat([X_val, X_test]), train_size=n_train)\n",
        "    \n",
        "    # Verify\n",
        "    builder.verify_no_leakage(edge_index, train_size=n_train)\n",
        "    \n",
        "    # Save\n",
        "    torch.save(edge_index, GRAPH_CACHE)\n",
        '    print(f"✓ Saved to {GRAPH_CACHE}")\n',
        "    \n",
        "    # Cleanup\n",
        "    del builder\n",
        "    gc.collect()\n",
        "    \n",
        "    # Move to device\n",
        "    X_full = X_full.to(device)\n",
        "    edge_index = edge_index.to(device)\n",
        '    print(f"\\nFinal Graph ready on {device}")\n',
    ]
    graph_cell["source"] = new_source
    print("Updated Graph Cell (Using GraphBuilder)")

# 3. Update Training (Loss Function)
if train_cell:
    source_lines = train_cell["source"]
    new_source = []
    skip = False
    for line in source_lines:
        if "optimizer = torch.optim.Adam" in line:
            new_source.append(line)
            # Inject new loss setup
            new_source.append("\n")
            new_source.append("# CrossEntropy with 15x penalty for fraud (Verified Local Fix)\n")
            new_source.append("weights = torch.tensor([1.0, 15.0]).to(device)\n")
            new_source.append("criterion = torch.nn.CrossEntropyLoss(weight=weights)\n")
            skip = True  # Skip the old loss definition lines until we hit blank/comments
        elif skip and ("criterion =" in line or "weights =" in line):
            continue
        elif skip and line.strip() == "":
            skip = False
            new_source.append(line)
        elif skip and ("# Training config" in line):
            skip = False
            new_source.append(line)
        elif not skip:
            new_source.append(line)

    train_cell["source"] = new_source
    print("Updated Training Cell (CrossEntropy, 15x)")

# Save
with open(nb_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=4)
