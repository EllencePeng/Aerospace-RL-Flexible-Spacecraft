import json
import os
import base64

notebooks = {
    "A": "Flex_TD3_A_Evaluate.ipynb",
    "A0": "Flex_TD3_A0_Evaluate.ipynb",
}

# (notebook_key, cell_index, filename)
tasks = [
    # A notebook — S0–S4
    ("A", 15, "A-S0.png"),
    ("A", 17, "A-S1.png"),
    ("A", 19, "A-S2.png"),
    ("A", 21, "A-S3.png"),
    ("A", 23, "S4-A.png"),
    ("A", 25, "S4-A-AFT-flex-s.png"),
    ("A", 27, "S4-PD-kp30-kd60.png"),
    ("A", 29, "S4-PD-kp30-kd60-AFT-flex-s.png"),
    # A0 notebook — S0–S3 only
    ("A0", 15, "A0-S0.png"),
    ("A0", 17, "A0-S1.png"),
    ("A0", 19, "A0-S2.png"),
    ("A0", 21, "A0-S3.png"),
]

out_dir = "Pics_MDPI"
os.makedirs(out_dir, exist_ok=True)

for nb_key, cell_idx, filename in tasks:
    nb_path = notebooks[nb_key]
    with open(nb_path, "r") as f:
        nb = json.load(f)

    cell = nb["cells"][cell_idx]
    png_found = False

    for output in cell.get("outputs", []):
        if output.get("output_type") != "display_data":
            continue
        data = output.get("data", {})
        png_b64 = data.get("image/png")
        if png_b64:
            png_bytes = base64.b64decode(png_b64)
            filepath = os.path.join(out_dir, filename)
            with open(filepath, "wb") as f:
                f.write(png_bytes)
            print(f"Saved: {filepath} ({len(png_bytes)} bytes)")
            png_found = True
            break  # only take the first PNG (plot_states)

    if not png_found:
        print(f"WARNING: No PNG found in {nb_key} cell {cell_idx} for {filename}")
