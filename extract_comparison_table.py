#!/usr/bin/env python3
"""
Extract performance metrics from pre-run notebook outputs and generate Markdown comparison tables.
No re-simulation required — only parses existing data.
S0-S3: compare A vs A0. S4: compare PD / A / PD+AFT / A+AFT.
Smaller values are automatically bolded.

Usage:
    python gen_comparison_table.py                          # default: S0-S4
    python gen_comparison_table.py --transient              # S0-S3 with transient metrics
    python gen_comparison_table.py -o tables/my-table.md    # save to file
"""

import argparse
import json
import math
import re
import os

# ============================================================
# Notebook -> Scenario/Controller mapping
# ============================================================
NOTEBOOK_A = "Flex_TD3_A_Evaluate.ipynb"
NOTEBOOK_A0 = "Flex_TD3_A0_Evaluate.ipynb"

SCENARIO_MODES = {
    "Scenario0": "S0",
    "Scenario1": "S1",
    "Scenario2": "S2",
    "Scenario3": "S3",
    "Scenario4": "S4",

    "test2": "test2",
    "test3": "test3",
    "test5": "test5",
    "train": "train",
}


def identify_cell(source_lines):
    """Identify (scenario, controller_type, is_A0) from cell source"""
    source = "".join(source_lines)
    mode_match = re.search(r"mode\s*=\s*'(\w+)'", source)
    mode = mode_match.group(1) if mode_match else None
    scenario = SCENARIO_MODES.get(mode, mode)

    is_A0 = "use_eta_reward=False" in source

    if "controllerTD3AFT" in source:
        ctrl = "A+AFT" if not is_A0 else "A0+AFT"
    elif "controllerAFT()" in source or "controllerAFT(" in source:
        ctrl = "PD+AFT"
    elif "nominal_control" in source:
        ctrl = "PD"
    elif "agent.take_action" in source:
        ctrl = "A0" if is_A0 else "A"
    else:
        ctrl = None

    return scenario, ctrl, is_A0


# ============================================================
# Output parsing
# ============================================================
def parse_metrics_from_output(outputs):
    """Extract all performance metrics from cell output text"""
    text = ""
    for out in outputs:
        if out.get("output_type") == "stream":
            text += "".join(out.get("text", []))

    metrics = {}

    # ---- performance_old ----
    m = re.search(r"Earliest time MRP converges within 0\.01\s*\(s\):\s*([\d.]+)", text)
    if m:
        metrics["conv_001_old"] = float(m.group(1))
    m = re.search(r"Maximum modal displacement:\s*([\d.]+)", text)
    if m:
        metrics["eta_peak_old"] = float(m.group(1))
    m = re.search(r"Earliest time modal displacement converges within 0\.01\s*\(s\):\s*(\d+)", text)
    if m:
        metrics["eta_conv_01"] = int(m.group(1))
    elif "Earliest time modal displacement converges within 0.01" in text:
        metrics["eta_conv_01"] = None

    # ---- performance_steadystate ----
    m = re.search(r"SS MRP \|\|p\|\|_inf\s*:\s*([\d.e+\-]+)", text)
    if m:
        metrics["ss_mrp_norm"] = float(m.group(1))
    m = re.search(r"SS Ang\. Velo\. \|\|ω\|\|_inf\s*:\s*([\d.e+\-]+)", text)
    if m:
        metrics["ss_omega_norm"] = float(m.group(1))
    m = re.search(r"SS Modal Disp\. RMS\s*:\s*([\d.e+\-]+)", text)
    if m:
        metrics["ss_eta_rms"] = float(m.group(1))
    m = re.search(r"SS Control Torque RMS\s*:\s*([\d.e+\-]+)", text)
    if m:
        metrics["ss_torque_rms"] = float(m.group(1))

    # ---- performance_transient ----
    m = re.search(r"MRP converge to 0\.01 time\s*:\s*(\d+)s", text)
    if m:
        metrics["converge_time_01"] = int(m.group(1))
    m = re.search(r"MRP converge to 0\.005 time\s*:\s*(\d+)s", text)
    if m:
        metrics["converge_time_005"] = int(m.group(1))
    m = re.search(r"Full Ang\. Velo\. \|\|ω\|\|_inf\s*:\s*([\d.e+\-]+)", text)
    if m:
        metrics["omega_inf_max"] = float(m.group(1))
    m = re.search(r"Full Modal Disp\. RMS\s*:\s*([\d.e+\-]+)", text)
    if m:
        metrics["eta_rms"] = float(m.group(1))
    m = re.search(r"Full Control Torque RMS\s*:\s*([\d.e+\-]+)", text)
    if m:
        metrics["torque_rms"] = float(m.group(1))

    # transient may say "Nones" for convergence
    if "Nones" in text:
        if "converge_time_01" not in metrics:
            metrics["converge_time_01"] = None
        if "converge_time_005" not in metrics:
            metrics["converge_time_005"] = None

    return metrics if metrics else None


# ============================================================
# Table construction
# ============================================================
def fmt_sci(x, ndigits=2):
    if x is None:
        return "---"
    if abs(x) < 1e-15:
        return "$0$"
    if 0.1 <= abs(x) < 100:
        return f"{x:.3f}"
    exp = int(math.floor(math.log10(abs(x))))
    mantissa = x / 10**exp
    return f"${mantissa:.{ndigits}f}{{\\times}}10^{{{exp}}}$"


def fmt_float(x, ndigits=3):
    if x is None:
        return "---"
    return f"{x:.{ndigits}f}"


def _is_better(v, v_other):
    """v is better if strictly less than v_other (smaller is better). No judgment if either is None."""
    return v is not None and v_other is not None and v < v_other


# ============================================================
# LaTeX formatting
# ============================================================
def fmt_sci_tex(x, ndigits=2):
    """Scientific notation LaTeX format: $1.74\\times10^{-2}$"""
    if x is None:
        return "---"
    if abs(x) < 1e-15:
        return "$0$"
    if 0.1 <= abs(x) < 100:
        return f"{x:.3f}"
    exp = int(math.floor(math.log10(abs(x))))
    mantissa = x / 10**exp
    return f"${mantissa:.{ndigits}f}\\times10^{{{exp}}}$"


def fmt_int_tex(x):
    """Integer LaTeX format, None -> ---"""
    if x is None:
        return "---"
    return str(x)


def _bold_tex(s):
    """Bold a LaTeX value: $1.23$ -> $\\mathbf{1.23}$, 1.23 -> $\\mathbf{1.23}$"""
    if s == "---":
        return s
    if s.startswith("$") and s.endswith("$"):
        return f"$\\mathbf{{{s[1:-1]}}}$"
    return f"$\\mathbf{{{s}}}$"


# ============================================================
# Markdown tables
# ============================================================


def build_table_s1_s3(data, scenarios, controllers, transient=False):
    """
    S0-S3 style: rows = Scenario×Agent, cols = metrics.
    Within each Scenario, A vs A0 per-metric comparison, smaller value bolded.
    """
    steady_cols = [
        ("$\\|p\\|_\\infty$",              "ss_mrp_norm",    fmt_sci),
        ("$\\|\\omega\\|_\\infty$ (rad/s)", "ss_omega_norm",  fmt_sci),
        ("$\\eta$ RMS",                     "ss_eta_rms",     fmt_sci),
        ("$\\eta$ conv. to 0.01 (s)",       "eta_conv_01",    lambda x: "---" if x is None else str(x)),
        ("$T$ RMS (Nm)",                    "ss_torque_rms",  fmt_sci),
    ]
    transient_cols = [
        ("Peak $\\|\\omega\\|_\\infty$ (rad/s)", "omega_inf_max",     fmt_sci),
        ("Integ. $\\eta$ RMS",                   "eta_rms",           fmt_sci),
        ("Conv. to 0.01 (s)",                    "converge_time_01",  fmt_float),
        ("Integ. $T$ RMS (Nm)",                  "torque_rms",        fmt_float),
    ]
    col_defs = steady_cols + (transient_cols if transient else [])

    header = "| Scenario | Agent | " + " | ".join(c[0] for c in col_defs) + " |"
    n_cols = 2 + len(col_defs)
    sep = "|" + "|".join([" --- "] * n_cols) + "|"
    lines = [header, sep]

    for si, s_name in enumerate(scenarios):
        vals = [data.get((s_name, cl)) for cl, _ in controllers]
        for ri, (c_label, _) in enumerate(controllers):
            val = vals[ri]
            other = vals[1 - ri]
            row = f"| {s_name} | {c_label} |"
            if val is None:
                row += " N/A |" * len(col_defs)
            else:
                for _, key, fmter in col_defs:
                    v = val.get(key)
                    v_other = other.get(key) if other else None
                    s = fmter(v)
                    if _is_better(v, v_other):
                        s = f"**{s}**"
                    row += f" {s} |"
            lines.append(row)
        if si < len(scenarios) - 1:
            lines.append("|" + "|".join([" --- "] * n_cols) + "|")

    return "\n".join(lines)


def build_table_s4(data, scenario, controllers):
    """
    S4 style: rows = metrics, cols = controllers (matching paper/scenario4-table.tex)
    Comparison pairs: PD vs A (no AFT), PD+AFT vs A+AFT (with AFT), smaller value bolded.
    controllers order must be [PD, A, PD+AFT, A+AFT].
    """
    steady_rows = [
        ("MRP $\\|p\\|_\\infty$",            "ss_mrp_norm",    fmt_sci),
        ("$\\|\\omega\\|_\\infty$ (rad/s)",   "ss_omega_norm",  fmt_sci),
        ("$\\eta$ RMS",                       "ss_eta_rms",     fmt_sci),
        ("Torque RMS (Nm)",                   "ss_torque_rms",  fmt_sci),
    ]
    transient_rows = [
        ("Peak $\\|\\eta\\|_\\infty$",              "eta_peak_old",     fmt_sci),
        ("Peak $\\|\\omega\\|_\\infty$ (rad/s)",     "omega_inf_max",    fmt_sci),
        ("Integrated $\\eta$ RMS",                   "eta_rms",          fmt_sci),
        ("MRP converge to $0.005$ (s)",              "converge_time_005", lambda x: "---" if x is None else str(x)),
        ("Integrated Torque RMS (Nm)",               "torque_rms",       fmt_float),
    ]
    # Comparison pairs: (PD, A) and (PD+AFT, A+AFT)
    pairs = [(0, 1), (2, 3)]

    n_ctrl = len(controllers)
    header = "| Metric | " + " | ".join(c[0] for c in controllers) + " |"
    sep = "|" + "|".join([" --- "] * (n_ctrl + 1)) + "|"
    lines = [header, sep]

    def add_section(title):
        lines.append(f"| **{title}** |" + " | ".join([""] * n_ctrl) + " |")

    def build_metric_row(label, key, fmter):
        ctrl_vals = []
        for _, cl in controllers:
            val = data.get((scenario, cl))
            ctrl_vals.append(val.get(key) if val else None)

        row = f"| {label} |"
        for ci in range(n_ctrl):
            v = ctrl_vals[ci]
            best = any(
                (ci == a and _is_better(v, ctrl_vals[b])) or
                (ci == b and _is_better(v, ctrl_vals[a]))
                for a, b in pairs
            )
            s = fmter(v)
            if best:
                s = f"**{s}**"
            row += f" {s} |"
        lines.append(row)

    add_section("Steady-state ($t > 160$ s)")
    for label, key, fmter in steady_rows:
        build_metric_row(label, key, fmter)

    add_section("Transient / full-trajectory")
    for label, key, fmter in transient_rows:
        build_metric_row(label, key, fmter)

    return "\n".join(lines)


# ============================================================
# LaTeX tables
# ============================================================
def build_table_s1_s3_tex(data, scenarios, controllers):
    """
    S0-S3 LaTeX three-line table: rows = Scenario×Agent, cols = metrics.
    A vs A0 per-metric comparison, smaller value \mathbf bolded.
    """
    col_defs = [
        (r"$\|p\|_\infty$",              "ss_mrp_norm",    fmt_sci_tex),
        (r"$\|\omega\|_\infty$ (rad/s)", "ss_omega_norm",  fmt_sci_tex),
        (r"$\eta$ RMS",                  "ss_eta_rms",     fmt_sci_tex),
        (r"$\eta_{0.01}$ (s)",           "eta_conv_01",    fmt_int_tex),
        (r"$T$ RMS (Nm)",                "ss_torque_rms",  fmt_sci_tex),
    ]
    n_metric = len(col_defs)
    n_col = 2 + n_metric  # Scenario + Agent + metrics

    lines = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\caption{Quantitative comparison of Agent A vs A0 across scenarios S0--S3 (steady-state, $t > 160$\,s).}")
    lines.append(r"\label{tab:s0s3}")
    lines.append(r"\begin{tabular}{" + "l" * 2 + "c" * n_metric + "}")
    lines.append(r"\toprule")
    # header
    header_cells = [r"\textbf{Scenario}", r"\textbf{Agent}"] + [r"\textbf{" + c[0] + "}" for c in col_defs]
    lines.append(" & ".join(header_cells) + r" \\")
    lines.append(r"\midrule")

    for si, s_name in enumerate(scenarios):
        vals = [data.get((s_name, cl)) for cl, _ in controllers]
        for ri, (c_label, _) in enumerate(controllers):
            val = vals[ri]
            other = vals[1 - ri]
            if ri == 0:
                cells = [f"\\multirow{{2}}{{*}}{{{s_name}}}", c_label]
            else:
                cells = ["", c_label]
            if val is None:
                cells += ["N/A"] * n_metric
            else:
                for _, key, fmter in col_defs:
                    v = val.get(key)
                    v_other = other.get(key) if other else None
                    s = fmter(v)
                    if _is_better(v, v_other):
                        s = _bold_tex(s)
                    cells.append(s)
            lines.append(" & ".join(cells) + r" \\")
        if si < len(scenarios) - 1:
            lines.append(r"\cmidrule(lr){{1-{}}}".format(n_col))

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def build_table_s4_tex(data, scenario, controllers):
    """
    S4 LaTeX three-line table: rows = metrics, cols = controllers.
    PD vs A, PD+AFT vs A+AFT pairwise comparison, smaller value \mathbf bolded.
    """
    steady_rows = [
        (r"MRP $\|p\|_\infty$",            "ss_mrp_norm",    fmt_sci_tex),
        (r"$\|\omega\|_\infty$ (rad/s)",   "ss_omega_norm",  fmt_sci_tex),
        (r"$\eta$ RMS",                    "ss_eta_rms",     fmt_sci_tex),
        (r"Torque RMS (Nm)",               "ss_torque_rms",  fmt_sci_tex),
    ]
    transient_rows = [
        (r"Peak $\|\eta\|_\infty$",              "eta_peak_old",     fmt_sci_tex),
        (r"Peak $\|\omega\|_\infty$ (rad/s)",    "omega_inf_max",    fmt_sci_tex),
        (r"Integrated $\eta$ RMS",               "eta_rms",          fmt_sci_tex),
        (r"MRP converge to $0.005$ (s)",         "converge_time_005", fmt_int_tex),
        (r"Integrated Torque RMS (Nm)",          "torque_rms",       fmt_sci_tex),
    ]
    pairs = [(0, 1), (2, 3)]
    n_ctrl = len(controllers)
    n_col = n_ctrl + 1

    lines = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\caption{Quantitative comparison of four control methods under Scenario~4 (actuator faults).}")
    lines.append(r"\label{tab:s4}")
    lines.append(r"\begin{tabular}{l" + "c" * n_ctrl + "}")
    lines.append(r"\toprule")
    # header with cmidrule grouping
    ctrl_names = [c[0] for c in controllers]
    lines.append("& " + " & ".join(f"\\textbf{{{n}}}" for n in ctrl_names) + r" \\")
    lines.append(r"\cmidrule(lr){1-3} \cmidrule(lr){4-5}")
    # steady-state sub-header
    lines.append(r"\multicolumn{{{}}}{{l}}{{\textit{{Steady-state ($t > 160$\,s)}}}} \\".format(n_col))
    lines.append(r"\cmidrule(lr){1-3} \cmidrule(lr){4-5}")

    def build_section(rows):
        for label, key, fmter in rows:
            ctrl_vals = []
            for _, cl in controllers:
                val = data.get((scenario, cl))
                ctrl_vals.append(val.get(key) if val else None)
            cells = [label]
            for ci in range(n_ctrl):
                v = ctrl_vals[ci]
                best = any(
                    (ci == a and _is_better(v, ctrl_vals[b])) or
                    (ci == b and _is_better(v, ctrl_vals[a]))
                    for a, b in pairs
                )
                s = fmter(v)
                if best:
                    s = _bold_tex(s)
                cells.append(s)
            lines.append(" & ".join(cells) + r" \\")

    build_section(steady_rows)

    # transient sub-header
    lines.append(r"\cmidrule(lr){1-3} \cmidrule(lr){4-5}")
    lines.append(r"\multicolumn{{{}}}{{l}}{{\textit{{Transient / full-trajectory}}}} \\".format(n_col))
    lines.append(r"\cmidrule(lr){1-3} \cmidrule(lr){4-5}")

    build_section(transient_rows)

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def write_tex_document(s1s3_table, s4_table, output_path):
    """Wrap two LaTeX tables into a complete compilable document."""
    preamble = r"""\documentclass[a4paper,11pt]{article}

\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage{amsmath,amssymb,bm}
\usepackage{booktabs}
\usepackage{multirow}
\usepackage[margin=2.5cm]{geometry}

\begin{document}
"""
    closing = r"""
\end{document}
"""
    with open(output_path, "w") as f:
        f.write(preamble)
        f.write(s1s3_table + "\n\n\n")
        f.write(s4_table + "\n")
        f.write(closing)


# ============================================================
# Parse notebooks
# ============================================================
def parse_notebook(nb_path):
    """Parse a notebook, return {(scenario, ctrl_label): metrics}"""
    with open(nb_path) as f:
        nb = json.load(f)

    results = {}
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        scenario, ctrl, _ = identify_cell(cell["source"])
        if scenario is None or ctrl is None:
            continue
        if "performance_steadystate" not in source:
            continue

        metrics = parse_metrics_from_output(cell.get("outputs", []))
        if metrics:
            results[(scenario, ctrl)] = metrics
    return results


def main():
    parser = argparse.ArgumentParser(description="Extract performance tables from notebook outputs")
    parser.add_argument("--scenarios", nargs="+",
                        default=["S0", "S1", "S2", "S3"],
                        help="S0-S3 scenario list (default: S0 S1 S2 S3)")
    parser.add_argument("--transient", action="store_true",
                        help="Include S0-S3 transient/full-trajectory metrics")
    parser.add_argument("--output", "-o", type=str,
                        default="paper/s0-s4-comparison.md",
                        help="Output file path (default: paper/s0-s4-comparison.md)")
    args = parser.parse_args()

    base = os.path.dirname(__file__) or "."
    data_a = parse_notebook(os.path.join(base, NOTEBOOK_A))
    data_a0 = parse_notebook(os.path.join(base, NOTEBOOK_A0))
    data_all = {**data_a, **data_a0}

    # ---- S0-S3: A vs A0 ----
    s1s3_ctrls = [("A", "A"), ("A0", "A0")]
    table_s1s3 = build_table_s1_s3(data_all, args.scenarios, s1s3_ctrls, args.transient)

    # ---- S4: PD / A / PD+AFT / A+AFT ----
    s4_ctrls = [("PD", "PD"), ("TD3", "A"), ("PD+AFT", "PD+AFT"), ("TD3+AFT", "A+AFT")]
    table_s4 = build_table_s4(data_all, "S4", s4_ctrls)

    full_output = table_s1s3 + "\n\n" + table_s4
    print(full_output)

    if args.output:
        with open(args.output, "w") as f:
            f.write(full_output + "\n")
        print(f"\nMarkdown saved: {args.output}")

        # Also output LaTeX file
        tex_path = os.path.splitext(args.output)[0] + ".tex"
        table_s1s3_tex = build_table_s1_s3_tex(data_all, args.scenarios, s1s3_ctrls)
        table_s4_tex = build_table_s4_tex(data_all, "S4", s4_ctrls)
        write_tex_document(table_s1s3_tex, table_s4_tex, tex_path)
        print(f"LaTeX saved:   {tex_path}")


if __name__ == "__main__":
    main()
