"""
Update paper figures and tables from new rl_algo_comparison.csv results.

Run after run_rl_only.py and run_rl_improved.py complete:
    python scripts/update_paper_figures.py
"""
from __future__ import annotations
import csv, re, os, sys
from collections import defaultdict

ROOT  = os.path.join(os.path.dirname(__file__), "..")
PAPER = os.path.join(ROOT, "paper", "rl_raw_policy_paper.tex")
ALGO_CSV     = os.path.join(ROOT, "results", "rl_algo_comparison.csv")
IMPROVED_CSV = os.path.join(ROOT, "results", "rl_improved_comparison.csv")

N_VALS = [100, 200, 400, 600, 800, 1000]
MODES  = ["tabular", "tabular_ddqn", "dqn", "ddqn", "ppo"]
IMP_MODES = ["tabular+", "tab_ddqn+", "dqn+", "ddqn+", "ppo"]

# Map CSV mode name → paper legend label
_MODE_LABEL = {
    "tabular":      "Q-learning",
    "tabular_ddqn": "Tab-DDQN",
    "dqn":          "DQN",
    "ddqn":         "DDQN",
    "ppo":          "PPO",
}

# ─────────────────────────────────────────────────────────────────────────────
# Load CSVs
# ─────────────────────────────────────────────────────────────────────────────
def load_algo(path):
    """Returns {(mode, n): row_dict}"""
    data = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            data[(row["mode"], int(row["n"]))] = row
    return data

def load_improved(path):
    """Returns {(label, n): row_dict}  — improved CSV uses 'label' key."""
    data = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            key_col = "label" if "label" in row else "mode"
            data[(row[key_col], int(row["n"]))] = row
    return data

# ─────────────────────────────────────────────────────────────────────────────
# Build coordinate strings
# ─────────────────────────────────────────────────────────────────────────────
def coords(data, mode, metric_mean, metric_ci_lo=None, metric_ci_hi=None):
    pts = []
    for n in N_VALS:
        key = (mode, n)
        if key not in data:
            continue
        v = float(data[key][metric_mean])
        pts.append(f"({n},{v:.4f})")
    return "".join(pts)

# ─────────────────────────────────────────────────────────────────────────────
# Build PDR CI table rows  (Table V in paper)
# ─────────────────────────────────────────────────────────────────────────────
def build_pdr_table(data):
    col_order = ["tabular","tabular_ddqn","dqn","ddqn","ppo"]
    label_map = {
        "tabular":"Q-learn","tabular_ddqn":"Tab-DDQN",
        "dqn":"DQN","ddqn":"DDQN","ppo":"PPO"
    }
    lines = []
    for n in N_VALS:
        cells = []
        best_pdr = max(float(data[(m,n)]["pdr_mean"]) for m in col_order if (m,n) in data)
        for m in col_order:
            key = (m, n)
            if key not in data:
                cells.append("—")
                continue
            mean = float(data[key]["pdr_mean"])
            ci_lo = float(data[key]["pdr_ci_lo"])
            ci_hi = float(data[key]["pdr_ci_hi"])
            half  = (ci_hi - ci_lo) / 2.0
            cell  = f"{mean:.3f}{{\\tiny$\\pm${half:.3f}}}"
            if abs(mean - best_pdr) < 1e-6:
                cell = f"\\textbf{{{cell}}}"
            cells.append(cell)
        lines.append(f"{n}  & " + " & ".join(cells) + r" \\")
    return "\n".join(lines)

# ─────────────────────────────────────────────────────────────────────────────
# Build improvements delta table  (Table VI in paper)
# ─────────────────────────────────────────────────────────────────────────────
def build_improvements_table(base_data, imp_data):
    base_modes = ["tabular","tabular_ddqn","dqn","ddqn","ppo"]
    imp_modes  = ["tabular+","tab_ddqn+","dqn+","ddqn+","ppo"]
    lines = []
    avgs = defaultdict(list)
    for n in N_VALS:
        cells = []
        for bm, im in zip(base_modes, imp_modes):
            bkey = (bm, n)
            ikey = (im, n)
            if bkey not in base_data or ikey not in imp_data:
                cells.append("—")
                continue
            delta = float(imp_data[ikey]["pdr_mean"]) - float(base_data[bkey]["pdr_mean"])
            avgs[im].append(delta)
            sign  = "+" if delta >= 0 else "$-$"
            val   = f"{sign}{abs(delta):.3f}"
            bold  = abs(delta) > 0.010
            cells.append(f"\\textbf{{{val}}}" if bold else val)
        lines.append(f"{n}  & " + " & ".join(cells) + r" \\")
    # Average row
    avg_cells = []
    for im in imp_modes:
        if avgs[im]:
            a = sum(avgs[im]) / len(avgs[im])
            s = "+" if a >= 0 else "$-$"
            v = f"{s}{abs(a):.3f}"
            avg_cells.append(f"\\textbf{{{v}}}" if abs(a) > 0.010 else v)
        else:
            avg_cells.append("—")
    lines.append(r"\midrule")
    lines.append("Avg  & " + " & ".join(avg_cells) + r" \\")
    return "\n".join(lines)

# ─────────────────────────────────────────────────────────────────────────────
# Patch helpers
# ─────────────────────────────────────────────────────────────────────────────
def replace_coordinates(tex, legend_label, new_coords):
    """Replace the coordinate block for the given legend label."""
    # Pattern: \addlegendentry{...<label>...} is the marker AFTER the data.
    # We look for the coordinates block that immediately precedes the legend entry.
    pattern = re.compile(
        r'(} coordinates \{)\s*\n\s*(' +
        re.escape(new_coords[:5]) + r'.*?\n' +   # old coord (first 5 chars match)
        r')(}\s*;\s*\\addlegendentry\{.*?' +
        re.escape(legend_label) + r'.*?\})',
        re.DOTALL
    )
    # Simpler: find the coordinates block just before the legend entry
    # and replace its content.
    entry_pat = re.compile(
        r'(} coordinates \{)\n(\s*\([^)]+\).*?\n)(\s*\};\s*\\addlegendentry\{[^}]*' +
        re.escape(legend_label) + r'[^}]*\})',
        re.DOTALL
    )
    def replacer(m):
        return m.group(1) + "\n  " + new_coords + "\n" + m.group(3)
    new_tex = entry_pat.sub(replacer, tex)
    return new_tex


def replace_table_body(tex, label, new_body):
    """Replace rows between \\midrule and \\bottomrule for given table label."""
    pat = re.compile(
        r'(\\label\{' + re.escape(label) + r'\}.*?\\midrule\n)'
        r'(.*?)'
        r'(\\bottomrule)',
        re.DOTALL
    )
    def replacer(m):
        return m.group(1) + new_body + "\n" + m.group(3)
    return pat.sub(replacer, tex)

# ─────────────────────────────────────────────────────────────────────────────
# Collect key scalar facts for abstract/conclusion text patches
# ─────────────────────────────────────────────────────────────────────────────
def key_facts(base_data):
    facts = {}
    # Best RL PDR at N=400 vs adaptive baseline (0.4291 from existing policy CSV)
    adaptive_400 = 0.4291  # from existing policy_comparison_with_rl.csv
    best_400 = max(float(base_data[(m,400)]["pdr_mean"]) for m in MODES if (m,400) in base_data)
    facts["best_pdr_400"] = best_400
    facts["pct_gain_400_vs_adaptive"] = 100*(best_400 - adaptive_400)/adaptive_400

    # EE gain at N=1000 (Q-learning vs static RAW 6.01)
    static_ee_1000 = 6.01
    q_ee_1000 = float(base_data.get(("tabular",1000),{}).get("ee_kbit_j_mean", 0))
    facts["q_ee_1000"] = q_ee_1000
    facts["pct_ee_gain_1000"] = 100*(q_ee_1000 - static_ee_1000)/static_ee_1000

    # Tab-DDQN supremacy at high N
    tab_ddqn_wins_all = all(
        float(base_data.get(("tabular_ddqn",n),{}).get("pdr_mean",0)) >
        max(0.4291 if n==400 else 0.2484 if n==600 else 0.1699 if n==800 else 0.1262,  # adaptive
            0.3658 if n==400 else 0.2623 if n==600 else 0.1894 if n==800 else 0.1475)  # static
        for n in [400, 600, 800, 1000]
    )
    facts["tab_ddqn_wins_all_ge400"] = tab_ddqn_wins_all
    return facts

# ─────────────────────────────────────────────────────────────────────────────
# Main patch routine
# ─────────────────────────────────────────────────────────────────────────────
def update_tex_coordinates(tex, base_data):
    """Replace pgfplots coordinate blocks for all 5 RL modes in all 3 figures."""
    for mode in MODES:
        label = _MODE_LABEL[mode]
        for metric, col in [("pdr_mean","pdr"), ("avg_delay_ms_mean","delay"), ("ee_kbit_j_mean","ee")]:
            c = coords(base_data, mode, metric)
            if not c:
                continue
            # Find block: } coordinates {\n  (...)(...)...\n}; \addlegendentry{...<label>...}
            pat = re.compile(
                r'(} coordinates \{)\n(\s*' +
                r'(?:\(\d+,[0-9.]+\))+' +
                r'\n)(\};\s*\\addlegendentry\{[^}]*\\textbf\{' +
                re.escape(label) + r'\}[^}]*\})',
                re.DOTALL
            )
            def make_replacer(new_c):
                def replacer(m):
                    return m.group(1) + "\n  " + new_c + "\n" + m.group(3)
                return replacer
            new_tex = pat.sub(make_replacer(c), tex)
            if new_tex != tex:
                tex = new_tex
    return tex


def patch_coordinates_direct(tex, base_data):
    """Direct string replacement of each coordinate block."""
    lines = tex.split('\n')
    result = []
    i = 0
    while i < len(lines):
        line = lines[i]
        # Look for a coordinates line followed by (N,val)(N,val)...
        if '} coordinates {' in line and i+1 < len(lines):
            coord_line = lines[i+1].strip()
            # Check it starts with a coordinate pattern
            if re.match(r'^\(\d+,', coord_line):
                # Peek at legend entry 2 lines ahead
                if i+2 < len(lines):
                    legend_line = lines[i+2].strip()
                    for mode in MODES:
                        label = _MODE_LABEL[mode]
                        if f'\\textbf{{{label}}}' in legend_line or f'textbf{{{label}}}' in legend_line:
                            # Determine which metric from the coord values
                            # PDR coords are in [0,1], delay in [0,30000], EE in [0,75]
                            vals = [float(x) for x in re.findall(r'\((\d+),([0-9.]+)\)', coord_line) and
                                    re.findall(r'[0-9.]+\)', coord_line)]
                            # Just check magnitude of first value
                            first_val_m = re.search(r'\(\d+,([0-9.]+)\)', coord_line)
                            if first_val_m:
                                fv = float(first_val_m.group(1))
                                if fv <= 1.1:
                                    metric = "pdr_mean"
                                elif fv <= 100:
                                    metric = "ee_kbit_j_mean"
                                else:
                                    metric = "avg_delay_ms_mean"
                                new_c = coords(base_data, mode, metric)
                                if new_c:
                                    result.append(line)
                                    result.append("  " + new_c)
                                    i += 2  # skip old coord line
                                    continue
                            break
        result.append(line)
        i += 1
    return '\n'.join(result)

# ─────────────────────────────────────────────────────────────────────────────
# Robust coordinate replacement using regex on the full text
# ─────────────────────────────────────────────────────────────────────────────
def replace_all_coords(tex, base_data):
    """Find every RL addplot block and replace coordinates."""
    for mode in MODES:
        label = _MODE_LABEL[mode]
        bold_label = f"\\textbf{{{label}}}"
        # Match: } coordinates {\n  (<coords>)\n}; \addlegendentry{...<bold_label>...}
        pat = re.compile(
            r'(\} coordinates \{)\n'
            r'([ \t]*(?:\(\d+,[0-9.]+\))+\n)'
            r'(\}; \\addlegendentry\{[^}]*' + re.escape(bold_label) + r'[^}]*\})',
        )
        for metric, check in [
            ("pdr_mean", lambda v: v <= 1.1),
            ("avg_delay_ms_mean", lambda v: v > 100),
            ("ee_kbit_j_mean", lambda v: 1 < v <= 100),
        ]:
            new_c = "  " + coords(base_data, mode, metric)
            if not new_c.strip():
                continue
            def replacer(m, nc=new_c, chk=check):
                old_coord = m.group(2)
                first_val = re.search(r'\(\d+,([0-9.]+)\)', old_coord)
                if first_val and chk(float(first_val.group(1))):
                    return m.group(1) + "\n" + nc + "\n" + m.group(3)
                return m.group(0)
            tex = pat.sub(replacer, tex)
    return tex

# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────
def main():
    if not os.path.exists(ALGO_CSV):
        print(f"ERROR: {ALGO_CSV} not found — run run_rl_only.py first")
        sys.exit(1)

    print("Loading results...")
    base_data = load_algo(ALGO_CSV)
    imp_data  = load_improved(IMPROVED_CSV) if os.path.exists(IMPROVED_CSV) else {}

    print("Reading paper...")
    with open(PAPER) as f:
        tex = f.read()

    print("Patching coordinate blocks...")
    tex = replace_all_coords(tex, base_data)

    print("Patching PDR table (tab:rl_pdr)...")
    new_pdr_body = build_pdr_table(base_data)
    tex = replace_table_body(tex, "tab:rl_pdr", new_pdr_body)

    if imp_data:
        print("Patching improvements table (tab:improvements)...")
        new_imp_body = build_improvements_table(base_data, imp_data)
        tex = replace_table_body(tex, "tab:improvements", new_imp_body)

    print("Writing patched paper...")
    with open(PAPER, 'w') as f:
        f.write(tex)

    # Print key facts for manual text updates
    print("\n=== KEY FACTS FOR MANUAL TEXT UPDATES ===")
    facts = key_facts(base_data)
    print(f"Best PDR at N=400: {facts['best_pdr_400']:.4f}  (+{facts['pct_gain_400_vs_adaptive']:.1f}% vs adaptive)")
    print(f"Q-learning EE at N=1000: {facts['q_ee_1000']:.2f} kbit/J  (+{facts['pct_ee_gain_1000']:.1f}% vs static)")
    print(f"Tab-DDQN wins all N≥400: {facts['tab_ddqn_wins_all_ge400']}")

    print("\n=== PDR MEANS PER MODE × N ===")
    header = f"{'N':>5}  " + "  ".join(f"{_MODE_LABEL[m]:>10}" for m in MODES)
    print(header)
    for n in N_VALS:
        vals = []
        for m in MODES:
            key = (m, n)
            vals.append(f"{float(base_data[key]['pdr_mean']):.4f}" if key in base_data else "  —   ")
        print(f"{n:>5}  " + "  ".join(f"{v:>10}" for v in vals))

    print("\nDone. Review paper for text claims needing manual updates.")

if __name__ == "__main__":
    main()
