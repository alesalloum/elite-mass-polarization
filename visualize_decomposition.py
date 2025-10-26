import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import matplotlib.gridspec as gridspec
import numpy as np

# Cache to store loaded graphs
graph_cache = {}


def load_graph(net):
    """Load a graph only once and store it in memory."""
    if net not in graph_cache:
        graph_cache[net] = nx.read_graphml(f"./networks/hierarchical/{net}.graphml")
    return graph_cache[net]


def return_manual_legend_hierarchical(ax):
    """Return a legend for the AEI decomposition plot."""
    legend_items = [
        ("Elite Cohesion A ($\\hat{i}_{c_{A}}$)", "#990000"),
        ("Elite Cohesion B ($\\hat{i}_{c_{B}}$)", "#1c4587"),
        ("Mass Amplification A ($\\hat{i}_{cp_{A}}$)", "#c27ba0"),
        ("Mass Amplification B ($\\hat{i}_{cp_{B}}$)", "#0097a7"),
        ("Mass Cohesion A ($\\hat{i}_{p_{A}}$)", "#fabedd"),
        ("Mass Cohesion B ($\\hat{i}_{p_{B}}$)", "#d0f2f0"),
        ("Bridge ($2\\hat{e}_{AB}$)", "darkgreen"),
    ]
    legend_patches = [Patch(color=color, label=label) for label, color in legend_items]
    return ax.legend(
        handles=legend_patches,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.15),
        ncol=2,
        fontsize=14,
    )


def return_manual_legend_groups(ax):
    """Return a legend for the group sizes plot."""
    legend_items = [
        ("Elite A", "#990000"),
        ("Elite B", "#1c4587"),
        ("Mass A", "#c27ba0"),
        ("Mass B", "#0097a7"),
    ]
    legend_patches = [Patch(color=color, label=label) for label, color in legend_items]
    return ax.legend(
        handles=legend_patches,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.15),
        ncol=2,
        fontsize=14,
    )


def get_polarization_segments_hierarchical(polarization_spectrum):
    """
    Organize the polarization spectrum into segments for plotting.
    The positive segments are for within-group components and the negative segment for the bridge.
    """
    A_elite_cohesion = polarization_spectrum["i_cA"]
    A_public_amplification = polarization_spectrum["i_cpA"]
    A_public_cohesion = polarization_spectrum["i_pA"]
    B_elite_cohesion = polarization_spectrum["i_cB"]
    B_public_amplification = polarization_spectrum["i_cpB"]
    B_public_cohesion = polarization_spectrum["i_pB"]
    AB_bridge = -(
        polarization_spectrum["e_cAB"]
        + polarization_spectrum["e_cpAB"]
        + polarization_spectrum["e_pAB"]
    )

    positive_segments = [
        (A_public_cohesion, "#fabedd"),
        (A_public_amplification, "#c27ba0"),
        (A_elite_cohesion, "#990000"),
        (B_elite_cohesion, "#1c4587"),
        (B_public_amplification, "#0097a7"),
        (B_public_cohesion, "#d0f2f0"),
    ]
    negative_segments = [(AB_bridge, "darkgreen")]

    # Compute cumulative widths for stacking
    pos_cumulative_widths = [0]
    for seg_val, _ in positive_segments:
        pos_cumulative_widths.append(pos_cumulative_widths[-1] + seg_val)

    neg_cumulative_widths = [0]
    for seg_val, _ in negative_segments:
        neg_cumulative_widths.append(neg_cumulative_widths[-1] + seg_val)

    return (
        positive_segments,
        negative_segments,
        pos_cumulative_widths,
        neg_cumulative_widths,
    )


def get_hierarchical_groups(G):
    """Categorize nodes of G into the four hierarchical groups."""
    hierarchy_attr = nx.get_node_attributes(G, "polhierarchy-sbm")
    groups = {
        "A_CORE": set(),
        "A_PERIPHERY": set(),
        "B_CORE": set(),
        "B_PERIPHERY": set(),
    }
    for node, hierarchy in hierarchy_attr.items():
        if hierarchy in groups:
            groups[hierarchy].add(node)
        else:
            print(f"Warning: Unrecognized hierarchy '{hierarchy}' for node '{node}'.")
    assert len(G) == sum(
        len(g) for g in groups.values()
    ), "Some nodes were not categorized."
    return groups


def get_spectrum_components_haei(net):
    """
    Calculate polarization spectrum components using the HAEI method.
    The network file is read from './networks/hierarchical/{net}.graphml'.
    """
    G = load_graph(net)
    hgroups = get_hierarchical_groups(G)

    # Define groups for A and B
    A_nodes = hgroups["A_CORE"] | hgroups["A_PERIPHERY"]
    B_nodes = hgroups["B_CORE"] | hgroups["B_PERIPHERY"]
    nA, nB = len(A_nodes), len(B_nodes)

    # Group A components
    A_CORE = nx.induced_subgraph(G, hgroups["A_CORE"])
    A_PERIPHERY = nx.induced_subgraph(G, hgroups["A_PERIPHERY"])
    I_cA = A_CORE.number_of_edges()
    I_cpA = nx.cut_size(G, hgroups["A_CORE"], hgroups["A_PERIPHERY"])
    I_pA = A_PERIPHERY.number_of_edges()
    i_cA = I_cA / (0.5 * nA * (nA - 1))
    i_cpA = I_cpA / (0.5 * nA * (nA - 1))
    i_pA = I_pA / (0.5 * nA * (nA - 1))

    # Group B components
    B_CORE = nx.induced_subgraph(G, hgroups["B_CORE"])
    B_PERIPHERY = nx.induced_subgraph(G, hgroups["B_PERIPHERY"])
    I_cB = B_CORE.number_of_edges()
    I_cpB = nx.cut_size(G, hgroups["B_CORE"], hgroups["B_PERIPHERY"])
    I_pB = B_PERIPHERY.number_of_edges()
    i_cB = I_cB / (0.5 * nB * (nB - 1))
    i_cpB = I_cpB / (0.5 * nB * (nB - 1))
    i_pB = I_pB / (0.5 * nB * (nB - 1))

    # Crossing components
    E_cAB = nx.cut_size(G, hgroups["A_CORE"], hgroups["B_CORE"])
    E_cpAB = nx.cut_size(G, hgroups["A_CORE"], hgroups["B_PERIPHERY"]) + nx.cut_size(
        G, hgroups["B_CORE"], hgroups["A_PERIPHERY"]
    )
    E_pAB = nx.cut_size(G, hgroups["A_PERIPHERY"], hgroups["B_PERIPHERY"])
    e_cAB = E_cAB / (nA * nB)
    e_cpAB = E_cpAB / (nA * nB)
    e_pAB = E_pAB / (nA * nB)

    # Overall mass used for normalization
    OVERALL_MASS = (
        (i_cA + i_cpA + i_pA) + (i_cB + i_cpB + i_pB) + 2 * (e_cAB + e_cpAB + e_pAB)
    )

    # Normalize components
    polarization_spectrum = {
        "i_cA": i_cA / OVERALL_MASS,
        "i_cpA": i_cpA / OVERALL_MASS,
        "i_pA": i_pA / OVERALL_MASS,
        "i_cB": i_cB / OVERALL_MASS,
        "i_cpB": i_cpB / OVERALL_MASS,
        "i_pB": i_pB / OVERALL_MASS,
        "e_cAB": (2 * e_cAB) / OVERALL_MASS,
        "e_cpAB": (2 * e_cpAB) / OVERALL_MASS,
        "e_pAB": (2 * e_pAB) / OVERALL_MASS,
    }

    return polarization_spectrum


def get_hierarchy_widths(net):
    """
    Get the proportion (in percent) of nodes in each group.
    Returns an array: [elite_left, public_left, elite_right, public_right].
    """
    G = load_graph(net)
    hgroups = get_hierarchical_groups(G)
    total = len(G)
    elite_left = len(hgroups["A_CORE"]) / total
    public_left = len(hgroups["A_PERIPHERY"]) / total
    elite_right = len(hgroups["B_CORE"]) / total
    public_right = len(hgroups["B_PERIPHERY"]) / total
    return np.array([elite_left, public_left, elite_right, public_right]) * 100


labels = ["EDUCATION", "ECONOMY", "SOCIAL", "IMMIGRATION", "CLIMATE"]
topics = [
    ("education_19", "education_23"),
    ("economy_19", "economy_23"),
    ("social_19", "social_23"),
    ("immigration_19", "immigration_23"),
    ("climate_19", "climate_23"),
]

fig = plt.figure(figsize=(12, 6.5))
gs = gridspec.GridSpec(1, 2, width_ratios=[2, 1])

# Left plot: Decomposition of AEI
ax1 = fig.add_subplot(gs[0, 0])
ax1.set_xlabel("Decomposition of AEI", fontsize=15)
ax1.tick_params(axis="x", labelsize=14)
ax1.set_yticks(np.linspace(0, 5, len(labels)) + 0.55)
ax1.set_yticklabels(labels, rotation=30, fontweight="bold")
ax1.set_xlim(-0.15, 1.05)
ax1.set_ylim(-0.25, 6)
return_manual_legend_hierarchical(ax1)

# Background shading for panels
for y_min, y_max in [(-0.25, 1), (2.25, 3.5), (4.75, 6.25)]:
    ax1.axhspan(y_min, y_max, color="lightgray", alpha=0.8)
ax1.axvline(0, ymin=-0.25, ymax=6, color="black", linestyle="--")
ax1.text(-0.13, 0.7, "2023", va="center", fontweight="bold", fontsize=11)
ax1.text(-0.13, 0.1, "2019", va="center", fontweight="bold", fontsize=11)

# Right plot: Group sizes (%)
ax2 = fig.add_subplot(gs[0, 1])
for y_min, y_max in [(-0.25, 1), (2.25, 3.5), (4.75, 6.25)]:
    ax2.axhspan(y_min, y_max, color="lightgray", alpha=0.8)
ax2.set_xlabel("Group sizes (%)", fontsize=15)
ax2.tick_params(axis="x", labelsize=14)
ax2.set_ylim(-0.25, 6)
ax2.set_xlim(-100, 100)
return_manual_legend_groups(ax2)
ax2.axvline(0, ymin=-0.25, ymax=6, color="black", linestyle="--")
ax2.tick_params(axis="y", which="both", left=False, labelleft=False)

# Loop over topics (each corresponding to a row)
for t_idx, (net_19, net_23) in zip(np.linspace(0, 5, len(labels)), topics):
    # Get spectrum components and segments for 2019 and 2023
    spec_19 = get_spectrum_components_haei(net_19)
    spec_23 = get_spectrum_components_haei(net_23)

    pos_seg19, neg_seg19, pos_cum19, neg_cum19 = get_polarization_segments_hierarchical(
        spec_19
    )
    pos_seg23, neg_seg23, pos_cum23, neg_cum23 = get_polarization_segments_hierarchical(
        spec_23
    )

    # Compute shift (from negative segments) once per year
    shift19 = neg_cum19[-1]
    shift23 = neg_cum23[-1]

    # Plot positive segments for both years
    for i in range(len(pos_seg19)):
        ax1.barh(
            y=t_idx + 0.2,
            width=pos_seg19[i][0],
            left=pos_cum19[i] + shift19,
            color=pos_seg19[i][1],
            height=0.2,
        )
        ax1.barh(
            y=t_idx + 0.75,
            width=pos_seg23[i][0],
            left=pos_cum23[i] + shift23,
            color=pos_seg23[i][1],
            height=0.2,
        )

    # Plot negative segments for both years
    ax1.barh(
        y=t_idx,
        width=neg_seg19[0][0],
        left=neg_cum19[0],
        color=neg_seg19[0][1],
        height=0.2,
    )
    ax1.barh(
        y=t_idx + 0.6,
        width=neg_seg23[0][0],
        left=neg_cum23[0],
        color=neg_seg23[0][1],
        height=0.2,
    )

    # Plot group sizes for both years
    # 2019 group sizes
    size_19 = get_hierarchy_widths(net_19)
    left_vals_19 = [-size_19[i] for i in range(2)]
    right_vals_19 = size_19[2:]
    left_cum = 0
    for value, color in zip(left_vals_19, ["#990000", "#c27ba0"]):
        ax2.barh(t_idx + 0.2, width=value, color=color, left=left_cum, height=0.2)
        left_cum += value
    right_cum = 0
    for value, color in zip(right_vals_19, ["#1c4587", "#0097a7"]):
        ax2.barh(t_idx + 0.2, width=value, color=color, left=right_cum, height=0.2)
        right_cum += value

    # 2023 group sizes
    size_23 = get_hierarchy_widths(net_23)
    left_vals_23 = [-size_23[i] for i in range(2)]
    right_vals_23 = size_23[2:]
    left_cum = 0
    for value, color in zip(left_vals_23, ["#990000", "#c27ba0"]):
        ax2.barh(t_idx + 0.75, width=value, color=color, left=left_cum, height=0.2)
        left_cum += value
    right_cum = 0
    for value, color in zip(right_vals_23, ["#1c4587", "#0097a7"]):
        ax2.barh(t_idx + 0.75, width=value, color=color, left=right_cum, height=0.2)
        right_cum += value

# Add panel labels
fig.text(0.04, 0.955, "A", va="center", fontsize=20, fontweight="bold")
fig.text(0.6705, 0.955, "B", va="center", fontsize=20, fontweight="bold")

# Modify x-axis tick labels to appear as positive values on both sides
xticks = np.linspace(-100, 100, 5).astype(int)  # Example tick locations
xtick_labels = [f"{abs(x)}" for x in xticks]  # Convert to absolute values

ax2.set_xticks(xticks)
ax2.set_xticklabels(xtick_labels)  # Display only positive values


plt.tight_layout()
plt.subplots_adjust(wspace=0.05)

# Save and show the figure
# plt.savefig('./plots/paper/decomposition_plot.pdf', format='pdf', dpi=300)
plt.show()
