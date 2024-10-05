# import necessary modules
import networkx as nx
import matplotlib.pyplot as plt

# Sample 3-colorable cycle graph
nodes = 5  # Number of nodes
edges = [(1, 2), (1, 3), (2, 4), (3, 5), (4, 5)]  # List of edges
def plot_graph(edges):
    # Create a graph object
    G = nx.Graph()

    # Add edges to the graph
    G.add_edges_from(edges)

    # Draw the graph
    pos = nx.spring_layout(G)  # Layout for visual spacing
    plt.figure(figsize=(4, 4))

    # Draw nodes and edges with labels
    nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=500, edge_color='black', font_size=14,
            font_weight='bold')

    # Display the plot
    plt.title("Sample Graph Visualization")
    plt.show()


# Call plot_graph function
plot_graph(edges)


def graph_to_cnf(k, nodes, edges):
    """
    Generates a CNF formula for the k-coloring problem on graph G.
    Each variable var(v,c) represents that node v has color c.

    Parameters:
    k (int): The number of colors
    nodes(int): The number nodes of the graph
    edges(list): The number of edges of the graph

    Returns:
    clauses (list of list of int): The CNF clauses
    """

    clauses = []

    ## STEP 1: variable assignment

    # map each node v with color c to a unique variable (integer)
    def var(v, c):
        return (v - 1) * k + c

    ## STEP 2: constraints

    # Constraint 1 and 2: Each node must be assigned EXACTLY one color
    for v in range(1, nodes + 1):
        # each node has to have at least one color
        clauses.append([var(v, c) for c in range(1, k + 1)])
        # a node cannot have two colors at the same time
        for c1 in range(1, k + 1):
            for c2 in range(c1 + 1, k + 1):
                # for each combination of two colors, at most one of these can be true
                clauses.append([-var(v, c1), -var(v, c2)])

    # Constraint 3: No two connected nodes share the same color
    ### TO DO: define this constraint:

    ### SOLUTION ###
    for (v1, v2) in edges:
        for c in range(1, k + 1):
            # v1 and v2 cannot both be color `c`
            clauses.append([-var(v1, c), -var(v2, c)])
    ### END SOLUTION ###

    return clauses


# Run the function with k = 3 (for 3-colorability)
k = 3
clauses = graph_to_cnf(k, nodes, edges)

# Number of variables and clauses
num_variables = nodes * k
num_clauses = len(clauses)

# Writing the CNF file in DIMACS-like format
cnf_output = f"p cnf {num_variables} {num_clauses}\n"
for clause in clauses:
    cnf_output += " ".join(map(str, clause)) + " 0\n"

# Output the CNF formula
print(cnf_output)

# Save the CNF to a file:
with open("NL_graph_4color.cnf", 'w') as f:
    f.write(cnf_output)

import subprocess

result = subprocess.run(['minisat', "NL_graph_4color.cnf", 'sample_output.txt'], capture_output=True)
print(result.stdout.decode())