from sympy import *
from tabulate import tabulate

# Declare the symbols used
p, q, r, s, t = symbols('p q r s t')

def new_basic_table():
    # Create the basic truth table
    table = {"p": ["T"] * 16 + ["F"] * 16,
             "q": (["T"] * 8 + ["F"] * 8) * 2,
             "r": (["T"] * 4 + ["F"] * 4) * 4,
             "s": (["T"] * 2 + ["F"] * 2) * 8,
             "t": ["T", "F"] * 16}
    return table

def tte(inp):
    if inp == "T":
        return true
    elif inp == "F":
        return false


def add_logical_statement(table, statement):
    row_entries = []
    symbols = list(statement.atoms())

    if len(symbols) == 2:
        sym_1, sym_2 = symbols[0], symbols[1]
        for i in range(32):
            row_entries.append(statement.subs({sym_1: tte(table[str(sym_1)][i]), sym_2: tte(table[str(sym_2)][i])}))

    elif len(symbols) == 3:
        sym_1, sym_2, sym_3 = symbols[0], symbols[1], symbols[2]
        for i in range(32):
            row_entries.append(statement.subs({sym_1: tte(table[str(sym_1)][i]),
                                               sym_2: tte(table[str(sym_2)][i]),
                                               sym_3: tte(table[str(sym_3)][i])}))

    elif len(symbols) == 4:
        sym_1, sym_2, sym_3, sym_4 = symbols[0], symbols[1], symbols[2], symbols[3]
        for i in range(32):
            row_entries.append(statement.subs({sym_1: tte(table[str(sym_1)][i]),
                                               sym_2: tte(table[str(sym_2)][i]),
                                               sym_3: tte(table[str(sym_3)][i]),
                                               sym_4: tte(table[str(sym_4)][i])}))

    elif (len(symbols)) == 5:
        sym_1, sym_2, sym_3, sym_4, sym_5 = symbols[0], symbols[1], symbols[2], symbols[3], symbols[4]
        for i in range(32):
            row_entries.append(statement.subs({sym_1: tte(table[str(sym_1)][i]),
                                               sym_2: tte(table[str(sym_2)][i]),
                                               sym_3: tte(table[str(sym_3)][i]),
                                               sym_4: tte(table[str(sym_4)][i]),
                                               sym_5: tte(table[str(sym_5)][i])}))

    table[statement] = row_entries

    return table

# Countings trues and falses in each column
def count_true_false(table, formula):
    true_count = table[formula].count(True)
    false_count = table[formula].count(False)
    return true_count, false_count

# Define the logical statements A,B
A = And(Implies(Or(p, q, s), And(p, q, r)), Not(s))
B = Implies(Or(t, q), Or(Not(t), Not(s), r))

# Define the combined logical operations
AndAB = And(A, B)
OrAB = Or(A, B)
ImpliesAB = Implies(A, B)
ImpliesBA = Implies(B, A)
customAandB = And(Or(And(p, q, r), And(Not(p), Not(q), r), And(Not(p), Not(q), Not(r))), Not(s)) # For assignment 3



# initialize the basic truth table
table = new_basic_table()

# add formulas to the table
table = add_logical_statement(table, A)
table = add_logical_statement(table, B)
table = add_logical_statement(table, AndAB)
table = add_logical_statement(table, OrAB)
table = add_logical_statement(table, ImpliesAB)
table = add_logical_statement(table, ImpliesBA)
table = add_logical_statement(table, customAandB) # Last column for assignment 3


# Count true and false values for each formula using count definition
formulas = [A, B, AndAB, OrAB, ImpliesAB, ImpliesBA, customAandB]  #added formula for assignment 3
for formula in formulas:
    true_count, false_count = count_true_false(table, formula)
    print(f"Formula: {formula}")
    print(f"True: {true_count}, False: {false_count}\n")

# display the truth table
headers = list(table.keys())
rows = list(zip(*table.values()))
print(tabulate(rows, headers=headers, tablefmt="grid"))
