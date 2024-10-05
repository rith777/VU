from sympy import *
from tabulate import tabulate

p, q, r, s, t = symbols('p q r s t') #declare the symboles used

def new_basic_table():
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

    # remove these three lines to always show all five symbols
    irrelevant = [i for i in list(table.keys()) if i not in [str(i) for i in symbols]]
    for symbol in irrelevant:
        del table[symbol]

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

table = new_basic_table()
print(tabulate(table, headers="keys"))

print(tte("T"))  # Should print True
print(tte("F"))  # Should print False

table = new_basic_table()
statement = And(p, q)  # This is the logical statement: p AND q
table = add_logical_statement(table, statement)

print(tabulate(table, headers="keys"))