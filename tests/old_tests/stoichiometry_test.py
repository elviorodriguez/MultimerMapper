
import multimer_mapper as mm
from src.stoichiometries import initialize_stoichiometry, generate_child

combined_graph = mm_output['combined_graph']

# Initialize
initial_stoich = initialize_stoichiometry(combined_graph)
initial_stoich.plot()

# Generate a single children
children = generate_child(initial_stoich, combined_graph)
children[0].plot()
children[1].plot()


for i in range(1000):
    initial_stoich = initialize_stoichiometry(combined_graph)
    initial_stoich.plot()
    
    children = generate_child(initial_stoich, combined_graph)
    try:
        children[1].plot()
        children[0].plot()
        print(f"DOUBLE!: {i}")
        break
    except IndexError:
        continue

