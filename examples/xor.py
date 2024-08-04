import sys
import progressbar
import numpy as np
import matplotlib.pyplot as plt
sys.path.append('../Argumented_Topology_NN')

from main import Graph, Input_Node, Output_Node, Edge, Node

training_data = [
    [[0, 0], [0]],
    [[0, 1], [1]],
    [[1, 0], [1]],
    [[1, 1], [0]]
]

graph = Graph()

# input and output nodes
graph.add_node(Input_Node("I1"))
graph.add_node(Input_Node("I2"))

graph.add_node(Output_Node("O1"))

graph.input_nodes = [graph.get_node("I1"), graph.get_node("I2")]
graph.output_nodes = [graph.get_node("O1")]

# intermidiate nodes
graph.add_node(Node("H1", activation="sigmoid"))
graph.add_node(Node("H2", activation="sigmoid"))

graph.connect("I1", "H1", np.random.rand())
graph.connect("I2", "H1", np.random.rand())
graph.connect("I1", "H2", np.random.rand())
graph.connect("I2", "H2", np.random.rand())

graph.connect("H1", "O1", np.random.rand())
graph.connect("H2", "O1", np.random.rand())

train_repeats = 10000
with progressbar.ProgressBar(max_value=train_repeats) as bar:
    for i in range(train_repeats):
        graph.train(training_data)
        bar.update(i)

graph.forward([0, 0], 10000)
print(graph.get_output())
graph.forward([1, 0], 10001)
print(graph.get_output())
graph.forward([0, 1], 10002)
print(graph.get_output())
graph.forward([1, 1], 10003)
print(graph.get_output())

# draw heat map
xmax = 100
ymax = 100
x = np.linspace(0, 1, xmax)
y = np.linspace(0, 1, ymax)


Z = np.zeros((xmax, ymax))

index = 0
for i in range(xmax):
    for j in range(ymax):
        graph.forward([x[i], y[j]], 5000000 + index)
        Z[i, j] = graph.get_output()[0]

        index += 1

plt.imshow(Z, extent=(0, 1, 0, 1), origin='lower')

plt.colorbar()
plt.title("Result of XOR")

plt.show()



