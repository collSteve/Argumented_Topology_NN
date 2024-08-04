from typing import List
import numpy as np

import uuid

from activation_funcs import activation_funcs 
from loss_funcs import loss_funcs

'''
In this implementation, for backpropagation, we need to handle weights and inpupts gradients in edges,
And biases gradients in nodes.
We precompute a'(z), d z/d b and store them in the node's Node_Backprop_Values.
We also precompute d z/d w_j, d z / d I_j in the edge's Edge_Backprop_Values.
'''

class Node_Result:
    def __init__(self):
        self.value = 0
        self.valid = False
        self.evl_num = 0

class Node_Backprop_Values:
    def __init__(self):
        # for backpropagation
        self.da_dz = 0
        self.dz_db = 0

        # results
        self.dL_dI0 = 0

        self.evl_num = 0
    
    

class Edge_Backprop_Values:
    def __init__(self):
        # for backpropagation
        self.dz_dw = 0
        self.dz_dI = np.NaN

        # results
        self.dL_dw = 0
        self.dL_dI = 0

        # keep track of evl_num (back_propagation has finished for this evl_num)
        self.evl_num = 0

class Node:
    def __init__(self, id: str, activation: str = "sigmoid"):
        self.bias: float = 0
        self.aggregation_func = lambda x: sum(x)        # TODO: change hard code
        self.activation_func = activation_funcs[activation]["func"]
        self.diff_activation_func = activation_funcs[activation]["diff"]

        self.input_edges: List[Edge] = []
        self.output_edges: List[Edge] = []

        self.id: str = id

        self.calculated_result: Node_Result = None

        self.node_backprop_precomp: Node_Backprop_Values = None
    
    def get_result(self)-> Node_Result:
        return self.calculated_result

    def cal_result(self, evl_num)-> Node_Result:
        result = Node_Result()
        if len(self.input_edges) == 0:
            return result

        inputs_results: List[float] = []
        for edge in self.input_edges:
            if (edge.input_node.calculated_result == None) or (not edge.input_node.calculated_result.valid) or (edge.input_node.calculated_result.evl_num != evl_num):
                # should not happen that the node is evaluated after the current node
                # change when we have recurrent network
                # if (edge.input_node.calculated_result != None):
                #     print(f"edge.input_node.calculated_result.evl_num = {edge.input_node.calculated_result.evl_num}, evl_num = {evl_num}")
                #     assert(edge.input_node.calculated_result.evl_num < evl_num) 

                edge.input_node.calculated_result = edge.input_node.cal_result(evl_num)

            inputs_results.append(edge.input_node.calculated_result.value * edge.weight)

            # precompute edge gradients for backpropagation
            edge.edge_backprop_precomp = Edge_Backprop_Values()

            edge.edge_backprop_precomp.evl_num = None
            edge.edge_backprop_precomp.dz_dw = edge.input_node.calculated_result.value  # d z / d w_j = I_j
            edge.edge_backprop_precomp.dz_dI = edge.weight  # d z / d I_j = w_j

        z = self.aggregation_func(inputs_results) + self.bias
        result.value = self.activation_func(z)
        result.valid = True

        self.calculated_result = result

        # precompute node gradients for backpropagation
        self.node_backprop_precomp = Node_Backprop_Values()
        self.node_backprop_precomp.evl_num = None
        self.node_backprop_precomp.da_dz = self.diff_activation_func(z)
        self.node_backprop_precomp.dz_db = 1
        return result
    
    def compute_dL_dI0(self, learning_rate: float, evl_num: int):
        dL_dI0_s = []
        for edge in self.output_edges:
            if (edge.edge_backprop_precomp.evl_num != evl_num):
                edge.backpropagate(learning_rate, evl_num)
            dL_dI0_s.append(edge.edge_backprop_precomp.dL_dI)

        return sum(dL_dI0_s)
    
    def backpropagate(self, learning_rate: float, evl_num: int):
        if self.node_backprop_precomp.evl_num == evl_num:
            return
        
        # backpropagate output edges and compute dL_dI0
        self.node_backprop_precomp.dL_dI0 = self.compute_dL_dI0(learning_rate, evl_num)

        # update bias
        dL_db = self.node_backprop_precomp.dL_dI0 * self.node_backprop_precomp.da_dz * self.node_backprop_precomp.dz_db
        self.bias -= dL_db * learning_rate

        # update evl_num
        self.node_backprop_precomp.evl_num = evl_num
        
    
class Input_Node(Node):
    def __init__(self, id: str):
        super().__init__(id)
        self.activation_func = lambda x: x

        value = 0
    
    def cal_result(self, evl_num) -> Node_Result:
        self.calculated_result = Node_Result()
        self.calculated_result.value = self.value
        self.calculated_result.valid = True
        self.calculated_result.evl_num = evl_num
        return self.calculated_result

class Output_Node(Node):
    def __init__(self, id: str):
        super().__init__(id)

        self.dL_dI0 = 0

    def set_dL_dI0(self, dL_dI0: float):
        self.dL_dI0 = dL_dI0

    def backpropagate(self, learning_rate: float, evl_num: int):
        if self.node_backprop_precomp.evl_num == evl_num:
            return
        
        self.node_backprop_precomp.dL_dI0 = self.dL_dI0

        # update bias
        dL_db = self.node_backprop_precomp.dL_dI0 * self.node_backprop_precomp.da_dz * self.node_backprop_precomp.dz_db
        self.bias -= dL_db * learning_rate

        # update evl_num
        self.node_backprop_precomp.evl_num = evl_num


class Edge:
    def __init__(self, id: str):
        self.weight: float = 0
        self.input_node: Node = None
        self.output_node: Node = None

        self.id: str = id

        self.edge_backprop_precomp: Edge_Backprop_Values = None

    def backpropagate(self, learning_rate: float, evl_num: int):
        if self.edge_backprop_precomp.evl_num == evl_num:
            return
        
        if (self.output_node.node_backprop_precomp.evl_num != evl_num):
            self.output_node.backpropagate(learning_rate, evl_num)

        assert(self.output_node.node_backprop_precomp.evl_num == evl_num)

        # update weight
        dL_dw = self.output_node.node_backprop_precomp.dL_dI0 * self.output_node.node_backprop_precomp.da_dz * self.edge_backprop_precomp.dz_dw
        dL_dI = self.output_node.node_backprop_precomp.dL_dI0 * self.output_node.node_backprop_precomp.da_dz * self.edge_backprop_precomp.dz_dI
        
        self.weight -= dL_dw * learning_rate

        # update edge_backprop_precomp
        self.edge_backprop_precomp.dL_dw = dL_dw
        self.edge_backprop_precomp.dL_dI = dL_dI

        # update evl_num
        self.edge_backprop_precomp.evl_num = evl_num



class Graph:
    def __init__(self, loss_func: str = "mse"):
        self.nodes: List[Node] = []
        self.edges: List[Edge] = []

        self.input_nodes: List[Input_Node] = []
        self.output_nodes: List[Output_Node] = []

        self.loss_func = loss_funcs[loss_func]["func"]
        self.diff_loss_func = loss_funcs[loss_func]["diff"]

    def add_node(self, node):
        self.nodes.append(node)

    def add_edge(self, edge):
        self.edges.append(edge)
        edge.input_node.output_edges.append(edge)
        edge.output_node.input_edges.append(edge)

    def get_node(self, id: str):
        for node in self.nodes:
            if node.id == id:
                return node
        return None
    
    def get_edge(self, id: str):
        for edge in self.edges:
            if edge.id == id:
                return edge
        return None

    def connect(self, input_node_id: str, output_node_id: str, weight: float):
        input_node = self.get_node(input_node_id)
        output_node = self.get_node(output_node_id)

        if (input_node is None) or (output_node is None):
            raise Exception("Node not found")

        edge = Edge(str(uuid.uuid4()))
        edge.input_node = input_node
        edge.output_node = output_node
        edge.weight = weight
        self.add_edge(edge)
    
    def forward(self , inputs: List[float], evl_num: int):
        if len(inputs) != len(self.input_nodes):
            raise Exception("Input size mismatch")
        
        for i in range(len(inputs)):
            self.input_nodes[i].value = inputs[i]

        # next_nodes: List[Node] = [i for i in self.input_nodes]

        for node in self.output_nodes:
            node.calculated_result = node.cal_result(evl_num)

    def backpropagate(self, inputs: List[float], target_outputs: List[float], learning_rate: float, evl_num: int):
        self.forward(inputs, evl_num)

        for output_node, target_output in zip(self.output_nodes, target_outputs):
            output_node.set_dL_dI0(self.diff_loss_func(target_output, output_node.calculated_result.value))

        # backpropagate on first column of input edges
        for input_node in self.input_nodes:
            for edge in input_node.output_edges:
                edge.backpropagate(learning_rate, evl_num)

    
    def train(self, training_data: List[List[List[float]]]):
        evl_num = 1
        for inputs, target_outputs in training_data:
            self.backpropagate(inputs, target_outputs, 0.1, evl_num)
            evl_num += 1

    def get_output(self):
        return [node.calculated_result.value for node in self.output_nodes]

    def print(self):
        for node in self.nodes:
            print("Node: ", node.activation)
            for edge in node.input_edges:
                print("Edge: ", edge.weight)

# for i in range(10):
#     print(uuid.uuid4())

# graph = Graph()
# graph.add_node(Input_Node("1"))
# graph.add_node(Input_Node("2"))
# graph.add_node(Node("3"))

# graph.connect("1", "3", 2)
# graph.connect("2", "3", 2)

# graph.get_node("3").aggregation_func = lambda x: sum(x)
# graph.get_node("3").activation_func = lambda x: x
# graph.get_node("3").bias = 0.5

# graph.input_nodes = [graph.get_node("1"), graph.get_node("2")]

# graph.output_nodes = [graph.get_node("3")]

# graph.forward([1, 2], 0)

# print(graph.get_output())
