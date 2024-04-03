from typing import List
import numpy as np

import uuid

class Node_Result:
    def __init__(self):
        self.value = 0
        self.valid = False
        self.evl_num = 0

class Node:
    def __init__(self, id: str):
        self.bias: float = 0
        self.aggregation_func = None
        self.activation_func = None

        self.input_edges: List[Edge] = []
        self.output_edges: List[Edge] = []

        self.id: str = id

        self.calculated_result: Node_Result = None
    
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
                if (edge.input_node.calculated_result != None):
                    assert(edge.input_node.calculated_result.evl_num < evl_num) 

                edge.input_node.calculated_result = edge.input_node.cal_result(evl_num)

            inputs_results.append(edge.input_node.calculated_result.value * edge.weight)

        result.value = self.activation_func(self.aggregation_func(inputs_results) + self.bias)
        result.valid = True
        return result
    
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


class Edge:
    def __init__(self, id: str):
        self.weight: float = 0
        self.input_node: Node = None
        self.output_node: Node = None

        self.id: str = id

class Graph:
    def __init__(self):
        self.nodes: List[Node] = []
        self.edges: List[Edge] = []

        self.input_nodes: List[Input_Node] = []
        self.output_nodes: List[Node] = []

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

    

    def get_output(self):
        return [node.calculated_result.value for node in self.output_nodes]

    def print(self):
        for node in self.nodes:
            print("Node: ", node.activation)
            for edge in node.input_edges:
                print("Edge: ", edge.weight)

# for i in range(10):
#     print(uuid.uuid4())

graph = Graph()
graph.add_node(Input_Node("1"))
graph.add_node(Input_Node("2"))
graph.add_node(Node("3"))

graph.connect("1", "3", 2)
graph.connect("2", "3", 2)

graph.get_node("3").aggregation_func = lambda x: sum(x)
graph.get_node("3").activation_func = lambda x: x
graph.get_node("3").bias = 0.5

graph.input_nodes = [graph.get_node("1"), graph.get_node("2")]

graph.output_nodes = [graph.get_node("3")]

graph.forward([1, 2], 0)

print(graph.get_output())
