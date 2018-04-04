from unittest import TestCase
from anytree import RenderTree, AnyNode, PreOrderIter
import arff
import pandas as pd
from lab2.ej5.src.id3 import id3
from lab2.ej5.src.strategy.entropy import select_attribute


def load_file(path):
    data = arff.load(open(path, 'r'))
    attributes = data["attributes"]
    columns = [x[0] for x in attributes]
    return pd.DataFrame(data=data['data'], columns=columns), attributes


def load_expected_tree_structure():
    root = AnyNode(attribute="Outlook")
    s0 = AnyNode(parent=root, root_value="Sunny", attribute="Humidity")
    s1 = AnyNode(parent=root, root_value="Overcast", value="YES")
    s2 = AnyNode(parent=root, root_value="Rain", attribute="Wind")

    s0_1 = AnyNode(parent=s0, root_value="High", value="NO")
    s0_2 = AnyNode(parent=s0, root_value="Normal", value="YES")

    s2_2 = AnyNode(parent=s2, root_value="Weak", value="YES")
    s2_1 = AnyNode(parent=s2, root_value="Strong", value="NO")

    return root


def equal_tree_structure(built_tree, expected_tree):
    print("testing if equal trees")
    print(RenderTree(built_tree))
    print(RenderTree(expected_tree))

    comparable_built_tree = list(map(get_comparable_node_info, PreOrderIter(built_tree)))
    comparable_expected_tree = list(map(get_comparable_node_info, PreOrderIter(expected_tree)))

    return comparable_built_tree == comparable_expected_tree


def get_comparable_node_info(node):
    if node.is_root:
        candidate = (node.__getattribute__("attribute"), None, None)
    elif node.is_leaf:
        candidate = (None, node.__getattribute__("root_value"), node.__getattribute__("value"))
    else:
        candidate = (node.__getattribute__("attribute"), node.__getattribute__("root_value"))

    return candidate


class TestAnyTree(TestCase):
    def test_id3_structure(self):
        (df, attributes) = load_file('../../datasets/tom_mitchell_example.arff')
        expected_tree_structure = load_expected_tree_structure()
        built_tree = id3(examples=df, select_attribute=select_attribute, target_attribute='PlayTennis', attributes=attributes)

        assert equal_tree_structure(built_tree, expected_tree_structure)