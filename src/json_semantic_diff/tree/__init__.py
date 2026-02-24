"""Tree subpackage for JSON-to-tree conversion primitives.

Re-exports the public API for the tree module:
- TreeNode: dataclass representing a node in the JSON tree
- NodeType: StrEnum of the five node kinds (OBJECT, KEY, ARRAY, ELEMENT, SCALAR)
- KeyNormalizer: normalizes JSON keys across camelCase, PascalCase, snake_case, kebab-case
- TreeBuilder: converts any valid JSON value into a typed TreeNode tree
"""

from json_semantic_diff.tree.builder import TreeBuilder
from json_semantic_diff.tree.nodes import NodeType, TreeNode
from json_semantic_diff.tree.normalizer import KeyNormalizer

__all__ = ["KeyNormalizer", "NodeType", "TreeBuilder", "TreeNode"]
