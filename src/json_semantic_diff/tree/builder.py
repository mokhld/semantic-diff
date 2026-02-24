"""TreeBuilder: converts any valid JSON value into a typed TreeNode tree.

Uses recursive dispatch to convert JSON dicts, lists, and scalar values into
a tree of TreeNode objects. Object keys are normalized via KeyNormalizer.
Array elements are wrapped in ELEMENT nodes with index-based labels.
Scalar values preserve their original Python type in the TreeNode.value field.

JSON Pointer paths (RFC 6901) are built during traversal:
- Root is "" (empty string)
- Each level appends "/{key_or_index}"
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from json_semantic_diff.tree.nodes import NodeType, TreeNode
from json_semantic_diff.tree.normalizer import KeyNormalizer

# Module-level normalizer (stateless, safe to share across all TreeBuilder instances)
_normalizer = KeyNormalizer()

# Type alias for valid JSON values
JsonValue = dict[str, Any] | list[Any] | str | int | float | bool | None


@dataclass
class TreeBuilder:
    """Converts any valid JSON value into a typed TreeNode tree.

    Uses recursive dispatch to handle all JSON types. The dispatch order is
    critical: bool MUST be checked before int because bool is a subclass of int
    in Python (isinstance(True, int) is True).

    JSON Pointer paths (RFC 6901):
        Root node has path="" (empty string).
        Each nested level appends "/{key_or_index}" to the parent path.

    Type preservation:
        SCALAR nodes store the original Python value in the `value` field.
        "5" (str) and 5 (int) produce distinct nodes with the same label "5"
        but different type(node.value). The STED algorithm uses this to detect
        type mismatches between compared documents.

    Example::
        builder = TreeBuilder()
        tree = builder.build({"user_name": "John"})
        # tree: OBJECT -> KEY("user name", raw="user_name") -> SCALAR("John")
    """

    def build(self, value: JsonValue, path: str = "") -> TreeNode:
        """Convert a JSON value to a TreeNode tree.

        Args:
            value: Any valid JSON value (dict, list, str, int, float, bool, None).
            path:  JSON Pointer path to this node. Defaults to "" (root).

        Returns:
            A TreeNode tree rooted at the appropriate node type.

        Raises:
            TypeError: If value is not a valid JSON type.
        """
        # CRITICAL: bool MUST be checked before int — bool subclasses int in Python
        if isinstance(value, bool):
            label = "true" if value else "false"
            return TreeNode(
                node_type=NodeType.SCALAR, label=label, path=path, value=value
            )

        if isinstance(value, dict):
            return self._build_object(value, path)

        if isinstance(value, list):
            return self._build_array(value, path)

        if isinstance(value, (str, int, float)):
            return TreeNode(
                node_type=NodeType.SCALAR, label=str(value), path=path, value=value
            )

        if value is None:
            return TreeNode(
                node_type=NodeType.SCALAR, label="null", path=path, value=None
            )

        raise TypeError(f"Unsupported JSON value type: {type(value)!r}")

    def _build_object(self, obj: dict[str, Any], path: str) -> TreeNode:
        """Build an OBJECT node with KEY->value child pairs.

        Args:
            obj:  The JSON object (Python dict).
            path: JSON Pointer path to this object node.

        Returns:
            An OBJECT TreeNode whose children are KEY nodes.
        """
        object_node = TreeNode(node_type=NodeType.OBJECT, label="", path=path)

        for key, val in obj.items():
            key_path = f"{path}/{key}"
            key_node = TreeNode(
                node_type=NodeType.KEY,
                label=_normalizer.normalize(key),
                path=key_path,
                raw_label=key,
            )
            child = self.build(val, path=key_path)
            key_node.children.append(child)
            object_node.children.append(key_node)

        return object_node

    def _build_array(self, arr: list[Any], path: str) -> TreeNode:
        """Build an ARRAY node with ELEMENT->value child pairs.

        Args:
            arr:  The JSON array (Python list).
            path: JSON Pointer path to this array node.

        Returns:
            An ARRAY TreeNode whose children are ELEMENT nodes indexed by position.
        """
        array_node = TreeNode(node_type=NodeType.ARRAY, label="", path=path)

        for idx, item in enumerate(arr):
            elem_path = f"{path}/{idx}"
            elem_node = TreeNode(
                node_type=NodeType.ELEMENT,
                label=str(idx),
                path=elem_path,
            )
            child = self.build(item, path=elem_path)
            elem_node.children.append(child)
            array_node.children.append(elem_node)

        return array_node
