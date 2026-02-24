"""Tests for TreeNode dataclass and NodeType StrEnum.

Verifies:
- NodeType has exactly 5 members with lowercase string values (StrEnum property)
- TreeNode constructs correctly with all fields
- Each TreeNode instance gets an independent children list (no shared mutable default)
- Default values for optional fields are correct
"""

import pytest

from semantic_diff.tree.nodes import NodeType, TreeNode


class TestNodeType:
    """Tests for the NodeType StrEnum."""

    def test_has_exactly_five_members(self) -> None:
        """NodeType must have exactly OBJECT, KEY, ARRAY, ELEMENT, SCALAR."""
        assert len(NodeType) == 5

    def test_expected_members_exist(self) -> None:
        """All five members must be accessible by name."""
        assert NodeType.OBJECT is not None
        assert NodeType.KEY is not None
        assert NodeType.ARRAY is not None
        assert NodeType.ELEMENT is not None
        assert NodeType.SCALAR is not None

    def test_members_are_str_instances(self) -> None:
        """StrEnum members must also be str instances (StrEnum property)."""
        for member in NodeType:
            assert isinstance(member, str), f"{member!r} is not a str instance"

    def test_values_are_lowercased(self) -> None:
        """auto() on StrEnum yields the lowercased member name (Python 3.11+)."""
        assert NodeType.OBJECT == "object"
        assert NodeType.KEY == "key"
        assert NodeType.ARRAY == "array"
        assert NodeType.ELEMENT == "element"
        assert NodeType.SCALAR == "scalar"

    def test_members_can_be_compared_to_strings(self) -> None:
        """StrEnum allows direct comparison with string literals."""
        assert NodeType.OBJECT == "object"
        assert NodeType.OBJECT == "object"


class TestTreeNode:
    """Tests for the TreeNode dataclass."""

    def test_construction_with_required_fields(self) -> None:
        """TreeNode can be constructed with only required fields."""
        node = TreeNode(node_type=NodeType.OBJECT, label="", path="/")
        assert node.node_type == NodeType.OBJECT
        assert node.label == ""
        assert node.path == "/"

    def test_construction_with_all_fields(self) -> None:
        """TreeNode accepts all six fields and stores them correctly."""
        child = TreeNode(node_type=NodeType.SCALAR, label="42", path="/age")
        node = TreeNode(
            node_type=NodeType.KEY,
            label="age",
            path="/age",
            raw_label="age",
            value=42,
            children=[child],
        )
        assert node.node_type == NodeType.KEY
        assert node.label == "age"
        assert node.path == "/age"
        assert node.raw_label == "age"
        assert node.value == 42
        assert len(node.children) == 1
        assert node.children[0] is child

    def test_default_raw_label_is_empty_string(self) -> None:
        """raw_label defaults to empty string."""
        node = TreeNode(node_type=NodeType.SCALAR, label="hello", path="/greeting")
        assert node.raw_label == ""

    def test_default_value_is_none(self) -> None:
        """value defaults to None."""
        node = TreeNode(node_type=NodeType.OBJECT, label="", path="/")
        assert node.value is None

    def test_default_children_is_empty_list(self) -> None:
        """children defaults to an empty list."""
        node = TreeNode(node_type=NodeType.OBJECT, label="", path="/")
        assert node.children == []
        assert isinstance(node.children, list)

    def test_children_lists_are_independent_per_instance(self) -> None:
        """Two TreeNode instances must NOT share the same children list object.

        This ensures the field(default_factory=list) pattern is used correctly
        rather than a shared mutable default (which would cause class-level mutation).
        """
        node_a = TreeNode(node_type=NodeType.OBJECT, label="", path="/a")
        node_b = TreeNode(node_type=NodeType.OBJECT, label="", path="/b")

        child = TreeNode(node_type=NodeType.SCALAR, label="x", path="/a/x")
        node_a.children.append(child)

        assert len(node_a.children) == 1
        assert len(node_b.children) == 0, (
            "Mutating node_a.children must not affect node_b.children"
        )
        assert node_a.children is not node_b.children

    def test_path_attribute_is_preserved(self) -> None:
        """Path string is stored and returned exactly as provided."""
        path = "/some/deep/nested/path"
        node = TreeNode(node_type=NodeType.KEY, label="path", path=path)
        assert node.path == path

    def test_scalar_node_stores_typed_value(self) -> None:
        """SCALAR nodes can store any Python value type."""
        int_node = TreeNode(node_type=NodeType.SCALAR, label="42", path="/n", value=42)
        bool_node = TreeNode(
            node_type=NodeType.SCALAR, label="True", path="/b", value=True
        )
        none_node = TreeNode(
            node_type=NodeType.SCALAR, label="null", path="/null", value=None
        )

        assert int_node.value == 42
        assert isinstance(int_node.value, int)
        assert bool_node.value is True
        assert none_node.value is None

    def test_node_type_stored_as_nodetype_instance(self) -> None:
        """node_type field stores a NodeType (StrEnum) value."""
        node = TreeNode(node_type=NodeType.ARRAY, label="", path="/items")
        assert isinstance(node.node_type, NodeType)
        assert node.node_type == "array"

    def test_uses_slots(self) -> None:
        """TreeNode should use __slots__ for memory efficiency."""
        assert hasattr(TreeNode, "__slots__")

    def test_cannot_add_arbitrary_attributes(self) -> None:
        """With slots=True, setting undefined attributes raises AttributeError."""
        node = TreeNode(node_type=NodeType.SCALAR, label="x", path="/x")
        with pytest.raises(AttributeError):
            node.undefined_attribute = "should fail"  # type: ignore[attr-defined]
