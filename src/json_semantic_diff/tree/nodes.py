"""TreeNode dataclass and NodeType StrEnum for JSON-to-tree representation.

Provides the foundational data types used by TreeBuilder (Plan 02) to convert
JSON documents into typed tree structures for semantic comparison.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum, auto
from typing import Any


class NodeType(StrEnum):
    """Enumeration of the five structural node types in a JSON tree.

    StrEnum values are the lowercased member names (Python 3.11+):
    - OBJECT   -> "object"  : JSON object {}
    - KEY      -> "key"     : A key within a JSON object
    - ARRAY    -> "array"   : JSON array []
    - ELEMENT  -> "element" : An element within a JSON array
    - SCALAR   -> "scalar"  : A leaf value (string, number, bool, null)
    """

    OBJECT = auto()
    KEY = auto()
    ARRAY = auto()
    ELEMENT = auto()
    SCALAR = auto()


@dataclass(slots=True)
class TreeNode:
    """A node in the JSON tree representation.

    Attributes:
        node_type:  Which kind of node this is (see NodeType).
        label:      Normalized key for KEY nodes; str(value) for SCALAR nodes;
                    empty string for structural nodes (OBJECT, ARRAY, ELEMENT).
        path:       JSON Pointer path (RFC 6901), e.g. "/user/name".
        raw_label:  Original (un-normalized) key for KEY nodes; empty for all others.
        value:      Original typed Python value for SCALAR nodes; None for structural.
        children:   Child nodes. Must use field(default_factory=list) to ensure
                    each instance gets its own independent list.
    """

    node_type: NodeType
    label: str
    path: str
    raw_label: str = ""
    value: Any = None
    children: list[TreeNode] = field(default_factory=list)
