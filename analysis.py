"""
EDA Analysis Engine
Provides: path queries, depth analysis, cone analysis, fanout, clock domains
"""
from __future__ import annotations
import re
from typing import Dict, List, Optional, Set, Tuple, Any
import networkx as nx

from netlist import Netlist, Gate, PRIMITIVE_GATES


def resolve_net(nl: Netlist, name: str) -> Optional[str]:
    """Resolve a user-specified name to an actual net or gate output name."""
    # Direct match in inputs or outputs
    if name in nl.inputs or name in nl.outputs:
        return name
    # Match as a gate output
    for g in nl.gates.values():
        if g.output == name:
            return name
    # Match by gate name (return its output)
    if name in nl.gates:
        return nl.gates[name].output
    # Fuzzy: case-insensitive
    name_lower = name.lower()
    for inp in nl.inputs:
        if inp.lower() == name_lower:
            return inp
    for out in nl.outputs:
        if out.lower() == name_lower:
            return out
    return None


def find_cone(nl: Netlist, net: str) -> Set[str]:
    """Return all gate names in the transitive fanin cone of 'net'."""
    drv = nl.driver_map()
    cone: Set[str] = set()
    stack = [net]
    visited: Set[str] = set()
    while stack:
        n = stack.pop()
        if n in visited:
            continue
        visited.add(n)
        gate_name = drv.get(n)
        if gate_name and gate_name not in ("PI", "CONST"):
            g = nl.gates.get(gate_name)
            if g:
                cone.add(gate_name)
                for inp in g.inputs:
                    if inp not in visited:
                        stack.append(inp)
    return cone


def compute_logic_depth(nl: Netlist, from_net: Optional[str] = None, to_net: Optional[str] = None) -> Tuple[int, List[str]]:
    """
    Compute the maximum combinational logic depth.
    If from_net and to_net are both given, computes depth on that specific path.
    Returns (depth, example_path_as_list_of_gate_names).
    DFFs are treated as pseudo-primary inputs (cut points).
    """
    drv = nl.driver_map()

    # Build DAG with edge weights = 1 per gate
    # nodes: net names (PI and DFF outputs) and gate names
    # We'll use a simple BFS/DP approach

    # depth[net] = max combinational depth to reach that net
    depth: Dict[str, int] = {}
    pred: Dict[str, Optional[str]] = {}  # for path reconstruction: net -> prev_gate_name

    # Initialize PIs and DFF outputs to depth 0
    for inp in nl.inputs:
        depth[inp] = 0
        pred[inp] = None
    for g in nl.gates.values():
        if g.is_dff():
            depth[g.output] = 0
            pred[g.output] = None
    # Constants
    depth["1'b0"] = 0
    depth["1'b1"] = 0

    # Topological order over combinational gates
    # Build dependency graph among combinational gates
    comb_gates = {n: g for n, g in nl.gates.items() if not g.is_dff()}
    # topo sort
    G = nx.DiGraph()
    for gname, g in comb_gates.items():
        G.add_node(gname)
        for inp in g.inputs:
            src_gate = drv.get(inp)
            if src_gate and src_gate not in ("PI", "CONST") and src_gate in comb_gates:
                G.add_edge(src_gate, gname)

    try:
        topo = list(nx.topological_sort(G))
    except nx.NetworkXUnfeasible:
        return -1, []

    gate_depth: Dict[str, int] = {}
    gate_pred: Dict[str, Optional[str]] = {}  # gate -> predecessor gate for longest path

    for gname in topo:
        g = comb_gates[gname]
        max_d = 0
        best_pred = None
        for inp in g.inputs:
            if inp in depth:
                d = depth[inp]
            else:
                src = drv.get(inp)
                d = gate_depth.get(src, 0) if src else 0
            if d > max_d:
                max_d = d
                best_pred = drv.get(inp)
        gate_depth[gname] = max_d + 1
        gate_pred[gname] = best_pred
        depth[g.output] = max_d + 1

    if from_net and to_net:
        # specific path
        to_gate = drv.get(to_net)
        if to_gate is None or to_gate in ("PI", "CONST"):
            return 0, []
        d = gate_depth.get(to_gate, 0)
        path = _reconstruct_path(gate_pred, to_gate, from_net, nl)
        return d, path

    # Find maximum across requested outputs (or all outputs)
    target_nets = [to_net] if to_net else nl.outputs
    best_d = 0
    best_path: List[str] = []
    for net in target_nets:
        gate_name = drv.get(net)
        if gate_name and gate_name in gate_depth:
            d = gate_depth[gate_name]
            if d > best_d:
                best_d = d
                best_path = _reconstruct_path(gate_pred, gate_name, from_net, nl)

    return best_d, best_path


def _reconstruct_path(gate_pred: Dict, end_gate: str, start_net: Optional[str], nl: Netlist) -> List[str]:
    """Walk back through gate_pred to reconstruct the longest path."""
    path = []
    cur = end_gate
    visited = set()
    while cur and cur not in visited:
        visited.add(cur)
        path.append(cur)
        prev = gate_pred.get(cur)
        if prev is None:
            break
        # prev is a gate name or net name
        if prev in nl.gates:
            cur = prev
        else:
            break
    path.reverse()
    return path


def path_exists(nl: Netlist, from_net: str, to_net: str, exclude: Optional[str] = None) -> Tuple[bool, List[str]]:
    """
    Check if a combinational path exists from from_net to to_net.
    Optionally exclude a specific net from the path.
    Returns (exists, example_path_as_gate_names).
    """
    drv = nl.driver_map()
    fo = nl.fanout_map()

    # BFS forward from from_net
    # Nodes are nets; move through gates
    stack = [(from_net, [from_net])]
    visited: Set[str] = set()

    while stack:
        net, path = stack.pop()
        if net in visited:
            continue
        visited.add(net)

        if net == to_net and net != from_net:
            return True, path

        if net == exclude:
            continue

        consumers = fo.get(net, [])
        for gname in consumers:
            g = nl.gates[gname]
            if g.is_dff():
                continue  # don't cross DFF
            out_net = g.output
            if out_net not in visited:
                stack.append((out_net, path + [gname, out_net]))

    return False, []


def every_path_passes_through(nl: Netlist, from_net: str, to_net: str, through_net: str) -> Tuple[bool, List[str]]:
    """
    Returns True if EVERY combinational path from from_net to to_net passes through through_net.
    Also returns a counterexample path if False.
    """
    # Check if a path exists that bypasses through_net
    exists_without, counterex = path_exists(nl, from_net, to_net, exclude=through_net)
    if exists_without:
        return False, counterex
    # Also check that at least one path exists (going through through_net)
    exists_through, _ = path_exists(nl, from_net, through_net)
    if not exists_through:
        return False, []
    return True, []


def find_max_fanout(nl: Netlist) -> Tuple[str, int]:
    """Return (net_name, fanout_count) for the net with highest fanout."""
    fo = nl.fanout_map()
    best = ("", 0)
    for net, consumers in fo.items():
        if len(consumers) > best[1]:
            best = (net, len(consumers))
    return best


def get_clock_domains(nl: Netlist) -> Dict[str, List[str]]:
    """Group DFFs by their clock signal."""
    domains: Dict[str, List[str]] = {}
    for g in nl.gates.values():
        if g.is_dff():
            clk = g.inputs[0] if g.inputs else "unknown"
            domains.setdefault(clk, []).append(g.name)
    return domains


def same_clock_domain(nl: Netlist, ff1: str, ff2: str) -> Tuple[bool, str, str]:
    """Check if two DFFs share the same clock."""
    g1 = nl.gates.get(ff1)
    g2 = nl.gates.get(ff2)
    if not g1 or not g2:
        return False, "unknown", "unknown"
    clk1 = g1.inputs[0] if g1.inputs else "unknown"
    clk2 = g2.inputs[0] if g2.inputs else "unknown"
    return clk1 == clk2, clk1, clk2


def cone_gate_count(nl: Netlist, net: str) -> int:
    return len(find_cone(nl, net))


def outputs_with_large_cones(nl: Netlist, threshold: int) -> List[Tuple[str, int]]:
    result = []
    for out in nl.outputs:
        cnt = cone_gate_count(nl, out)
        if cnt > threshold:
            result.append((out, cnt))
    return sorted(result, key=lambda x: -x[1])


def dangling_gates(nl: Netlist) -> Set[str]:
    """Find gates whose output does not affect any primary output."""
    # Compute all gates in any output's cone
    useful: Set[str] = set()
    for out in nl.outputs:
        useful.update(find_cone(nl, out))
    return set(nl.gates.keys()) - useful


def simple_equivalence_check(nl: Netlist, net_a: str, net_b: str) -> Optional[bool]:
    """
    Heuristic structural equivalence: checks if two nets are driven by
    functionally identical sub-cones (same gate tree structure).
    Returns None if uncertain.
    """
    def cone_signature(net: str) -> str:
        drv = nl.driver_map()
        cache: Dict[str, str] = {}

        def sig(n: str) -> str:
            if n in cache:
                return cache[n]
            gate_name = drv.get(n)
            if not gate_name or gate_name in ("PI", "CONST"):
                cache[n] = n
                return n
            g = nl.gates.get(gate_name)
            if not g:
                cache[n] = n
                return n
            child_sigs = sorted(sig(inp) for inp in g.inputs)
            s = f"{g.gate_type}({','.join(child_sigs)})"
            cache[n] = s
            return s

        return sig(net)

    s1 = cone_signature(net_a)
    s2 = cone_signature(net_b)
    if s1 == s2:
        return True
    return None  # inconclusive without formal SAT


def find_gates_matching(nl: Netlist, gate_type: str, name_pattern: Optional[str] = None) -> List[Gate]:
    """Find gates of given type, optionally filtered by name pattern."""
    result = []
    for g in nl.gates.values():
        if g.gate_type != gate_type:
            continue
        if name_pattern and name_pattern not in g.name:
            continue
        result.append(g)
    return result


def verify_output_condition(nl: Netlist, output_net: str, condition: str) -> Dict[str, Any]:
    """
    Simple structural check for conditions like 'output is 1 only when req=1 and busy=0'.
    Returns a dict with 'result' and 'explanation'.
    """
    # This is a heuristic / structural analysis placeholder
    # Full SAT would require a SAT solver integration
    cone = find_cone(nl, output_net)
    return {
        "result": "unknown",
        "explanation": f"Formal verification requires SAT/BDD engine. "
                       f"Logic cone of '{output_net}' contains {len(cone)} gates. "
                       f"Structural analysis cannot conclusively verify the condition: {condition}"
    }
