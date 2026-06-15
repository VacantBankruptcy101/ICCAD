"""
EDA Transformation Engine
Supports: buffer insertion, gate replacement, cone optimization,
          dead gate removal, fanout balancing, logic resynthesis
"""
from __future__ import annotations
import re
import copy
from typing import Dict, List, Optional, Set, Tuple, Any

from netlist import Netlist, Gate, PRIMITIVE_GATES, ONE_INPUT_GATES, TWO_INPUT_GATES
from analysis import find_cone, compute_logic_depth, dangling_gates, find_gates_matching


def _fresh_name(nl: Netlist, prefix: str) -> str:
    """Generate a unique gate/wire name."""
    existing = set(nl.gates.keys()) | set(nl.wires) | set(nl.inputs) | set(nl.outputs)
    i = 0
    while True:
        name = f"{prefix}{i}"
        if name not in existing:
            return name
        i += 1


def insert_gate_before(nl: Netlist, target_gate_name: str, new_type: str,
                        extra_input: str) -> Tuple[bool, str]:
    """
    Replace gate 'target_gate_name' (must be buf or not) with 'new_type' gate
    that has the original input + extra_input, preserving output net.
    Returns (success, message).
    """
    g = nl.gates.get(target_gate_name)
    if not g:
        return False, f"Gate '{target_gate_name}' not found."

    orig_input = g.inputs[0] if g.inputs else None
    if orig_input is None:
        return False, f"Gate '{target_gate_name}' has no inputs."

    new_gate = Gate(
        name=target_gate_name,  # keep same instance name
        gate_type=new_type,
        output=g.output,
        inputs=[orig_input, extra_input]
    )
    nl.gates[target_gate_name] = new_gate
    return True, f"Replaced {g.gate_type} '{target_gate_name}' with {new_type}."


def replace_buffers_with_gate(nl: Netlist, name_pattern: str, new_type: str,
                               extra_input: str) -> Tuple[int, List[str]]:
    """
    Find all buf gates whose name contains name_pattern, replace each with new_type.
    Returns (count_replaced, list_of_instance_names).
    """
    targets = find_gates_matching(nl, 'buf', name_pattern)
    replaced = []
    for g in targets:
        ok, _ = insert_gate_before(nl, g.name, new_type, extra_input)
        if ok:
            replaced.append(g.name)
    return len(replaced), replaced


def remove_dangling_gates(nl: Netlist) -> Tuple[int, List[str]]:
    """Remove all gates not contributing to any primary output."""
    dangle = dangling_gates(nl)
    # Also remove wires driven by dangling gates
    removed_nets: Set[str] = set()
    for gname in dangle:
        g = nl.gates.pop(gname)
        removed_nets.add(g.output)
    # Clean up wires
    nl.wires = [w for w in nl.wires if w not in removed_nets]
    return len(dangle), list(dangle)


def replace_gate_type(nl: Netlist, old_type: str, new_types: List[str],
                       cone_output: Optional[str] = None) -> Tuple[int, List[str]]:
    """
    Replace all gates of old_type with equivalent logic using new_types gates.
    Supports: OR->NAND+NOT, AND->NAND+NOT, NOR->OR+NOT, etc.
    cone_output: if given, only within that output's cone.
    """
    scope: Optional[Set[str]] = None
    if cone_output:
        scope = find_cone(nl, cone_output)

    replaced = []
    # We may need to add new gates, so collect modifications
    to_add: Dict[str, Gate] = {}
    to_remove: Set[str] = set()

    for gname, g in list(nl.gates.items()):
        if g.gate_type != old_type:
            continue
        if scope is not None and gname not in scope:
            continue

        new_gates = _decompose_gate(nl, g, new_types, to_add)
        if new_gates:
            to_remove.add(gname)
            replaced.append(gname)
            for ng in new_gates:
                to_add[ng.name] = ng

    for gname in to_remove:
        del nl.gates[gname]
    nl.gates.update(to_add)
    return len(replaced), replaced


def _decompose_gate(nl: Netlist, g: Gate, allowed_types: List[str],
                     existing_new: Dict[str, Gate]) -> List[Gate]:
    """Decompose gate g into equivalent gates using only allowed_types."""
    # OR using NAND: a | b = nand(not a, not b)
    # OR using NOR+NOT: a | b = not(nor(a,b))
    # AND using NAND: a & b = not(nand(a,b))
    # NOT using NAND: not a = nand(a,a)
    # XOR using NAND: 4-NAND construction

    a, b = (g.inputs[0], g.inputs[1]) if len(g.inputs) >= 2 else (g.inputs[0], g.inputs[0])

    def fresh(prefix: str) -> str:
        existing = set(nl.gates.keys()) | set(nl.wires) | set(existing_new.keys())
        i = 0
        while True:
            nm = f"{prefix}_{i}"
            if nm not in existing:
                existing.add(nm)
                return nm
            i += 1

    def make(gtype: str, out: str, ins: List[str]) -> Gate:
        # add wire for intermediate
        if out not in nl.outputs and out not in nl.inputs:
            if out not in nl.wires:
                nl.wires.append(out)
        return Gate(name=fresh(f"U_{gtype}"), gate_type=gtype, output=out, inputs=ins)

    gtype = g.gate_type
    out = g.output
    new_gates: List[Gate] = []

    if 'nand' in allowed_types and 'not' in allowed_types:
        if gtype == 'or':
            # a|b = nand(not(a), not(b))
            na = fresh("n_nota"); nb = fresh("n_notb")
            nl.wires.extend([na, nb])
            ng1 = Gate(fresh(f"Unot"), 'not', na, [a])
            ng2 = Gate(fresh(f"Unot"), 'not', nb, [b])
            ng3 = Gate(fresh(f"Unand"), 'nand', out, [na, nb])
            return [ng1, ng2, ng3]
        elif gtype == 'and':
            tmp = fresh("n_nand")
            nl.wires.append(tmp)
            ng1 = Gate(fresh("Unand"), 'nand', tmp, [a, b])
            ng2 = Gate(fresh("Unot"), 'not', out, [tmp])
            return [ng1, ng2]
        elif gtype == 'nor':
            tmp = fresh("n_or")
            nl.wires.append(tmp)
            na = fresh("n_nota"); nb = fresh("n_notb")
            nl.wires.extend([na, nb])
            ng1 = Gate(fresh("Unot"), 'not', na, [a])
            ng2 = Gate(fresh("Unot"), 'not', nb, [b])
            ng3 = Gate(fresh("Unand"), 'nand', tmp, [na, nb])
            ng4 = Gate(fresh("Unot"), 'not', out, [tmp])
            return [ng1, ng2, ng3, ng4]

    if 'nor' in allowed_types and 'not' in allowed_types:
        if gtype == 'or':
            tmp = fresh("n_nor")
            nl.wires.append(tmp)
            ng1 = Gate(fresh("Unor"), 'nor', tmp, [a, b])
            ng2 = Gate(fresh("Unot"), 'not', out, [tmp])
            return [ng1, ng2]

    return []  # unsupported decomposition


def insert_buffer_for_fanout(nl: Netlist, net: str, max_fanout: int) -> Tuple[int, List[str]]:
    """
    Insert buffer tree on net so that no gate has fanout > max_fanout.
    Returns (buffers_inserted, list_of_new_gate_names).
    """
    fo = nl.fanout_map()
    consumers = fo.get(net, [])
    if len(consumers) <= max_fanout:
        return 0, []

    inserted: List[str] = []
    # Create buffer groups
    remaining = list(consumers)
    current_net = net

    while len(remaining) > max_fanout:
        # Create a buffer to drive first max_fanout consumers
        chunk = remaining[:max_fanout - 1]
        remaining = remaining[max_fanout - 1:]
        buf_out = _fresh_name(nl, f"_fo_buf_{net}_")
        buf_name = _fresh_name(nl, f"Ufo_buf_")
        nl.wires.append(buf_out)
        nl.gates[buf_name] = Gate(name=buf_name, gate_type='buf', output=buf_out, inputs=[current_net])
        inserted.append(buf_name)
        # Reconnect chunk consumers to buf_out
        for gname in chunk:
            g = nl.gates[gname]
            nl.gates[gname] = Gate(
                name=g.name, gate_type=g.gate_type, output=g.output,
                inputs=[buf_out if inp == current_net else inp for inp in g.inputs]
            )

    return len(inserted), inserted


def balance_depth(nl: Netlist, source_net: str, targets: List[str]) -> Tuple[int, List[str]]:
    """
    Add buffer stages so all paths from source_net to each target in targets
    have the same depth (minimal buffer insertion).
    """
    from analysis import compute_logic_depth
    depths = {}
    for t in targets:
        d, _ = compute_logic_depth(nl, source_net, t)
        depths[t] = d

    max_d = max(depths.values()) if depths else 0
    inserted: List[str] = []

    for t, d in depths.items():
        deficit = max_d - d
        if deficit <= 0:
            continue
        # Find the gate driving t and add buffers in a chain
        drv = nl.driver_map()
        cur_gate = drv.get(t)
        if not cur_gate or cur_gate in ("PI", "CONST"):
            continue
        g = nl.gates.get(cur_gate)
        if not g:
            continue
        # Insert 'deficit' buffers before the driving gate's input
        prev_net = g.inputs[0]
        for i in range(deficit):
            buf_out = _fresh_name(nl, f"_bal_")
            buf_name = _fresh_name(nl, f"Ubal_")
            nl.wires.append(buf_out)
            nl.gates[buf_name] = Gate(name=buf_name, gate_type='buf', output=buf_out, inputs=[prev_net])
            inserted.append(buf_name)
            prev_net = buf_out
        # Reconnect gate input
        new_inputs = [prev_net if inp == g.inputs[0] else inp for inp in g.inputs]
        nl.gates[cur_gate] = Gate(name=cur_gate, gate_type=g.gate_type, output=g.output, inputs=new_inputs)

    return len(inserted), inserted


def replace_inv_buf_pairs(nl: Netlist) -> Tuple[int, List[str]]:
    """
    Replace all (inverter -> buffer) pairs with a single inverter.
    Returns (count, list of removed buffer names).
    """
    fo = nl.fanout_map()
    drv = nl.driver_map()
    removed = []

    for gname, g in list(nl.gates.items()):
        if g.gate_type != 'buf':
            continue
        inp_net = g.inputs[0] if g.inputs else None
        if not inp_net:
            continue
        drv_gate_name = drv.get(inp_net)
        if not drv_gate_name or drv_gate_name in ("PI", "CONST"):
            continue
        drv_gate = nl.gates.get(drv_gate_name)
        if not drv_gate or drv_gate.gate_type != 'not':
            continue
        # Check inverter has only this buffer as consumer (fanout=1)
        if len(fo.get(inp_net, [])) != 1:
            continue

        # Reconnect: buf's output net now driven by inverter directly
        old_inv_output = inp_net
        buf_output = g.output

        # Change inverter's output to buf_output
        nl.gates[drv_gate_name] = Gate(
            name=drv_gate_name, gate_type='not',
            output=buf_output, inputs=drv_gate.inputs
        )
        # Remove buffer
        del nl.gates[gname]
        if old_inv_output in nl.wires:
            nl.wires.remove(old_inv_output)
        removed.append(gname)

    return len(removed), removed


def optimize_cone_depth(nl: Netlist, output_net: str, max_depth: int) -> Tuple[bool, str]:
    """
    Attempt to optimize the cone of output_net so depth <= max_depth.
    Uses simple restructuring: balance AND/OR trees.
    Returns (success, explanation).
    """
    from analysis import compute_logic_depth, find_cone
    cone = find_cone(nl, output_net)
    current_depth, path = compute_logic_depth(nl, to_net=output_net)

    if current_depth <= max_depth:
        return True, f"Current depth {current_depth} already satisfies max_depth={max_depth}."

    # Strategy: find the deepest chain and try to balance associative gates
    explanation = _try_balance_cone(nl, output_net, max_depth)
    new_depth, _ = compute_logic_depth(nl, to_net=output_net)

    if new_depth <= max_depth:
        return True, f"Optimized cone depth from {current_depth} to {new_depth}. {explanation}"
    else:
        return False, (f"Could not reduce depth to {max_depth} (achieved {new_depth}). "
                       f"Deep structural restructuring required beyond current engine capability.")


def _try_balance_cone(nl: Netlist, output_net: str, max_depth: int) -> str:
    """
    Simple balancing: find long chains of the same associative gate type and
    reconstruct as a balanced tree.
    """
    from analysis import compute_logic_depth, find_cone
    cone = find_cone(nl, output_net)
    drv = nl.driver_map()
    fo = nl.fanout_map()

    # Collect chains of same-type gates
    def collect_leaves(net: str, gtype: str, visited: set) -> List[str]:
        """Collect all leaf inputs of a tree of gtype gates."""
        if net in visited:
            return [net]
        gn = drv.get(net)
        if not gn or gn not in nl.gates:
            return [net]
        g = nl.gates[gn]
        if g.gate_type != gtype or gn not in cone:
            return [net]
        # Check fanout - only optimize if single-output use
        if len(fo.get(net, [])) > 1:
            return [net]
        visited.add(net)
        leaves = []
        for inp in g.inputs:
            leaves.extend(collect_leaves(inp, gtype, visited))
        return leaves

    changed = False
    for gname in list(cone):
        if gname not in nl.gates:
            continue
        g = nl.gates[gname]
        if g.gate_type not in ('and', 'or', 'nand', 'nor'):
            continue
        # Collect the full tree
        visited: Set[str] = set()
        leaves = collect_leaves(g.output, g.gate_type, visited)
        if len(leaves) < 4:  # only bother for trees with 4+ leaves
            continue

        # Remove old tree nodes
        for old_gate in list(visited):
            old_gn = drv.get(old_gate)
            if old_gn and old_gn in nl.gates and old_gn in cone:
                pass  # we'll rebuild

        # Build balanced binary tree
        _rebuild_balanced_tree(nl, leaves, g.gate_type, g.output)
        changed = True
        break  # one restructuring per call for safety

    return "Applied balanced tree restructuring." if changed else "No restructuring applied."


def _rebuild_balanced_tree(nl: Netlist, leaves: List[str], gtype: str, final_output: str):
    """Build a balanced binary tree of gtype gates driving final_output from leaves."""
    if len(leaves) == 1:
        # single input - use buf
        buf_name = _fresh_name(nl, "Ubal_tree_")
        nl.gates[buf_name] = Gate(name=buf_name, gate_type='buf', output=final_output, inputs=[leaves[0]])
        return

    current = list(leaves)
    level = 0
    while len(current) > 1:
        next_level = []
        for i in range(0, len(current) - 1, 2):
            a, b = current[i], current[i+1]
            if i + 2 < len(current) or len(current) > 2:
                out_net = _fresh_name(nl, f"_btree_{level}_")
                nl.wires.append(out_net)
            else:
                out_net = final_output  # last gate drives the actual output
            gname = _fresh_name(nl, f"Ubtree_")
            nl.gates[gname] = Gate(name=gname, gate_type=gtype, output=out_net, inputs=[a, b])
            next_level.append(out_net)
        if len(current) % 2 == 1:
            next_level.append(current[-1])
        current = next_level
        level += 1
