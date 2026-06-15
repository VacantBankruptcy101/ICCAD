"""
Tool Dispatcher: maps LLM tool call dicts to EDA engine operations
and formats human-readable responses.
"""
from __future__ import annotations
import os
import re
from typing import Any, Dict, List, Optional, Tuple

from netlist import Netlist, parse_verilog, write_verilog
import analysis as ana
import transform as tr


class ToolDispatcher:
    def __init__(self):
        self.netlist: Optional[Netlist] = None
        self.design_path: Optional[str] = None

    def dispatch(self, tool_call: Dict) -> str:
        tool = tool_call.get("tool", "unknown")

        try:
            handler = getattr(self, f"_tool_{tool}", None)
            if handler is None:
                return f"Unknown tool: '{tool}'. Please rephrase the request."
            return handler(tool_call)
        except Exception as e:
            return f"Error executing '{tool}': {e}"

    def _require_design(self) -> Optional[str]:
        if self.netlist is None:
            return "No design loaded. Please read a design first."
        return None

    # ──────────────────────────────────────────────────────────────────
    # Basic Operations
    # ──────────────────────────────────────────────────────────────────

    def _tool_set_testcase(self, args: Dict) -> str:
        name = args.get("name", "unknown")
        log = args.get("log", f"{name}.log")
        return f"Acknowledged. Initialized testcase \"{name}\". All subsequent responses will be recorded to {log}.\nDesign state is empty and ready for commands."

    def _tool_read_design(self, args: Dict) -> str:
        path = args.get("path", "")
        path = path.strip('"\'')

        if not os.path.exists(path):
            return f"Error: file not found at '{path}'."

        with open(path) as f:
            text = f.read()

        self.netlist = parse_verilog(text)
        self.design_path = path

        n = self.netlist
        num_gates = len(n.gates)
        num_ff = sum(1 for g in n.gates.values() if g.is_dff())
        num_comb = num_gates - num_ff
        return (
            f"Loaded gate-level Verilog from \"{path}\" successfully.\n"
            f"  Module: {n.module_name}\n"
            f"  Primary inputs:  {len(n.inputs)}\n"
            f"  Primary outputs: {len(n.outputs)}\n"
            f"  Combinational gates: {num_comb}\n"
            f"  Flip-flops (DFF):    {num_ff}\n"
            f"  Total instances:     {num_gates}"
        )

    def _tool_write_design(self, args: Dict) -> str:
        err = self._require_design()
        if err:
            return err
        path = args.get("path", "output.v").strip('"\'')

        # Ensure directory exists
        dirpath = os.path.dirname(path)
        if dirpath:
            os.makedirs(dirpath, exist_ok=True)

        write_verilog(self.netlist, path)
        n = self.netlist
        num_gates = len(n.gates)
        return (
            f"Wrote the modified netlist to \"{path}\" successfully.\n"
            f"  Module: {n.module_name}\n"
            f"  Total instances: {num_gates}"
        )

    def _tool_design_info(self, args: Dict) -> str:
        err = self._require_design()
        if err:
            return err
        n = self.netlist
        num_gates = len(n.gates)
        num_ff = sum(1 for g in n.gates.values() if g.is_dff())
        from collections import Counter
        type_counts = Counter(g.gate_type for g in n.gates.values())
        lines = [
            f"Design: {n.module_name}",
            f"  Primary inputs:  {len(n.inputs)} — {', '.join(n.inputs[:8])}{'...' if len(n.inputs) > 8 else ''}",
            f"  Primary outputs: {len(n.outputs)} — {', '.join(n.outputs[:8])}{'...' if len(n.outputs) > 8 else ''}",
            f"  Total instances: {num_gates}",
        ]
        for gtype, cnt in sorted(type_counts.items()):
            lines.append(f"    {gtype}: {cnt}")
        return "\n".join(lines)

    # ──────────────────────────────────────────────────────────────────
    # Analysis Tools
    # ──────────────────────────────────────────────────────────────────

    def _tool_path_exists(self, args: Dict) -> str:
        err = self._require_design()
        if err:
            return err
        n = self.netlist
        from_name = args.get("from", "")
        to_name = args.get("to", "")
        exclude = args.get("exclude")

        from_net = _resolve(n, from_name)
        to_net = _resolve(n, to_name)
        excl_net = _resolve(n, exclude) if exclude else None

        if from_net is None:
            return f"Signal '{from_name}' not found in design."
        if to_net is None:
            return f"Signal '{to_name}' not found in design."

        exists, path = ana.path_exists(n, from_net, to_net, excl_net)

        if exists:
            path_str = " → ".join(path[:20])
            if len(path) > 20:
                path_str += " → ..."
            return (
                f"Yes, a combinational path exists from '{from_net}' to '{to_net}'"
                + (f" (excluding '{excl_net}')" if excl_net else "") + ".\n"
                f"Example path: {path_str}"
            )
        else:
            return (
                f"No combinational path found from '{from_net}' to '{to_net}'"
                + (f" (excluding '{excl_net}')" if excl_net else "") + "."
            )

    def _tool_every_path_through(self, args: Dict) -> str:
        err = self._require_design()
        if err:
            return err
        n = self.netlist
        from_net = _resolve(n, args.get("from", ""))
        to_net = _resolve(n, args.get("to", ""))
        through_net = _resolve(n, args.get("through", ""))

        if not from_net:
            return f"Signal '{args.get('from')}' not found."
        if not to_net:
            return f"Signal '{args.get('to')}' not found."
        if not through_net:
            return f"Signal '{args.get('through')}' not found."

        result, counterex = ana.every_path_passes_through(n, from_net, to_net, through_net)
        if result:
            return (
                f"Yes, every combinational path from '{from_net}' to '{to_net}' "
                f"passes through '{through_net}'."
            )
        else:
            if counterex:
                path_str = " → ".join(counterex[:20])
                return (
                    f"No. There exists a path from '{from_net}' to '{to_net}' "
                    f"that does NOT pass through '{through_net}'.\n"
                    f"Counterexample: {path_str}"
                )
            else:
                return (
                    f"No path exists from '{from_net}' to '{to_net}' "
                    f"(regardless of '{through_net}')."
                )

    def _tool_max_depth(self, args: Dict) -> str:
        err = self._require_design()
        if err:
            return err
        n = self.netlist
        from_raw = args.get("from")
        to_raw = args.get("to")

        from_net = _resolve(n, from_raw) if from_raw else None
        to_net = _resolve(n, to_raw) if to_raw else None

        depth, path = ana.compute_logic_depth(n, from_net, to_net)

        loc = ""
        if from_net:
            loc += f" from '{from_net}'"
        if to_net:
            loc += f" to '{to_net}'"

        if depth < 0:
            return f"Could not compute depth{loc} (combinational cycle or parse error)."

        lines = [f"The maximum logic depth{loc} is {depth} gate levels."]
        if path:
            lines.append(f"One longest combinational path ({depth} levels):")
            for gname in path:
                g = n.gates.get(gname)
                if g:
                    lines.append(f"  [{g.gate_type.upper()}] {gname}: {g.output} = {g.gate_type}({', '.join(g.inputs)})")
        return "\n".join(lines)

    def _tool_cone_info(self, args: Dict) -> str:
        err = self._require_design()
        if err:
            return err
        n = self.netlist
        net_raw = args.get("net", "")
        net = _resolve(n, net_raw)
        if not net:
            return f"Signal '{net_raw}' not found."

        cone = ana.find_cone(n, net)
        from collections import Counter
        type_counts = Counter(n.gates[gn].gate_type for gn in cone if gn in n.gates)
        lines = [f"Logic cone of '{net}': {len(cone)} gates total."]
        for gtype, cnt in sorted(type_counts.items()):
            lines.append(f"  {gtype}: {cnt}")
        return "\n".join(lines)

    def _tool_large_cones(self, args: Dict) -> str:
        err = self._require_design()
        if err:
            return err
        threshold = int(args.get("threshold", 100))
        results = ana.outputs_with_large_cones(self.netlist, threshold)
        if not results:
            return f"No primary outputs have a logic cone with more than {threshold} gates."
        lines = [f"Primary outputs with cone > {threshold} gates:"]
        for out, cnt in results:
            lines.append(f"  {out}: {cnt} gates")
        return "\n".join(lines)

    def _tool_clock_domains(self, args: Dict) -> str:
        err = self._require_design()
        if err:
            return err
        domains = ana.get_clock_domains(self.netlist)
        if not domains:
            return "No flip-flops found in the current design."
        lines = ["Clock domain report:"]
        for clk, ffs in sorted(domains.items()):
            lines.append(f"  Clock '{clk}': {len(ffs)} DFF(s) — {', '.join(ffs[:8])}{'...' if len(ffs) > 8 else ''}")
        return "\n".join(lines)

    def _tool_same_clock(self, args: Dict) -> str:
        err = self._require_design()
        if err:
            return err
        ff1 = args.get("ff1", "")
        ff2 = args.get("ff2", "")
        same, clk1, clk2 = ana.same_clock_domain(self.netlist, ff1, ff2)
        if same:
            return f"Yes, '{ff1}' and '{ff2}' are in the same clock domain (clock: '{clk1}')."
        else:
            return f"No, '{ff1}' (clock: '{clk1}') and '{ff2}' (clock: '{clk2}') are in different clock domains."

    def _tool_find_gates(self, args: Dict) -> str:
        err = self._require_design()
        if err:
            return err
        gtype = args.get("gate_type", "buf")
        pattern = args.get("name_pattern")
        gates = ana.find_gates_matching(self.netlist, gtype, pattern)
        if not gates:
            desc = f"{gtype}" + (f" with name containing '{pattern}'" if pattern else "")
            return f"No {desc} gates found in the current design."
        pat_desc = f" with name containing '{pattern}'" if pattern else ""
        lines = [f"Found {len(gates)} {gtype} gate(s){pat_desc}:"]
        for g in gates[:50]:
            lines.append(f"  {g.name} — output: {g.output}, inputs: {g.inputs}")
        if len(gates) > 50:
            lines.append(f"  ... and {len(gates) - 50} more.")
        return "\n".join(lines)

    def _tool_verify_condition(self, args: Dict) -> str:
        err = self._require_design()
        if err:
            return err
        output = _resolve(self.netlist, args.get("output", ""))
        condition = args.get("condition", "")
        if not output:
            return f"Output signal '{args.get('output')}' not found."
        result = ana.verify_condition(self.netlist, output, condition)
        return f"Verification of '{output}': {result['explanation']}"

    # ──────────────────────────────────────────────────────────────────
    # Transformation Tools
    # ──────────────────────────────────────────────────────────────────

    def _tool_replace_buf_with_gate(self, args: Dict) -> str:
        err = self._require_design()
        if err:
            return err
        pattern = args.get("name_pattern", "")
        new_type = args.get("new_type", "and")
        extra_input = args.get("extra_input", "")
        count, names = tr.replace_buffers_with_gate(self.netlist, pattern, new_type, extra_input)
        if count == 0:
            return f"No buf gates found with name containing '{pattern}'."
        lines = [f"Replaced {count} buf gate(s) matching '{pattern}' with '{new_type}' gates (extra input: '{extra_input}'):"]
        for nm in names:
            lines.append(f"  {nm}")
        return "\n".join(lines)

    def _tool_remove_dangling(self, args: Dict) -> str:
        err = self._require_design()
        if err:
            return err
        count, names = tr.remove_dangling_gates(self.netlist)
        if count == 0:
            return "No dangling gates found. Design is clean."
        lines = [f"Removed {count} dangling gate(s) and their nets:"]
        for nm in names[:20]:
            lines.append(f"  {nm}")
        if len(names) > 20:
            lines.append(f"  ... and {len(names) - 20} more.")
        return "\n".join(lines)

    def _tool_insert_fanout_buffers(self, args: Dict) -> str:
        err = self._require_design()
        if err:
            return err
        net_raw = args.get("net", "")
        max_fo = int(args.get("max_fanout", 8))
        net = _resolve(self.netlist, net_raw) or net_raw
        count, names = tr.insert_buffer_for_fanout(self.netlist, net, max_fo)
        if count == 0:
            fo = self.netlist.fanout_map()
            actual = len(fo.get(net, []))
            return f"Net '{net}' already has fanout {actual} ≤ {max_fo}. No buffers needed."
        return f"Inserted {count} buffer(s) on net '{net}' to limit fanout to ≤ {max_fo}."

    def _tool_balance_depth(self, args: Dict) -> str:
        err = self._require_design()
        if err:
            return err
        source_raw = args.get("source", "")
        targets_raw = args.get("targets", [])
        source = _resolve(self.netlist, source_raw) or source_raw
        targets = [_resolve(self.netlist, t) or t for t in targets_raw]
        count, names = tr.balance_depth(self.netlist, source, targets)
        return f"Balanced paths from '{source}' to {len(targets)} target(s). Inserted {count} buffer(s)."

    def _tool_replace_gate_type(self, args: Dict) -> str:
        err = self._require_design()
        if err:
            return err
        old_type = args.get("old_type", "")
        new_types = args.get("new_types", [])
        cone_out_raw = args.get("cone_output")
        cone_out = _resolve(self.netlist, cone_out_raw) if cone_out_raw else None
        count, names = tr.replace_gate_type(self.netlist, old_type, new_types, cone_out)
        if count == 0:
            return f"No '{old_type}' gates found" + (f" in cone of '{cone_out}'" if cone_out else "") + "."
        scope_desc = f" in cone of '{cone_out}'" if cone_out else ""
        return f"Replaced {count} '{old_type}' gate(s){scope_desc} with equivalent logic using {new_types}."

    def _tool_remove_inv_buf_pairs(self, args: Dict) -> str:
        err = self._require_design()
        if err:
            return err
        count, names = tr.replace_inv_buf_pairs(self.netlist)
        if count == 0:
            return "No inverter→buffer pairs found."
        return f"Removed {count} inverter→buffer pair(s), replaced with single inverter."

    def _tool_optimize_cone_depth(self, args: Dict) -> str:
        err = self._require_design()
        if err:
            return err
        out_raw = args.get("output", "")
        max_d = int(args.get("max_depth", 5))
        out_net = _resolve(self.netlist, out_raw) or out_raw
        success, msg = tr.optimize_cone_depth(self.netlist, out_net, max_d)
        status = "Optimization succeeded" if success else "Optimization attempted"
        return f"{status} for cone of '{out_net}' (target depth ≤ {max_d}): {msg}"

    def _tool_unknown(self, args: Dict) -> str:
        raw = args.get("raw", "")
        return f"Could not parse the LLM response into a known EDA tool. Raw: {raw[:200]}"


def _resolve(nl: Netlist, name: str) -> Optional[str]:
    """Resolve a signal name to a net in the netlist."""
    if not name:
        return None
    # Direct match
    if name in nl.inputs or name in nl.outputs:
        return name
    for g in nl.gates.values():
        if g.output == name:
            return name
    if name in nl.gates:
        return nl.gates[name].output
    # Case-insensitive
    name_l = name.lower()
    for s in list(nl.inputs) + list(nl.outputs):
        if s.lower() == name_l:
            return s
    # Partial match for DFF names
    for gname, g in nl.gates.items():
        if gname.lower() == name_l:
            return g.output
    return None
