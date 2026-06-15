"""
Netlist: internal representation of a gate-level Verilog design.
Supports primitives: and, or, nand, nor, not, buf, xor, xnor, dff
"""
from __future__ import annotations
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple
import networkx as nx


PRIMITIVE_GATES = {"and", "or", "nand", "nor", "not", "buf", "xor", "xnor"}
TWO_INPUT_GATES = {"and", "or", "nand", "nor", "xor", "xnor"}
ONE_INPUT_GATES = {"not", "buf"}


@dataclass
class Gate:
    name: str          # instance name
    gate_type: str     # and/or/not/buf/dff/...
    output: str        # output net name
    inputs: List[str]  # input net names (in order)

    def is_dff(self) -> bool:
        return self.gate_type == "dff"

    def is_combinational(self) -> bool:
        return self.gate_type in PRIMITIVE_GATES


@dataclass
class Netlist:
    module_name: str = ""
    inputs: List[str] = field(default_factory=list)   # port names
    outputs: List[str] = field(default_factory=list)  # port names
    wires: List[str] = field(default_factory=list)    # internal wires
    gates: Dict[str, Gate] = field(default_factory=dict)  # name -> Gate

    # Bus declarations: port/wire name -> (high, low) or None for scalar
    port_widths: Dict[str, Optional[Tuple[int, int]]] = field(default_factory=dict)

    def all_nets(self) -> Set[str]:
        nets: Set[str] = set()
        for g in self.gates.values():
            nets.add(g.output)
            nets.update(g.inputs)
        nets.update(self.inputs)
        nets.update(self.outputs)
        return nets

    def fanout_map(self) -> Dict[str, List[str]]:
        """net -> list of gate names that consume it"""
        fo: Dict[str, List[str]] = {}
        for g in self.gates.values():
            for inp in g.inputs:
                fo.setdefault(inp, []).append(g.name)
        return fo

    def driver_map(self) -> Dict[str, str]:
        """net -> gate name that drives it (or 'PI' for primary inputs)"""
        drv: Dict[str, str] = {}
        for name in self.inputs:
            drv[name] = "PI"
        for g in self.gates.values():
            drv[g.output] = g.name
        # constants
        drv["1'b0"] = "CONST"
        drv["1'b1"] = "CONST"
        return drv

    def combinational_graph(self) -> nx.DiGraph:
        """DAG of combinational gates only (ignores DFF outputs as sources)."""
        G = nx.DiGraph()
        # nodes
        for inp in self.inputs:
            G.add_node(inp, kind="PI")
        for g in self.gates.values():
            G.add_node(g.name, kind=g.gate_type)
        # add constant nodes
        G.add_node("1'b0", kind="CONST")
        G.add_node("1'b1", kind="CONST")

        drv = self.driver_map()
        # DFF q outputs act as pseudo-PIs
        for g in self.gates.values():
            if g.is_dff():
                G.add_node(g.output, kind="DFF_Q")

        for g in self.gates.values():
            if g.is_dff():
                continue  # don't propagate through FF
            for net in g.inputs:
                src = drv.get(net)
                if src is None:
                    continue
                if src == "PI":
                    src = net
                elif src == "CONST":
                    src = net
                G.add_edge(src, g.name, net=net)
        return G

    def full_graph(self) -> nx.DiGraph:
        """Full signal-flow graph including DFFs."""
        G = nx.DiGraph()
        drv = self.driver_map()
        for g in self.gates.values():
            G.add_node(g.name, kind=g.gate_type, output=g.output)
            for net in g.inputs:
                src = drv.get(net)
                if src and src not in ("PI", "CONST"):
                    G.add_edge(src, g.name, net=net)
        return G

    def get_gate_by_output(self, net: str) -> Optional[Gate]:
        for g in self.gates.values():
            if g.output == net:
                return g
        return None

    def clone(self) -> "Netlist":
        import copy
        return copy.deepcopy(self)


# ─────────────────────────────────────────────────────────────────────────────
# Verilog Parser
# ─────────────────────────────────────────────────────────────────────────────

def _strip_comments(text: str) -> str:
    # Remove // line comments
    text = re.sub(r'//[^\n]*', '', text)
    # Remove /* block comments */
    text = re.sub(r'/\*.*?\*/', '', text, flags=re.DOTALL)
    return text


def _expand_bus_names(decl_names: str, width: Optional[Tuple[int, int]]) -> List[Tuple[str, Optional[Tuple[int, int]]]]:
    """Parse comma-separated names, return list of (name, width)."""
    names = [n.strip() for n in decl_names.split(',') if n.strip()]
    return [(n, width) for n in names]


def parse_verilog(text: str) -> Netlist:
    text = _strip_comments(text)
    nl = Netlist()

    # Extract module name
    m = re.search(r'\bmodule\s+(\w+)', text)
    if m:
        nl.module_name = m.group(1)

    # Find module body (between first ; and endmodule)
    body_m = re.search(r'\bmodule\b.*?;(.*?)\bendmodule\b', text, re.DOTALL)
    body = body_m.group(1) if body_m else text

    # Parse input/output/wire declarations
    # Match: input/output/wire [msb:lsb] name1, name2;
    decl_re = re.compile(
        r'\b(input|output|wire|reg)\b\s*(?:\[(\d+)\s*:\s*(\d+)\])?\s*([\w\s,\[\]:]+?)\s*;',
        re.DOTALL
    )
    for m in decl_re.finditer(body):
        kind = m.group(1)
        msb = int(m.group(2)) if m.group(2) else None
        lsb = int(m.group(3)) if m.group(3) else None
        width = (msb, lsb) if msb is not None else None
        raw_names = m.group(4)
        # raw_names may contain bus subscripts like a[3:0], skip those
        # Extract just identifier names
        names_part = raw_names
        # Remove [..] suffixes for bus-typed names
        names_part = re.sub(r'\[[\d:]+\]', '', names_part)
        names = [n.strip() for n in names_part.split(',') if n.strip() and re.match(r'^\w+$', n.strip())]
        for nm in names:
            nl.port_widths[nm] = width
            if kind == 'input':
                if nm not in nl.inputs:
                    nl.inputs.append(nm)
            elif kind == 'output':
                if nm not in nl.outputs:
                    nl.outputs.append(nm)
            else:
                if nm not in nl.wires:
                    nl.wires.append(nm)

    # Parse gate instantiations
    # Pattern: gate_type inst_name ( port_list );
    # Also handles dff
    gate_types_pattern = '|'.join(PRIMITIVE_GATES | {'dff'})
    inst_re = re.compile(
        rf'\b({gate_types_pattern})\s+(\w+)\s*\((.*?)\)\s*;',
        re.DOTALL
    )
    for m in inst_re.finditer(body):
        gtype = m.group(1)
        gname = m.group(2)
        port_str = m.group(3).strip()

        if gtype == 'dff':
            gate = _parse_dff_ports(gname, port_str)
        else:
            gate = _parse_primitive_ports(gname, gtype, port_str)

        if gate:
            nl.gates[gname] = gate

    return nl


def _parse_primitive_ports(name: str, gtype: str, port_str: str) -> Optional[Gate]:
    """Primitive gates: first port is output, rest are inputs."""
    # Split by comma, but be careful about nested brackets
    ports = _split_ports(port_str)
    if not ports:
        return None
    output = ports[0].strip()
    inputs = [p.strip() for p in ports[1:]]
    return Gate(name=name, gate_type=gtype, output=output, inputs=inputs)


def _parse_dff_ports(name: str, port_str: str) -> Optional[Gate]:
    """DFF: named port connections .clk(), .rst_n(), .d(), .q()"""
    port_map = {}
    named_re = re.compile(r'\.(\w+)\s*\(\s*(\S+?)\s*\)')
    for m in named_re.finditer(port_str):
        port_map[m.group(1)] = m.group(2)

    if 'q' not in port_map:
        return None

    output = port_map.get('q', '')
    inputs = []
    for p in ['clk', 'rst_n', 'd']:
        inputs.append(port_map.get(p, "1'b0"))

    return Gate(name=name, gate_type='dff', output=output, inputs=inputs)


def _split_ports(s: str) -> List[str]:
    """Split comma-separated port list respecting brackets."""
    parts = []
    depth = 0
    cur = []
    for ch in s:
        if ch in '([':
            depth += 1
            cur.append(ch)
        elif ch in ')]':
            depth -= 1
            cur.append(ch)
        elif ch == ',' and depth == 0:
            parts.append(''.join(cur).strip())
            cur = []
        else:
            cur.append(ch)
    if cur:
        parts.append(''.join(cur).strip())
    return parts


# ─────────────────────────────────────────────────────────────────────────────
# Verilog Writer
# ─────────────────────────────────────────────────────────────────────────────

def write_verilog(nl: Netlist, filepath: str):
    lines = []
    lines.append(f"// Generated by cada0001 EDA engine")
    lines.append(f"")

    # Collect all internal wire nets
    all_internal = set(nl.wires)
    # Also add any net that is a gate output but not a primary port
    for g in nl.gates.values():
        if g.output not in nl.inputs and g.output not in nl.outputs:
            all_internal.add(g.output)

    # Module header
    all_ports = nl.inputs + nl.outputs
    lines.append(f"module {nl.module_name} (")
    port_lines = []
    for p in all_ports:
        port_lines.append(f"    {p}")
    lines.append(",\n".join(port_lines))
    lines.append(");")
    lines.append("")

    # Port declarations
    for p in nl.inputs:
        w = nl.port_widths.get(p)
        if w:
            lines.append(f"    input [{w[0]}:{w[1]}] {p};")
        else:
            lines.append(f"    input {p};")

    for p in nl.outputs:
        w = nl.port_widths.get(p)
        if w:
            lines.append(f"    output [{w[0]}:{w[1]}] {p};")
        else:
            lines.append(f"    output {p};")

    lines.append("")

    # Wire declarations
    if all_internal:
        for w in sorted(all_internal):
            lines.append(f"    wire {w};")
        lines.append("")

    # Gate instantiations
    for gname, g in nl.gates.items():
        if g.gate_type == 'dff':
            lines.append(
                f"    dff {gname} (.clk({g.inputs[0]}), .rst_n({g.inputs[1]}), "
                f".d({g.inputs[2]}), .q({g.output}));"
            )
        else:
            port_list = ", ".join([g.output] + g.inputs)
            lines.append(f"    {g.gate_type} {gname} ({port_list});")

    lines.append("")
    lines.append("endmodule")

    with open(filepath, 'w') as f:
        f.write("\n".join(lines))
