# ICCAD 2026 Contest Problem A — Submission: cada0001_alpha

## Overview

This system implements an LLM-assisted gate-level Verilog netlist exploration
and transformation engine. It accepts natural-language requests via stdin,
interprets them, and executes EDA operations on a loaded netlist.

## Architecture

```
stdin ──► ContestSystem ──► LocalHeuristic ──────────────► ToolDispatcher ──► EDA Engine
                    │                                              ▲
                    └──► EDAAgent ──► LLM API (OpenAI/Anthropic) ─┘
```

### Files

| File | Purpose |
|------|---------|
| `cada0001_alpha` | Shell wrapper (entry point) |
| `cada0001/cada0001_alpha.py` | Main loop, stdin/stdout, log management |
| `cada0001/agent.py` | LLM API client + tool spec prompt |
| `cada0001/dispatcher.py` | Maps tool-call dicts → EDA operations + formats responses |
| `cada0001/netlist.py` | Verilog parser, Netlist data structure, Verilog writer |
| `cada0001/analysis.py` | Path analysis, depth computation, cone analysis, clock domains |
| `cada0001/transform.py` | Gate replacement, buffer insertion, optimization, dead removal |

## Usage

```bash
./cada0001_alpha -config config.yaml
```

Requests are read from stdin, one per line. Responses are printed to stdout
in the required `#RESPONSE N` / `#END N` format and mirrored to `<case>.log`.

## Supported Capabilities

### Analysis
- Maximum logic depth (from/to specific nets or globally)
- Combinational path existence (with optional exclusion)
- "Every path passes through" queries
- Fanin cone gate count per output
- Clock domain analysis / same-clock check for DFFs
- Large-cone output reporting
- Gate search by type and name pattern

### Transformations
- Replace buf gates matching a name pattern with a 2-input gate (+ extra input)
- Remove dangling (unconnected-to-output) gates and nets
- Insert buffer trees for high-fanout nets
- Balance path depths with minimal buffer insertion
- Replace gate types with equivalent logic (e.g. OR → NAND+NOT)
- Remove inverter→buffer pairs (replace with single inverter)
- Optimize cone logic depth via balanced tree restructuring

## Design Decisions

1. **Local heuristic parsing first**: ~90% of contest patterns are handled by
   regex rules without any LLM API call, giving sub-second response times and
   eliminating API latency/error risk for basic operations.

2. **Stateful request tracking**: Remembers last-found gate type/pattern for
   natural follow-ups like "replace the found buffers with...".

3. **LLM as fallback**: For novel phrasings, the EDAAgent calls the configured
   LLM (OpenAI or Anthropic) with a structured tool-call prompt describing all
   available operations.

4. **DFF-aware analysis**: DFFs act as combinational cut points; paths are not
   traced through flip-flops, matching real timing analysis semantics.

5. **Robust Verilog I/O**: Parser handles primitives, DFFs, buses, constants,
   and comments. Writer produces clean, standards-compliant output.

## Dependencies

- Python 3.8+
- `networkx` (graph algorithms)
- `pyyaml` (config file parsing)
- Standard library only for LLM API calls (urllib)
