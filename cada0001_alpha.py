#!/usr/bin/env python3
"""
cada0001_alpha — ICCAD 2026 Contest Problem A
LLM-Assisted Netlist Exploration and Transformation System

Usage:
    ./cada0001_alpha -config <config_file_path>
"""
from __future__ import annotations
import argparse
import os
import re
import sys
import traceback
from typing import Optional

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agent import EDAAgent, load_config
from dispatcher import ToolDispatcher


class ContestSystem:
    def __init__(self, config_path: str):
        self.config = load_config(config_path)
        self.agent = EDAAgent(self.config)
        self.dispatcher = ToolDispatcher()
        self.response_id = 0
        self.case_name: Optional[str] = None
        self.log_path: Optional[str] = None
        self.log_file = None
        # State: last-found gate type and pattern (for stateful "replace the found X")
        self._last_found_gate_type: Optional[str] = None
        self._last_found_pattern: Optional[str] = None

    def _open_log(self, log_path: str):
        if self.log_file:
            self.log_file.close()
        os.makedirs(os.path.dirname(log_path) if os.path.dirname(log_path) else '.', exist_ok=True)
        self.log_file = open(log_path, 'w')

    def _emit(self, text: str):
        """Write to stdout and log file."""
        print(text, flush=True)
        if self.log_file:
            print(text, file=self.log_file, flush=True)

    def _detect_testcase_start(self, request: str) -> Optional[str]:
        """Try to extract testcase name from request without LLM."""
        # Patterns: "testcase case28", "case name is 'test8'", etc.
        patterns = [
            r"testcase\s+(\w+)",
            r"case\s+name\s+is\s+['\"]?(\w+)['\"]?",
            r"beginning of\s+(\w+)",
            r"\bcase\s*(\w+)\b",
        ]
        for pat in patterns:
            m = re.search(pat, request, re.IGNORECASE)
            if m:
                return m.group(1)
        return None

    def _detect_log_file(self, request: str, case_name: str) -> str:
        """Extract log filename from request."""
        m = re.search(r'(\w+\.log)', request, re.IGNORECASE)
        if m:
            return m.group(1)
        return f"{case_name}.log"

    def _detect_file_path(self, request: str, keyword: str) -> Optional[str]:
        """
        Heuristic: extract file path near keyword (load/read/write) without LLM call.
        """
        # Quoted paths
        m = re.search(r"['\"]([^'\"]+\.v)['\"]", request)
        if m:
            return m.group(1)
        # From directory pattern: "from <dir>/" or "in <dir>/"
        m = re.search(r'(?:from|in)\s+(?:directory\s+)?[\'"]?([^\s\'"]+/[^\s\'"]*\.v)[\'"]?', request, re.I)
        if m:
            return m.group(1)
        # Simple filename.v
        m = re.search(r'\b([\w./\-]+\.v)\b', request)
        if m:
            return m.group(1)
        return None

    def handle_request(self, request: str) -> str:
        """
        Main request handler: detect simple cases locally, otherwise call LLM.
        """
        req_lower = request.lower()

        # ── Detect testcase start locally ─────────────────────────────
        is_testcase_start = any(kw in req_lower for kw in [
            "beginning of", "testcase", "case name", "new testcase"
        ])
        if is_testcase_start:
            case_name = self._detect_testcase_start(request)
            if case_name:
                self.case_name = case_name
                log_path = self._detect_log_file(request, case_name)
                self.log_path = log_path
                self._open_log(log_path)
                self.agent.reset_conversation()
                return (
                    f'Acknowledged. Initialized testcase "{case_name}". '
                    f'All subsequent responses will be recorded to {log_path}.\n'
                    f'Design state is empty and ready for commands.'
                )

        # ── Detect read design locally ────────────────────────────────
        is_read = any(kw in req_lower for kw in ["load", "read in", "read design", "read file", "load file"])
        if is_read and '.v' in request:
            path = self._detect_file_path(request, "read")
            if path:
                result = self.dispatcher.dispatch({"tool": "read_design", "path": path})
                return result

        # ── Detect write design locally ───────────────────────────────
        is_write = any(kw in req_lower for kw in ["write out", "output the design", "write the", "write design", "save the design"])
        if is_write and '.v' in request:
            path = self._detect_file_path(request, "write")
            if path:
                result = self.dispatcher.dispatch({"tool": "write_design", "path": path})
                return result

        # ── Local heuristic interpretation before LLM ─────────────────
        local_call = self._local_interpret(request)
        if local_call:
            # Track find_gates results for stateful follow-up
            if local_call.get("tool") == "find_gates":
                self._last_found_gate_type = local_call.get("gate_type")
                self._last_found_pattern = local_call.get("name_pattern")
            return self.dispatcher.dispatch(local_call)

        # ── LLM interpretation ────────────────────────────────────────
        try:
            tool_call = self.agent.interpret(request)
        except Exception as e:
            return f"LLM interpretation error: {e}"

        return self.dispatcher.dispatch(tool_call)

    def _local_interpret(self, request: str) -> Optional[Dict]:
        """
        Pattern-match common requests locally to avoid LLM latency.
        Returns tool call dict or None if uncertain.
        """
        r = request.strip()
        rl = r.lower()

        # max depth / logic depth
        if re.search(r'\b(max(imum)?\s+logic\s+depth|logic\s+depth|maximum\s+depth)', rl):
            from_m = re.search(r'from\s+(?:input\s+)?[\'"]?(\w+)[\'"]?', rl)
            to_m = re.search(r'to\s+(?:output\s+)?[\'"]?(\w+)[\'"]?', rl)
            return {
                "tool": "max_depth",
                "from": from_m.group(1) if from_m else None,
                "to": to_m.group(1) if to_m else None
            }

        # every path through
        if re.search(r'every\s+path', rl):
            from_m = re.search(r'from\s+[\'"]?(\w+)[\'"]?', rl)
            to_m = re.search(r'to\s+[\'"]?(\w+)[\'"]?', rl)
            through_m = re.search(r'through\s+[\'"]?(\w+)[\'"]?', rl)
            if from_m and to_m and through_m:
                return {
                    "tool": "every_path_through",
                    "from": from_m.group(1),
                    "to": to_m.group(1),
                    "through": through_m.group(1)
                }

        # path exists / find a path
        if re.search(r'\b(path\s+from|find\s+a?\s*path)', rl):
            from_m = re.search(r'from\s+(?:input\s+)?[\'"]?(\w+)[\'"]?', rl)
            to_m = re.search(r'to\s+(?:output\s+)?[\'"]?(\w+)[\'"]?', rl)
            excl_m = re.search(r'(?:not\s+pass\s+through|without|excluding)\s+[\'"]?(\w+)[\'"]?', rl)
            if from_m and to_m:
                return {
                    "tool": "path_exists",
                    "from": from_m.group(1),
                    "to": to_m.group(1),
                    "exclude": excl_m.group(1) if excl_m else None
                }

        # clock domain
        if 'clock domain' in rl or 'same clock' in rl:
            # Check if two specific FFs mentioned
            words = re.findall(r'\b(dff\w*|\w*ff\w*)\b', rl)
            ff_candidates = [w for w in words if re.match(r'[a-z]*ff\d*', w, re.I)]
            if len(ff_candidates) >= 2:
                return {"tool": "same_clock", "ff1": ff_candidates[0], "ff2": ff_candidates[1]}
            return {"tool": "clock_domains"}

        # cone info / gate count in cone
        if re.search(r'(logic\s+cone|fanin\s+cone|cone\s+of|gates?\s+in\s+the\s+cone)', rl):
            net_m = re.search(r'(?:cone\s+of|for\s+output)\s+[\'"]?(\w+)[\'"]?', rl)
            if net_m:
                return {"tool": "cone_info", "net": net_m.group(1)}

        # large cones / outputs with more than N gates
        if re.search(r'more\s+than\s+(\d+)\s+gates?', rl):
            n_m = re.search(r'more\s+than\s+(\d+)', rl)
            return {"tool": "large_cones", "threshold": int(n_m.group(1)) if n_m else 100}

        # find buffers / find gates
        if re.search(r'find\b', rl) and re.search(r'\b(buffers?|bufs?|inverters?|not\s+gate|and\s+gate|or\s+gate|nand|nor|xor)', rl):
            gtype_m = re.search(r'\b(buffers?|bufs?|inverters?|nand|nor|xor|xnor|not|and|or)\b', rl)
            gtype = gtype_m.group(1).rstrip('s') if gtype_m else 'buf'
            if gtype in ('buffer', 'buf', 'buffe'):
                gtype = 'buf'
            if gtype == 'inverter':
                gtype = 'not'
            pattern_m = re.search(r"(?:name\s+(?:includes?|include|contains?|match\w*))\s+['\"]?([^\s'\"]+)['\"]?", r, re.I)
            if not pattern_m:
                pattern_m = re.search(r"['\"]([^'\"]+)['\"]", r)
            return {
                "tool": "find_gates",
                "gate_type": gtype,
                "name_pattern": pattern_m.group(1) if pattern_m else None
            }

        # replace buf with AND/gate (gc pattern)
        if re.search(r'(replace|insert)\b', rl) and re.search(r'\b(buffers?|bufs?)\b', rl):
            # Stateful: "replace the found buffers" → use last found pattern
            pattern_m = re.search(r"(?:name\s+(?:includes?|include|contains?|with?))\s+['\"]?([^\s'\"]+)['\"]?", r, re.I)
            if not pattern_m and re.search(r'\bfound\b', rl) and self._last_found_pattern:
                pattern = self._last_found_pattern
            elif pattern_m:
                pattern = pattern_m.group(1)
            else:
                pattern = None
            extra_m = re.search(r'(?:connect\s+(?:the\s+)?other\s+input\s+to|extra\s+input\s+is?)\s+[\'"]?(\w+)[\'"]?', rl)
            new_type_m = re.search(r'\b(and|or|nand|nor|xor|xnor)\b', rl)
            if pattern and extra_m:
                return {
                    "tool": "replace_buf_with_gate",
                    "name_pattern": pattern,
                    "new_type": new_type_m.group(1) if new_type_m else "and",
                    "extra_input": extra_m.group(1)
                }

        # remove dangling
        if re.search(r'\b(remov|delet|clean)\w*\b', rl) and re.search(r'\b(dangling|dead|unused|do not affect)\b', rl):
            return {"tool": "remove_dangling"}

        # remove inverter-buffer pairs
        if re.search(r'(inverter|not).*(buffer|buf)', rl) and 'single' in rl:
            return {"tool": "remove_inv_buf_pairs"}

        # fanout buffering
        if re.search(r'fanout\s+(greater|more|larger)', rl) or 'high-fanout' in rl:
            net_m = re.search(r'(?:net|signal|wire)\s+[\'"]?(\w+)[\'"]?', rl)
            fo_m = re.search(r'(?:greater|more)\s+than\s+(\d+)', rl)
            if net_m:
                return {
                    "tool": "insert_fanout_buffers",
                    "net": net_m.group(1),
                    "max_fanout": int(fo_m.group(1)) if fo_m else 8
                }

        # optimize depth
        if re.search(r'optim\w*.*depth|depth.*(?:≤|<=|less than|at most)', rl):
            out_m = re.search(r'(?:cone\s+of|output\s+|for\s+)\s*[\'"]?(\w+)[\'"]?', rl)
            depth_m = re.search(r'(?:≤|<=|less\s+than\s+or\s+equal\s+to|at\s+most|max\w*\s+depth\s+(?:is\s+)?(?:≤|<=)?)\s*(\d+)', rl)
            if out_m and depth_m:
                return {
                    "tool": "optimize_cone_depth",
                    "output": out_m.group(1),
                    "max_depth": int(depth_m.group(1))
                }

        # replace gate type (OR -> NAND+NOT)
        if re.search(r'replace\s+all\s+\w+\s+(gate|or|and|nor|nand)', rl):
            old_m = re.search(r'replace\s+all\s+(\w+)\s+(?:gate|or|and)', rl)
            # "equivalent logic built only from NAND and NOT"
            allowed = re.findall(r'\b(nand|nor|and|or|not|buf|xor|xnor)\b', rl)
            if old_m and allowed:
                old = old_m.group(1)
                cone_m = re.search(r'(?:in\s+the\s+cone\s+of)\s+[\'"]?(\w+)[\'"]?', rl)
                return {
                    "tool": "replace_gate_type",
                    "old_type": old,
                    "new_types": list(dict.fromkeys(allowed)),
                    "cone_output": cone_m.group(1) if cone_m else None
                }

        # design info
        if re.search(r'(design\s+info|statistics|report\s+the\s+design|what.*in\s+the\s+design)', rl):
            return {"tool": "design_info"}

        return None  # fall through to LLM

    def respond(self, request: str):
        """Process one request, print and log the formatted response."""
        self.response_id += 1
        rid = self.response_id

        response_text = ""
        try:
            response_text = self.handle_request(request)
        except Exception as e:
            response_text = f"Internal error processing request: {e}\n{traceback.format_exc()}"

        self._emit(f"#RESPONSE {rid}")
        self._emit(response_text)
        self._emit(f"#END {rid}")

    def run(self):
        """Main stdin loop."""
        for line in sys.stdin:
            line = line.rstrip('\n')
            if not line:
                continue
            self.respond(line)

        if self.log_file:
            self.log_file.close()


def main():
    parser = argparse.ArgumentParser(description="ICCAD 2026 Contest Problem A — EDA Agent")
    parser.add_argument("-config", required=True, help="Path to LLM configuration YAML file")
    args = parser.parse_args()

    system = ContestSystem(args.config)
    system.run()


if __name__ == "__main__":
    main()
