"""
LLM Agent: interprets natural-language requests and dispatches to EDA engine.
Uses function-calling style prompting with JSON tool dispatch.
"""
from __future__ import annotations
import json
import re
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import yaml

# ─────────────────────────────────────────────────────────────────────────────
# LLM API Client
# ─────────────────────────────────────────────────────────────────────────────

def load_config(config_path: str) -> Dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def call_llm(config: Dict, messages: List[Dict], system: str = "") -> str:
    """Call configured LLM and return text response."""
    provider = config.get("provider", "openai")

    if provider == "anthropic":
        return _call_anthropic(config, messages, system)
    else:
        return _call_openai(config, messages, system)


def _call_openai(config: Dict, messages: List[Dict], system: str) -> str:
    import urllib.request
    import urllib.error

    cfg = config.get("openai", {})
    api_key = cfg.get("api_key", os.environ.get("OPENAI_API_KEY", ""))
    model = cfg.get("model", "gpt-4o-mini")
    gen = config.get("generation", {})
    temperature = gen.get("temperature", 0.2)
    max_tokens = gen.get("max_output_tokens", 4096)

    payload = {
        "model": model,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "messages": []
    }
    if system:
        payload["messages"].append({"role": "system", "content": system})
    payload["messages"].extend(messages)

    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        "https://api.openai.com/v1/chat/completions",
        data=data,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        },
        method="POST"
    )
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=90) as resp:
                result = json.loads(resp.read())
                return result["choices"][0]["message"]["content"]
        except Exception as e:
            if attempt == 2:
                return f"LLM_ERROR: {e}"
            time.sleep(2 ** attempt)
    return "LLM_ERROR: max retries exceeded"


def _call_anthropic(config: Dict, messages: List[Dict], system: str) -> str:
    import urllib.request
    import urllib.error

    cfg = config.get("anthropic", {})
    api_key = cfg.get("api_key", os.environ.get("ANTHROPIC_API_KEY", ""))
    model = cfg.get("model", "claude-haiku-4-5")
    gen = config.get("generation", {})
    temperature = gen.get("temperature", 0.2)
    max_tokens = gen.get("max_output_tokens", 4096)

    payload = {
        "model": model,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "messages": messages
    }
    if system:
        payload["system"] = system

    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages",
        data=data,
        headers={
            "Content-Type": "application/json",
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01"
        },
        method="POST"
    )
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=90) as resp:
                result = json.loads(resp.read())
                return result["content"][0]["text"]
        except Exception as e:
            if attempt == 2:
                return f"LLM_ERROR: {e}"
            time.sleep(2 ** attempt)
    return "LLM_ERROR: max retries exceeded"


# ─────────────────────────────────────────────────────────────────────────────
# Tool Definitions (sent to LLM as part of system prompt)
# ─────────────────────────────────────────────────────────────────────────────

TOOL_SPEC = """
You are an EDA (Electronic Design Automation) assistant that operates on gate-level Verilog netlists.
You interpret natural-language requests and respond with a JSON object indicating which EDA tool to call.

Available tools and their JSON format:

1. read_design - Load a Verilog file
   {"tool": "read_design", "path": "<filepath>"}

2. write_design - Write current netlist to file
   {"tool": "write_design", "path": "<filepath>"}

3. set_testcase - Set testcase name and start logging
   {"tool": "set_testcase", "name": "<testcase_name>", "log": "<log_filename>"}

4. path_exists - Check if a combinational path exists between two nets
   {"tool": "path_exists", "from": "<net_or_signal>", "to": "<net_or_signal>", "exclude": "<optional_net>"}

5. every_path_through - Check if every path from A to B passes through C
   {"tool": "every_path_through", "from": "<A>", "to": "<B>", "through": "<C>"}

6. max_depth - Compute maximum logic depth (optionally from/to specific nets)
   {"tool": "max_depth", "from": "<optional_net>", "to": "<optional_net>"}

7. cone_info - Get gate count and info about the fanin cone of a net
   {"tool": "cone_info", "net": "<net_or_output>"}

8. large_cones - Find outputs whose cone exceeds a threshold
   {"tool": "large_cones", "threshold": <int>}

9. clock_domains - Report clock domain information for DFFs
   {"tool": "clock_domains"}

10. same_clock - Check if two flip-flops share the same clock
    {"tool": "same_clock", "ff1": "<dff_name>", "ff2": "<dff_name>"}

11. find_gates - Find gates by type and/or name pattern
    {"tool": "find_gates", "gate_type": "<type>", "name_pattern": "<optional_substring>"}

12. replace_buf_with_gate - Replace buf gates matching a name pattern with a 2-input gate
    {"tool": "replace_buf_with_gate", "name_pattern": "<pattern>", "new_type": "<gate_type>", "extra_input": "<net_name>"}

13. remove_dangling - Remove gates/nets not driving any primary output
    {"tool": "remove_dangling"}

14. insert_fanout_buffers - Insert buffer tree on a net to limit fanout
    {"tool": "insert_fanout_buffers", "net": "<net>", "max_fanout": <int>}

15. balance_depth - Balance path depths from source to multiple targets with minimal buffers
    {"tool": "balance_depth", "source": "<net>", "targets": ["<t1>", "<t2>", ...]}

16. replace_gate_type - Replace all gates of one type with equivalent logic from another type set
    {"tool": "replace_gate_type", "old_type": "<type>", "new_types": ["<t1>", "<t2>"], "cone_output": "<optional_net>"}

17. remove_inv_buf_pairs - Remove inverter->buffer pairs, replacing with single inverter
    {"tool": "remove_inv_buf_pairs"}

18. optimize_cone_depth - Optimize logic cone of an output to meet a max depth constraint
    {"tool": "optimize_cone_depth", "output": "<net>", "max_depth": <int>}

19. design_info - Report basic statistics about the loaded design
    {"tool": "design_info"}

20. verify_condition - Structurally analyze a condition on an output
    {"tool": "verify_condition", "output": "<net>", "condition": "<description>"}

IMPORTANT RULES:
- Always respond with ONLY a JSON object. No prose, no explanation outside JSON.
- If a request mentions "gate enable" AND gate or similar insertions before buffers, use replace_buf_with_gate.
- Use the EXACT net names and file paths given in the request.
- For "max depth" questions, use max_depth tool.
- For "path from X to Y" questions, use path_exists.
- For "every path passes through" questions, use every_path_through.
- If the request is purely a greeting or testcase start, use set_testcase.
- For unknown or ambiguous tasks, use design_info as a fallback.
"""


# ─────────────────────────────────────────────────────────────────────────────
# EDA Agent
# ─────────────────────────────────────────────────────────────────────────────

class EDAAgent:
    def __init__(self, config: Dict):
        self.config = config
        self.netlist = None
        self.conversation_history: List[Dict] = []

    def interpret(self, request: str) -> Dict:
        """Ask LLM to interpret request, return tool call dict."""
        self.conversation_history.append({"role": "user", "content": request})

        response = call_llm(
            self.config,
            self.conversation_history,
            system=TOOL_SPEC
        )

        self.conversation_history.append({"role": "assistant", "content": response})

        # Parse JSON from response
        tool_call = _extract_json(response)
        return tool_call

    def reset_conversation(self):
        self.conversation_history = []


def _extract_json(text: str) -> Dict:
    """Extract JSON object from LLM response."""
    # Try direct parse
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Find JSON block
    for pattern in [r'```json\s*(.*?)\s*```', r'```\s*(.*?)\s*```', r'(\{.*?\})', r'(\{.*\})']:
        m = re.search(pattern, text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(1))
            except json.JSONDecodeError:
                continue

    # Fallback: try to extract the largest {...} block
    start = text.find('{')
    end = text.rfind('}')
    if start != -1 and end != -1:
        try:
            return json.loads(text[start:end+1])
        except json.JSONDecodeError:
            pass

    return {"tool": "unknown", "raw": text}
