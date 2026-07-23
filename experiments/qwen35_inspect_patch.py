"""Teach Inspect's HF handler to parse Qwen3.5-style tool calls, so the hf provider can drive
Qwen3.5 agentically in-process (and therefore expose hidden_states to a scorer).

Two gaps in inspect_ai/model/_providers/util/hf_handler.py:

1. `model_specific_tool_parse` gates on the model-name string: the Qwen branch requires BOTH
   "qwen" and "instruct". `Qwen/Qwen3.5-27B` has neither "instruct" nor "coder", so it falls
   through to `parse_unknown_tool_calls`. That fallback DOES match on "<tool_call>" — but it then
   hands the inner text to `parse_tool_call_content`, which expects Qwen2.5-style JSON
   (`{"name":..., "arguments":{...}}`). Qwen3.5 instead emits nested XML:

       <tool_call>
       <function=NAME>
       <parameter=KEY>
       value
       </parameter>
       </function>
       </tool_call>

   so `json.loads` fails and every call becomes ToolCall(function="unknown", parse_error=...).

2. `parse_tool_call_content` rejects empty arguments (`if not name or not arguments`), which would
   break the 14 zero-argument AgentHarm tools (get_contacts, get_client_emails, ...) even once the
   format is parsed correctly.

We patch both. Detection is FORMAT-driven ("<function=" in the response) rather than name-driven, so
it fixes the name-gating problem generally and leaves every other model's path untouched.

Usage:  import qwen35_inspect_patch; qwen35_inspect_patch.install()   (before inspect_eval)
"""
import json
import re

from shortuuid import uuid

from inspect_ai.model._call_tools import parse_tool_call, tool_parse_error_message
from inspect_ai.model._providers.util import hf_handler
from inspect_ai.tool._tool_call import ToolCall

FUNC_RE = re.compile(r"<function=([^>\s]+)\s*>(.*?)</function>", re.DOTALL)
PARAM_RE = re.compile(r"<parameter=([^>\s]+)\s*>(.*?)</parameter>", re.DOTALL)
TOOLCALL_BLOCK_RE = re.compile(r"<tool_call>.*?</tool_call>", re.DOTALL)

_orig_model_specific_tool_parse = hf_handler.model_specific_tool_parse
_orig_parse_tool_call_content = hf_handler.parse_tool_call_content


def _coerce(raw: str):
    """Parameter values arrive as raw text. Try JSON so ints/bools/lists come through typed,
    else keep the string (trimming only the newlines the template puts around the value)."""
    v = raw.strip("\n")
    try:
        return json.loads(v.strip())
    except Exception:
        return v


def parse_qwen35_tool_calls(response: str) -> tuple[list[str], str]:
    """Qwen3.5 nested-XML tool calls -> the JSON strings Inspect's parser expects.

    Returns (function_call_json_strings, remaining_content).
    """
    calls = []
    for m in FUNC_RE.finditer(response):
        name = m.group(1).strip()
        args = {k.strip(): _coerce(v) for k, v in PARAM_RE.findall(m.group(2))}
        calls.append(json.dumps({"name": name, "arguments": args}))

    remaining = TOOLCALL_BLOCK_RE.sub("", response)      # whole wrapped blocks
    remaining = FUNC_RE.sub("", remaining)               # any unwrapped function blocks
    remaining = remaining.replace("<tool_call>", "").replace("</tool_call>", "")
    return calls, remaining.strip()


def _model_specific_tool_parse(response: str, model_name: str) -> tuple[str, list[str]]:
    if "<function=" in response:
        function_calls, content = parse_qwen35_tool_calls(response)
        return content, function_calls
    return _orig_model_specific_tool_parse(response, model_name)


def _parse_tool_call_content(content: str, tools) -> ToolCall:
    """Same as upstream, except an empty `arguments` dict is valid (zero-argument tools)."""
    try:
        data = json.loads(content)
        if not isinstance(data, dict):
            raise ValueError("The provided arguments are not a JSON dictionary.")
        if "parameters" in data:
            data["arguments"] = data.pop("parameters")
        name = data.get("name", None)
        arguments = data.get("arguments", None)
        if not name or arguments is None:          # <-- upstream: `not arguments`
            raise ValueError(
                "Required 'name' and 'arguments' not provided in JSON dictionary."
            )
        return parse_tool_call(f"{name}_{uuid()}", name, json.dumps(arguments), tools)
    except Exception as ex:
        parse_error = tool_parse_error_message(content, ex)
        hf_handler.logger.info(parse_error)
        return ToolCall(id="unknown", function="unknown", arguments={}, parse_error=parse_error)


def install() -> None:
    """Patch the HF handler in place (call before inspect_eval)."""
    hf_handler.model_specific_tool_parse = _model_specific_tool_parse
    hf_handler.parse_tool_call_content = _parse_tool_call_content


def uninstall() -> None:
    hf_handler.model_specific_tool_parse = _orig_model_specific_tool_parse
    hf_handler.parse_tool_call_content = _orig_parse_tool_call_content
