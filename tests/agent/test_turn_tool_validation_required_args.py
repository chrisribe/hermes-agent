import types

from agent.turn_tool_validation import validate_tool_calls


class _StubAgent:
    def __init__(self):
        self.valid_tool_names = {"terminal"}
        self.tools = [
            {
                "type": "function",
                "function": {
                    "name": "terminal",
                    "parameters": {
                        "type": "object",
                        "properties": {"command": {"type": "string"}},
                        "required": ["command"],
                    },
                },
            }
        ]
        self._invalid_tool_retries = 0
        self._invalid_json_retries = 0
        self._invalid_required_args_retries = 0
        self.log_prefix = ""

    def _uniquify_tool_call_ids(self, tool_calls):
        return tool_calls

    def _repair_tool_call(self, _name):
        return None

    def _buffer_vprint(self, *_args, **_kwargs):
        return None

    def _flush_status_buffer(self):
        return None

    def _vprint(self, *_args, **_kwargs):
        return None

    def _build_assistant_message(self, _assistant_message, _finish_reason):
        return {"role": "assistant", "tool_calls": []}

    def _persist_session(self, _messages, _conversation_history):
        return None

    def _cleanup_task_resources(self, _task_id):
        return None


def _tool_call(name: str, arguments: str, call_id: str = "call_1"):
    return types.SimpleNamespace(
        id=call_id,
        call_id=call_id,
        function=types.SimpleNamespace(name=name, arguments=arguments),
    )


def test_missing_required_args_returns_tool_error_and_continue():
    agent = _StubAgent()
    assistant_message = types.SimpleNamespace(tool_calls=[_tool_call("terminal", "")])
    messages = []

    verdict = validate_tool_calls(
        agent,
        assistant_message,
        "tool_calls",
        messages=messages,
        conversation_history=[],
        api_call_count=1,
        effective_task_id="default",
    )

    assert verdict.action == "continue"
    assert agent._invalid_required_args_retries == 1
    assert len(messages) == 2
    assert messages[0]["role"] == "assistant"
    assert messages[1]["role"] == "tool"
    assert "missing required argument(s): command" in messages[1]["content"]


def test_missing_required_args_third_strike_returns_partial_exit():
    agent = _StubAgent()
    agent._invalid_required_args_retries = 2
    assistant_message = types.SimpleNamespace(tool_calls=[_tool_call("terminal", "{}")])

    verdict = validate_tool_calls(
        agent,
        assistant_message,
        "tool_calls",
        messages=[],
        conversation_history=[],
        api_call_count=3,
        effective_task_id="default",
    )

    assert verdict.action == "return"
    assert verdict.result is not None
    assert verdict.result.get("partial") is True
    assert "repeatedly omitted required tool arguments" in verdict.result.get("error", "")
    assert agent._invalid_required_args_retries == 0
