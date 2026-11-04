import sys
import os
import json
import re
import ollama
from tools import AVAILABLE_FUNCTIONS, OLLAMA_TOOLS_SCHEMA

# The model name matches the one created by: `ollama create Shellcoder -f Modelfile`
MODEL_NAME = "Shellcoder"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def load_identity():
    """Extract the SYSTEM prompt from the Modelfile so the agent always stays in character."""
    modelfile = os.path.join(BASE_DIR, "Shellcoder.Modelfile")
    try:
        with open(modelfile, 'r', encoding='utf-8') as f:
            content = f.read()
        match = re.search(r'SYSTEM\s+"""(.*?)"""', content, re.DOTALL)
        if match:
            return match.group(1).strip()
    except Exception:
        pass
    return ""

IDENTITY_PROMPT = load_identity()

# Injected as a system message when the model doesn't support native Ollama tools.
# Instructs the model to emit <tool_call> JSON blocks instead.
TOOL_PROMPT = (
    "You have access to tools. When you need to call one, output ONLY a JSON block "
    "in this exact format and nothing else on that turn:\n"
    "<tool_call>\n"
    '{"name": "<tool_name>", "arguments": {"<key>": "<value>"}}\n'
    "</tool_call>\n\n"
    "Available tools:\n"
    "  read_file(filepath)             - Read a local file.\n"
    "  write_file(filepath, content)   - Write content to a file.\n"
    "  run_bash(command)               - Run a shell command.\n"
    "  search_vault(query_text)        - Search the local RAG knowledge base.\n\n"
    "After receiving a tool result, continue your response normally. "
    "Do NOT use tools unless the user's request requires it."
)


def execute_tool(name: str, args: dict) -> str:
    """Dispatch a tool call and return the string result."""
    print(f"\n[!] Agent executing tool: {name}({args})")
    if name not in AVAILABLE_FUNCTIONS:
        return f"Unknown tool: {name}"
    try:
        return str(AVAILABLE_FUNCTIONS[name](**args))
    except Exception as e:
        return f"Error in {name}: {e}"


def chat_with_native_tools(messages: list) -> list:
    """One turn via Ollama's native tools API. Returns new messages to append."""
    response = ollama.chat(model=MODEL_NAME, messages=messages, tools=OLLAMA_TOOLS_SCHEMA)
    new_msgs = [response['message']]

    tool_calls = response['message'].get('tool_calls')
    if not tool_calls:
        print(f"\n[SHELLCODER] > {response['message']['content']}")
        return new_msgs

    for call in tool_calls:
        result = execute_tool(call.function.name, call.function.arguments)
        new_msgs.append({'role': 'tool', 'name': call.function.name, 'content': result})

    print("\n[SHELLCODER] > ", end='', flush=True)
    final_text = ""
    for chunk in ollama.chat(model=MODEL_NAME, messages=messages + new_msgs, stream=True):
        token = chunk['message']['content']
        print(token, end='', flush=True)
        final_text += token
    print()
    new_msgs.append({'role': 'assistant', 'content': final_text})
    return new_msgs


def chat_with_prompt_tools(messages: list) -> list:
    """One turn using text-parsed <tool_call> blocks. Returns new messages to append."""
    full_messages = messages.copy()
    if not any(TOOL_PROMPT in m.get('content', '') for m in full_messages):
        # Insert after the identity system message (index 1) if it exists, else at 0
        idx = 1 if full_messages and full_messages[0]['role'] == 'system' else 0
        full_messages.insert(idx, {'role': 'system', 'content': TOOL_PROMPT})

    new_msgs = []
    for _ in range(6):  # up to 5 tool rounds then a final answer
        response = ollama.chat(model=MODEL_NAME, messages=full_messages + new_msgs)
        content = response['message']['content']
        new_msgs.append({'role': 'assistant', 'content': content})

        hits = re.findall(r'<tool_call>\s*(\{.*?\})\s*</tool_call>', content, re.DOTALL)
        if not hits:
            print(f"\n[SHELLCODER] > {content}")
            return new_msgs

        for raw in hits:
            try:
                call = json.loads(raw)
                result = execute_tool(call.get('name', ''), call.get('arguments', {}))
            except json.JSONDecodeError as e:
                result = f"JSON parse error: {e}"
            new_msgs.append({'role': 'user', 'content': f"[Tool result]\n{result}"})

    print(f"\n[SHELLCODER] > {content}")
    return new_msgs


def chat_loop():
    print("===============================================================")
    print("    SHELLCODER AGENT - Unified Security & Exploit Interface    ")
    print("===============================================================")
    print("Type 'exit' or 'quit' to close the session.\n")

    messages = []
    if IDENTITY_PROMPT:
        messages.append({'role': 'system', 'content': IDENTITY_PROMPT})
    use_native_tools = True  # try native tools first; flip to False on first failure

    while True:
        try:
            user_input = input("\n[USER] > ")
            if user_input.strip().lower() in ['exit', 'quit']:
                print("\nTerminating session. Goodbye.")
                break
            if not user_input.strip():
                continue

            messages.append({'role': 'user', 'content': user_input})

            if use_native_tools:
                try:
                    new_msgs = chat_with_native_tools(messages)
                except Exception as e:
                    if 'does not support tools' in str(e):
                        print("[*] Native tool-calling not supported — switching to prompt-based mode.")
                        use_native_tools = False
                        new_msgs = chat_with_prompt_tools(messages)
                    else:
                        messages.pop()
                        print(f"\n[X] Error: {e}")
                        continue
            else:
                new_msgs = chat_with_prompt_tools(messages)

            messages.extend(new_msgs)

        except KeyboardInterrupt:
            print("\n[!] Interrupted. Type 'exit' to quit.")
        except Exception as e:
            print(f"\n[X] Error: {e}")


if __name__ == "__main__":
    chat_loop()
