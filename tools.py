import os
import subprocess
from rag_engine import search_vault

def read_file(filepath: str) -> str:
    """Read the contents of a file."""
    filepath = os.path.abspath(os.path.expanduser(filepath))
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        return f"Error reading {filepath}: {str(e)}"

def write_file(filepath: str, content: str) -> str:
    """Write or overwrite content to a file."""
    filepath = os.path.abspath(os.path.expanduser(filepath))
    os.makedirs(os.path.dirname(filepath) or '.', exist_ok=True)
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        return f"Successfully wrote to {filepath}"
    except Exception as e:
        return f"Error writing to {filepath}: {str(e)}"

def run_bash(command: str) -> str:
    """Execute a bash command on the host system."""
    # Guardrail: the command is LLM-generated and runs with shell=True (full RCE
    # on the host). Require explicit human confirmation before executing.
    confirm = input(f"\n[CONFIRM] Run shell command?\n  {command}\n[y/N] ").strip().lower()
    if confirm not in ('y', 'yes'):
        return f"Execution declined by user. Command not run: {command}"
    try:
        result = subprocess.run(
            command, shell=True, text=True, 
            capture_output=True, timeout=30
        )
        out = result.stdout.strip()
        err = result.stderr.strip()
        if not out and not err:
            return f"Command '{command}' executed successfully with no output."
        return f"STDOUT:\n{out}\n\nSTDERR:\n{err}"
    except subprocess.TimeoutExpired:
        return f"Command '{command}' timed out after 30 seconds."
    except Exception as e:
        return f"Error executing command: {str(e)}"

# Dictionary mapping function names to actual Python functions
AVAILABLE_FUNCTIONS = {
    "read_file": read_file,
    "write_file": write_file,
    "run_bash": run_bash,
    "search_vault": search_vault,
}

# The JSON schema to feed to Ollama to declare available tools
OLLAMA_TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read the contents of a local file to inspect code or configuration.",
            "parameters": {
                "type": "object",
                "properties": {
                    "filepath": {
                        "type": "string",
                        "description": "Absolute or relative path to the file."
                    }
                },
                "required": ["filepath"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Create a new file or overwrite an existing file with provided content.",
            "parameters": {
                "type": "object",
                "properties": {
                    "filepath": {
                        "type": "string",
                        "description": "Path to the file to write."
                    },
                    "content": {
                        "type": "string",
                        "description": "The full exact content to write to the file."
                    }
                },
                "required": ["filepath", "content"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "run_bash",
            "description": "Execute a bash shell command on the user's host system (e.g. gcc, ls, cat, python).",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The exact shell command to execute."
                    }
                },
                "required": ["command"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_vault",
            "description": "Search the local RAG knowledge base for technical reference, CVE details, or exploit techniques.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query_text": {
                        "type": "string",
                        "description": "The technical query to search for."
                    }
                },
                "required": ["query_text"]
            }
        }
    }
]
