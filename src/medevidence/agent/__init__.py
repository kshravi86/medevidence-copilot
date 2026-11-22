"""Week 3 agentic layer: safety gate -> planner -> executor (tools) -> critic, with persistent sessions and tracing."""
from .agent import Agent
from .state import SessionStore


def build_agent(rerank=True, sessions=True, **kw):
    """Construct the production agent (shared Qwen3-8B endpoint + local index)."""
    from ..llm import LLM
    from .tools import Tools
    return Agent(LLM(), Tools(rerank=rerank), SessionStore() if sessions else None, **kw)
