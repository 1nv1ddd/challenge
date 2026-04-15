from __future__ import annotations

from pathlib import Path

from ..providers import AIProvider

from .context import AgentContextMixin
from .facts_wm import AgentFactsMixin
from .memory_branches import AgentMemoryBranchesMixin
from .normalize import AgentStateMixin
from .prompts import AgentPromptsMixin
from .provider_utils import AgentProviderUtilsMixin
from .rag_context import AgentRagMixin
from .streaming import AgentStreamingMixin
from .task_fsm import AgentTaskFsmMixin


class SimpleChatAgent(
    AgentStreamingMixin,
    AgentContextMixin,
    AgentRagMixin,
    AgentPromptsMixin,
    AgentFactsMixin,
    AgentProviderUtilsMixin,
    AgentTaskFsmMixin,
    AgentMemoryBranchesMixin,
    AgentStateMixin,
):
    """Encapsulates chat request/response logic for LLM providers."""

    def __init__(
        self,
        providers: dict[str, AIProvider],
        memory_path: str | Path = "data/agent_memory.json",
    ):
        self.providers = providers
        self.memory_path = Path(memory_path)
        self.state_by_conversation: dict[str, dict] = {}
        self.global_memory: dict = {
            "long_term": {},
            "profiles": {
                "default": {
                    "name": "Default",
                    "style": "",
                    "format": "",
                    "constraints": "",
                }
            },
            "default_profile_id": "default",
        }
        self._load_history()

    def list_models(self) -> dict[str, list[dict]]:
        return {name: prov.models for name, prov in self.providers.items()}
