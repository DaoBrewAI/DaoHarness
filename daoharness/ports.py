"""Ports the core depends on: a workflow store, a model, membership, a notifier and, optionally, a decider."""
from __future__ import annotations

from contextlib import AbstractContextManager
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any, Mapping, Protocol, Sequence

if TYPE_CHECKING:
    from .decision import DecisionAnswer, DecisionItem

MODEL_OPERATIONS = frozenset({"compile", "step", "decide"})


@dataclass(frozen=True)
class ModelReceipt:
    run_id: str
    provider: str
    model: str
    output_digest: str
    # Token counts: measured when the provider reports them, else estimated from text length (tokens_estimated).
    # Left out of to_dict while unknown.
    input_tokens: int | None = None
    output_tokens: int | None = None
    tokens_estimated: bool | None = None

    def to_dict(self) -> dict:
        return {k: v for k, v in asdict(self).items() if v is not None}


class ModelError(RuntimeError):
    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


class ModelPort(Protocol):
    def generate(self, *, org_id: str, operation: str, skill_id: str, input_data: Mapping[str, Any],
                 instructions: str, evidence: Sequence[Mapping[str, Any]], schema: Mapping[str, Any]
                 ) -> tuple[dict, ModelReceipt]: ...


class DecisionPort(Protocol):
    """A decider you plug in instead of the model for finite-choice decisions. Raises
    decision.DecisionUnavailable when it cannot answer."""

    def decide(self, items: Sequence[DecisionItem], *, rubric: str, org_id: str) -> list[DecisionAnswer]: ...


class WorkflowStore(Protocol):
    def read(self, org_id: str, collection: str, key: str) -> dict | None: ...
    def write(self, org_id: str, collection: str, key: str, value: Mapping[str, Any]) -> None: ...
    def create(self, org_id: str, collection: str, key: str, value: Mapping[str, Any]) -> bool: ...
    def list(self, org_id: str, collection: str) -> list[dict]: ...
    def list_by(self, org_id: str, collection: str, **equals: str) -> list[dict]: ...   # top-level str fields
    def locked(self, org_id: str) -> AbstractContextManager: ...


class MembershipPort(Protocol):
    def resolve(self, org_id: str, approver_role: str) -> str | None: ...


class Notifier(Protocol):
    def approval_requested(self, org_id: str, approval: Mapping[str, Any]) -> None: ...
