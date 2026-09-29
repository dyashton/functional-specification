"""Generation stack: DesignObjective → FAN → FunctionalSpecification → Generator."""

from functionalspec.gen.fan import FunctionalAbstractionNetwork
from functionalspec.gen.fan_context import ContextFAN
from functionalspec.gen.generate import GenerationResult, generate_from_spec
from functionalspec.gen.objective import DesignObjective
from functionalspec.gen.spec import FlatSPayload, FunctionalSpecification

__all__ = [
    "ContextFAN",
    "DesignObjective",
    "FlatSPayload",
    "FunctionalAbstractionNetwork",
    "FunctionalSpecification",
    "GenerationResult",
    "generate_from_spec",
]
