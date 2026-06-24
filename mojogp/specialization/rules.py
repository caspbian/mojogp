"""Rule interfaces for specialization selection."""

from __future__ import annotations

from abc import ABC, abstractmethod

from .descriptor import WorkloadDescriptor
from .profile import SpecializationProfile


class SpecializationRule(ABC):
    """Abstract base class for narrow workload specialization rules."""

    @abstractmethod
    def matches(self, descriptor: WorkloadDescriptor) -> bool:
        raise NotImplementedError

    @abstractmethod
    def build_profile(self, descriptor: WorkloadDescriptor) -> SpecializationProfile:
        raise NotImplementedError

    def priority(self) -> int:
        return 0
