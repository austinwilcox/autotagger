"""Provider interface: turn a query into normalized `Candidate` objects."""

from __future__ import annotations

from abc import ABC, abstractmethod

from ..models import AudioFile, Candidate


class Provider(ABC):
    name: str

    @abstractmethod
    def search(self, query: str, limit: int = 20) -> list[Candidate]:
        """Free-text search. Must never raise for ordinary failures — return []."""

    def search_for_file(self, af: AudioFile, limit: int = 20) -> list[Candidate]:
        """Try the file's query terms in order, accumulating deduplicated results."""
        seen: set[str] = set()
        out: list[Candidate] = []
        for term in af.search_terms():
            for cand in self.search(term, limit=limit):
                key = f"{self.name}:{cand.source_id}"
                if key in seen:
                    continue
                seen.add(key)
                out.append(cand)
            if len(out) >= limit:
                break
        return out

    def enrich(self, candidate: Candidate) -> Candidate:
        """Optionally fill extra fields on a chosen candidate. Default: no-op."""
        return candidate
