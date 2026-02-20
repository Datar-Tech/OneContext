"""Event data models for grouping related commits."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Dict, Any
from enum import Enum


class EventSource(Enum):
    """How the event was generated."""

    USER = "user"  # Manually created via `aline watcher event generate`
    PRESET_DAY = "preset_day"  # Auto-generated daily aggregation
    PRESET_WEEK = "preset_week"  # Auto-generated weekly aggregation


@dataclass
class Event:
    """
    Represents a high-level event grouping multiple sessions.

    An Event is a semantic abstraction that groups related sessions
    together based on user selection or time-based presets.
    """

    id: str  # Unique event ID (UUID)
    title: str  # Human-readable event title
    description: Optional[str] = None  # Detailed description
    source: EventSource = EventSource.USER  # How the event was generated

    # Commit references
    commit_hashes: List[str] = field(default_factory=list)  # Full commit hashes

    # Temporal metadata
    start_timestamp: Optional[datetime] = None  # Earliest commit timestamp
    end_timestamp: Optional[datetime] = None  # Latest commit timestamp
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

    # Semantic metadata
    tags: List[str] = field(default_factory=list)  # User-defined tags
    primary_files: List[str] = field(default_factory=list)  # Main files involved
    session_ids: List[str] = field(default_factory=list)  # Related session IDs

    # Generation metadata
    auto_generated: bool = True  # Whether auto-generated or manual
    generation_method: Optional[str] = None  # e.g., "llm_clustering", "time_window"
    confidence_score: Optional[float] = None  # Clustering confidence (0-1)

    # UI metadata (optional, for shares)
    ui_metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert Event to JSON-serializable dict."""
        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "source": self.source.value,
            "commit_hashes": self.commit_hashes,
            "start_timestamp": self.start_timestamp.isoformat() if self.start_timestamp else None,
            "end_timestamp": self.end_timestamp.isoformat() if self.end_timestamp else None,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "tags": self.tags,
            "primary_files": self.primary_files,
            "session_ids": self.session_ids,
            "auto_generated": self.auto_generated,
            "generation_method": self.generation_method,
            "confidence_score": self.confidence_score,
            "ui_metadata": self.ui_metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Event":
        """Create Event from dict."""
        # Handle legacy data with event_type/status
        source_value = data.get("source") or data.get("event_type", "user")
        if source_value == "task":
            source_value = "user"  # Migrate legacy 'task' to 'user'

        return cls(
            id=data["id"],
            title=data["title"],
            description=data.get("description"),
            source=EventSource(source_value),
            commit_hashes=data.get("commit_hashes", []),
            start_timestamp=(
                datetime.fromisoformat(data["start_timestamp"])
                if data.get("start_timestamp")
                else None
            ),
            end_timestamp=(
                datetime.fromisoformat(data["end_timestamp"]) if data.get("end_timestamp") else None
            ),
            created_at=datetime.fromisoformat(data["created_at"]),
            updated_at=datetime.fromisoformat(data["updated_at"]),
            tags=data.get("tags", []),
            primary_files=data.get("primary_files", []),
            session_ids=data.get("session_ids", []),
            auto_generated=data.get("auto_generated", True),
            generation_method=data.get("generation_method"),
            confidence_score=data.get("confidence_score"),
            ui_metadata=data.get("ui_metadata"),
        )


@dataclass
class EventCollection:
    """Container for all events in a project."""

    version: int = 1  # Schema version for migration
    events: List[Event] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_event_by_id(self, event_id: str) -> Optional[Event]:
        """Retrieve event by ID."""
        return next((e for e in self.events if e.id == event_id), None)

    def get_events_by_source(self, source: EventSource) -> List[Event]:
        """Get all events from a specific source."""
        return [e for e in self.events if e.source == source]

    def to_dict(self) -> Dict[str, Any]:
        """Convert EventCollection to JSON-serializable dict."""
        return {
            "version": self.version,
            "metadata": self.metadata,
            "events": [e.to_dict() for e in self.events],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EventCollection":
        """Create EventCollection from dict."""
        return cls(
            version=data.get("version", 1),
            events=[Event.from_dict(e) for e in data.get("events", [])],
            metadata=data.get("metadata", {}),
        )
