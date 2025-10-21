"""
ReAct Scratchpad Implementation

A scratchpad for tracking agent thoughts, observations, and discoveries
during the ReAct reasoning process. Provides structured storage and
formatted output for agent reasoning.
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
import json


@dataclass
class ScratchpadEntry:
    """Single entry in the scratchpad"""
    timestamp: datetime
    entry_type: str  # "thought", "observation", "discovery", "plan"
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __str__(self) -> str:
        time_str = self.timestamp.strftime("%H:%M:%S")
        return f"[{time_str}] {self.entry_type.upper()}: {self.content}"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "timestamp": self.timestamp.isoformat(),
            "entry_type": self.entry_type,
            "content": self.content,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ScratchpadEntry':
        """Create from dictionary"""
        return cls(
            timestamp=datetime.fromisoformat(data["timestamp"]),
            entry_type=data["entry_type"],
            content=data["content"],
            metadata=data.get("metadata", {})
        )


class Scratchpad:
    """
    ReAct agent scratchpad for tracking reasoning process
    
    Stores thoughts, observations, discoveries, and plans in a structured way
    that can be easily formatted for LLM consumption and human review.
    """
    
    def __init__(self, max_entries: int = 50):
        """
        Initialize scratchpad
        
        Args:
            max_entries: Maximum number of entries to keep (oldest removed first)
        """
        self.entries: List[ScratchpadEntry] = []
        self.max_entries = max_entries
        self.created_at = datetime.now()
    
    def add_thought(self, content: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Add a thought/reasoning entry"""
        self._add_entry("thought", content, metadata or {})
    
    def add_observation(self, content: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Add an observation from action results"""
        self._add_entry("observation", content, metadata or {})
    
    def add_discovery(self, content: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Add a significant discovery or insight"""
        self._add_entry("discovery", content, metadata or {})
    
    def add_plan(self, content: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Add a plan or strategy update"""
        self._add_entry("plan", content, metadata or {})
    
    def _add_entry(self, entry_type: str, content: str, metadata: Dict[str, Any]) -> None:
        """Internal method to add an entry"""
        entry = ScratchpadEntry(
            timestamp=datetime.now(),
            entry_type=entry_type,
            content=content,
            metadata=metadata
        )
        
        self.entries.append(entry)
        
        # Trim if we exceed max entries
        if len(self.entries) > self.max_entries:
            self.entries = self.entries[-self.max_entries:]
    
    def get_formatted_content(self, entry_types: Optional[List[str]] = None, 
                            last_n: Optional[int] = None) -> str:
        """
        Get formatted scratchpad content for LLM consumption
        
        Args:
            entry_types: Filter by entry types (e.g., ["thought", "observation"])
            last_n: Only include last N entries
            
        Returns:
            Formatted string suitable for LLM prompts
        """
        entries = self.entries
        
        # Filter by entry types if specified
        if entry_types:
            entries = [e for e in entries if e.entry_type in entry_types]
        
        # Limit to last N entries if specified
        if last_n:
            entries = entries[-last_n:]
        
        if not entries:
            return "No entries yet."
        
        # Format entries
        formatted_lines = []
        for entry in entries:
            formatted_lines.append(str(entry))
        
        return "\n".join(formatted_lines)
    
    def get_summary(self, max_length: int = 500) -> str:
        """
        Get a concise summary of the scratchpad content
        
        Args:
            max_length: Maximum length of summary
            
        Returns:
            Summary string
        """
        if not self.entries:
            return "Empty scratchpad - no reasoning history yet."
        
        # Count entries by type
        type_counts = {}
        for entry in self.entries:
            type_counts[entry.entry_type] = type_counts.get(entry.entry_type, 0) + 1
        
        # Get recent discoveries and key observations
        recent_entries = self.entries[-5:]  # Last 5 entries
        discoveries = [e for e in self.entries if e.entry_type == "discovery"]
        
        summary_parts = [
            f"Scratchpad Summary ({len(self.entries)} entries):"
        ]
        
        # Add type breakdown
        type_summary = ", ".join([f"{count} {type_}" for type_, count in type_counts.items()])
        summary_parts.append(f"Content: {type_summary}")
        
        # Add key discoveries
        if discoveries:
            latest_discovery = discoveries[-1]
            summary_parts.append(f"Latest discovery: {latest_discovery.content[:100]}...")
        
        # Add recent activity
        if recent_entries:
            summary_parts.append("Recent activity:")
            for entry in recent_entries[-3:]:  # Last 3 entries
                summary_parts.append(f"  • {entry.entry_type}: {entry.content[:80]}...")
        
        summary = "\n".join(summary_parts)
        
        # Truncate if too long
        if len(summary) > max_length:
            summary = summary[:max_length-3] + "..."
        
        return summary
    
    def get_thoughts_only(self, last_n: int = 5) -> str:
        """Get only the thinking/reasoning entries"""
        return self.get_formatted_content(entry_types=["thought"], last_n=last_n)
    
    def get_observations_only(self, last_n: int = 5) -> str:
        """Get only the observation entries"""
        return self.get_formatted_content(entry_types=["observation"], last_n=last_n)
    
    def get_discoveries(self) -> List[ScratchpadEntry]:
        """Get all discovery entries"""
        return [e for e in self.entries if e.entry_type == "discovery"]
    
    def has_content(self) -> bool:
        """Check if scratchpad has any entries"""
        return len(self.entries) > 0
    
    def clear(self) -> None:
        """Clear all entries"""
        self.entries.clear()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert scratchpad to dictionary for serialization"""
        return {
            "created_at": self.created_at.isoformat(),
            "max_entries": self.max_entries,
            "entries": [entry.to_dict() for entry in self.entries]
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Scratchpad':
        """Create scratchpad from dictionary"""
        scratchpad = cls(max_entries=data.get("max_entries", 50))
        scratchpad.created_at = datetime.fromisoformat(data["created_at"])
        
        for entry_data in data.get("entries", []):
            entry = ScratchpadEntry.from_dict(entry_data)
            scratchpad.entries.append(entry)
        
        return scratchpad
    
    def save_to_file(self, filepath: str) -> None:
        """Save scratchpad to JSON file"""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
    
    @classmethod
    def load_from_file(cls, filepath: str) -> 'Scratchpad':
        """Load scratchpad from JSON file"""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return cls.from_dict(data)
    
    def __len__(self) -> int:
        """Return number of entries"""
        return len(self.entries)
    
    def __str__(self) -> str:
        """String representation"""
        if not self.entries:
            return "Empty Scratchpad"
        
        return f"Scratchpad ({len(self.entries)} entries)\n{self.get_formatted_content()}"
