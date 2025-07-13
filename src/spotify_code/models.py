"""
Data models for Spotify API responses and music data structures.
"""

from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any
import json

@dataclass
class AudioFeatures:
    """Audio features for a track."""
    danceability: float
    energy: float
    valence: float
    acousticness: float
    instrumentalness: float
    speechiness: float
    tempo: float
    loudness: float
    mode: float
    key: float
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, float]) -> 'AudioFeatures':
        """Create from dictionary."""
        return cls(**data)

@dataclass
class Track:
    """Represents a music track with metadata and features."""
    id: str
    name: str
    artist: str
    album: str
    album_art_url: str
    preview_url: Optional[str]
    popularity: int
    features: Dict[str, float]
    genre: Optional[str] = None
    year: Optional[int] = None
    duration_ms: Optional[int] = None
    
    @property
    def audio_features(self) -> AudioFeatures:
        """Get audio features as AudioFeatures object."""
        return AudioFeatures.from_dict(self.features)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Track':
        """Create Track from dictionary."""
        return cls(**data)
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict())
    
    @classmethod
    def from_json(cls, json_str: str) -> 'Track':
        """Create Track from JSON string."""
        data = json.loads(json_str)
        return cls.from_dict(data)
    
    def __str__(self) -> str:
        return f"{self.name} by {self.artist}"
    
    def __repr__(self) -> str:
        return f"Track(id='{self.id}', name='{self.name}', artist='{self.artist}')"

@dataclass
class Artist:
    """Represents an artist."""
    id: str
    name: str
    genres: List[str]
    popularity: int
    image_url: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

@dataclass
class Album:
    """Represents an album."""
    id: str
    name: str
    artist: str
    release_date: str
    image_url: Optional[str]
    total_tracks: int
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

@dataclass
class Playlist:
    """Represents a playlist."""
    id: str
    name: str
    description: str
    tracks: List[Track]
    owner: str
    public: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'id': self.id,
            'name': self.name,
            'description': self.description,
            'tracks': [track.to_dict() for track in self.tracks],
            'owner': self.owner,
            'public': self.public
        }