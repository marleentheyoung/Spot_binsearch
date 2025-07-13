"""
Simplified Spotify API client without audio features.
Uses genres, popularity, release year, and other metadata for music discovery.
"""

import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import streamlit as st
import logging
import random
import time
from typing import List, Dict, Optional, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class Track:
    """Track data structure using basic Spotify metadata."""
    id: str
    name: str
    artist: str
    album: str
    album_art_url: str
    preview_url: Optional[str]
    popularity: int
    year: Optional[int]
    genres: List[str]  # Artist genres
    duration_ms: int
    explicit: bool

class SimplifiedSpotifyClient:
    """
    Simplified Spotify client that works with basic metadata only.
    No audio features required - uses genres, popularity, year, etc.
    """
    
    def __init__(self):
        self.sp: Optional[spotipy.Spotify] = None
        self._authenticated = False
        self._setup_spotify()
        
    def _setup_spotify(self):
        """Setup Spotify client with credentials."""
        try:
            # Get credentials
            client_id = None
            client_secret = None
            
            # Try Streamlit secrets first
            try:
                if hasattr(st, 'secrets'):
                    client_id = st.secrets.get('SPOTIFY_CLIENT_ID')
                    client_secret = st.secrets.get('SPOTIFY_CLIENT_SECRET')
            except:
                pass
            
            # Fallback to environment variables
            if not client_id or not client_secret:
                import os
                client_id = os.getenv('SPOTIFY_CLIENT_ID')
                client_secret = os.getenv('SPOTIFY_CLIENT_SECRET')
            
            if not client_id or not client_secret:
                raise ValueError("Spotify credentials not found")
            
            # Setup client
            client_credentials_manager = SpotifyClientCredentials(
                client_id=client_id,
                client_secret=client_secret
            )
            
            self.sp = spotipy.Spotify(
                client_credentials_manager=client_credentials_manager,
                requests_timeout=10,
                retries=3
            )
            
            # Test authentication
            self.sp.search(q="test", type="track", limit=1)
            self._authenticated = True
            logger.info("Spotify authentication successful")
                
        except Exception as e:
            logger.error(f"Spotify setup failed: {e}")
            raise
    
    def is_authenticated(self) -> bool:
        """Check if client is authenticated."""
        return self._authenticated and self.sp is not None
    
    def search_tracks(self, query: str, limit: int = 20, offset: int = 0) -> List[Track]:
        """Search for tracks using basic metadata."""
        try:
            results = self.sp.search(q=query, type='track', limit=limit, offset=offset)
            tracks = []
            
            for item in results['tracks']['items']:
                track = self._convert_to_track(item)
                if track:
                    tracks.append(track)
            
            logger.info(f"Found {len(tracks)} tracks for query: {query}")
            return tracks
            
        except Exception as e:
            logger.error(f"Error searching tracks: {e}")
            return []
    
    def _convert_to_track(self, track_data: Dict) -> Optional[Track]:
        """Convert Spotify API track data to Track object."""
        try:
            # Basic track info
            track_id = track_data['id']
            name = track_data['name']
            artists = track_data.get('artists', [])
            artist = artists[0]['name'] if artists else "Unknown Artist"
            album = track_data.get('album', {}).get('name', 'Unknown Album')
            popularity = track_data.get('popularity', 50)
            preview_url = track_data.get('preview_url')
            duration_ms = track_data.get('duration_ms', 0)
            explicit = track_data.get('explicit', False)
            
            # Album art
            album_art_url = ""
            album_images = track_data.get('album', {}).get('images', [])
            if album_images:
                album_art_url = album_images[0]['url']
            
            # Extract year
            year = None
            release_date = track_data.get('album', {}).get('release_date', '')
            if release_date:
                try:
                    year = int(release_date.split('-')[0])
                except:
                    pass
            
            # Get artist genres (requires additional API call)
            genres = []
            if artists:
                try:
                    artist_info = self.sp.artist(artists[0]['id'])
                    genres = artist_info.get('genres', [])
                except:
                    pass
            
            return Track(
                id=track_id,
                name=name,
                artist=artist,
                album=album,
                album_art_url=album_art_url,
                preview_url=preview_url,
                popularity=popularity,
                year=year,
                genres=genres,
                duration_ms=duration_ms,
                explicit=explicit
            )
            
        except Exception as e:
            logger.warning(f"Error converting track: {e}")
            return None
    
    def get_diverse_tracks(self, limit: int = 100) -> List[Track]:
        """Get diverse tracks across different genres and time periods."""
        all_tracks = []
        
        # Search queries for diversity
        search_queries = [
            "year:2020-2024",  # Recent hits
            "year:2010-2019",  # 2010s music
            "year:2000-2009",  # 2000s music  
            "year:1990-1999",  # 90s classics
            "genre:pop",
            "genre:rock", 
            "genre:hip-hop",
            "genre:electronic",
            "genre:indie",
            "genre:r&b",
            "genre:country",
            "genre:jazz",
            "acoustic",
            "instrumental",
            "dance",
            "chill"
        ]
        
        tracks_per_query = max(1, limit // len(search_queries))
        
        for query in search_queries:
            try:
                tracks = self.search_tracks(query, limit=tracks_per_query)
                all_tracks.extend(tracks)
                time.sleep(0.1)  # Rate limiting
            except Exception as e:
                logger.warning(f"Failed to search for '{query}': {e}")
                continue
        
        # Remove duplicates and shuffle
        seen_ids = set()
        unique_tracks = []
        for track in all_tracks:
            if track.id not in seen_ids:
                seen_ids.add(track.id)
                unique_tracks.append(track)
        
        random.shuffle(unique_tracks)
        return unique_tracks[:limit]
    
    def get_popular_tracks_by_genre(self, genre: str, limit: int = 20) -> List[Track]:
        """Get popular tracks for a specific genre."""
        query = f"genre:{genre}"
        return self.search_tracks(query, limit=limit)
    
    def get_tracks_by_year_range(self, start_year: int, end_year: int, limit: int = 20) -> List[Track]:
        """Get tracks from a specific year range."""
        query = f"year:{start_year}-{end_year}"
        return self.search_tracks(query, limit=limit)


# Global instance
_client_instance = None

def get_spotify_client() -> SimplifiedSpotifyClient:
    """Get singleton client instance."""
    global _client_instance
    
    if _client_instance is None:
        _client_instance = SimplifiedSpotifyClient()
    
    return _client_instance

def reset_spotify_client():
    """Reset the client instance."""
    global _client_instance
    _client_instance = None