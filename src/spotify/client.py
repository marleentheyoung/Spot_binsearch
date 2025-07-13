"""
Spotify API client wrapper for music discovery application.
Handles authentication, API calls, and data fetching with caching support.
"""

import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import streamlit as st
import logging
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass
import time
from tenacity import retry, stop_after_attempt, wait_exponential
import requests
from ..utils.config import get_spotify_config, get_cache_config
from ..utils.exceptions import SpotifyAPIError, AuthenticationError
import pickle
import hashlib
from pathlib import Path

logger = logging.getLogger(__name__)

@dataclass
class Track:
    """Simplified track data structure."""
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

class SpotifyClient:
    """Enhanced Spotify client with caching and error handling."""
    
    def __init__(self):
        self.config = get_spotify_config()
        self.cache_config = get_cache_config()
        self.sp: Optional[spotipy.Spotify] = None
        self._authenticated = False
        self._cache_dir = self.cache_config.cache_dir
        self._setup_cache_directory()
        self._authenticate()
    
    def _setup_cache_directory(self):
        """Ensure cache directory exists."""
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Cache directory ready: {self._cache_dir}")
    
    def _authenticate(self):
        """Authenticate with Spotify API."""
        if not self.config:
            raise AuthenticationError("Spotify configuration not found")
        
        try:
            client_credentials_manager = SpotifyClientCredentials(
                client_id=self.config.client_id,
                client_secret=self.config.client_secret
            )
            
            self.sp = spotipy.Spotify(
                client_credentials_manager=client_credentials_manager,
                requests_timeout=10,
                retries=3
            )
            
            # Test authentication with a simple API call
            self.sp.search(q="test", type="track", limit=1)
            self._authenticated = True
            logger.info("Spotify authentication successful")
            
        except Exception as e:
            logger.error(f"Spotify authentication failed: {e}")
            raise AuthenticationError(f"Failed to authenticate with Spotify: {e}")
    
    def is_authenticated(self) -> bool:
        """Check if client is properly authenticated."""
        return self._authenticated and self.sp is not None
    
    def _get_cache_key(self, operation: str, **kwargs) -> str:
        """Generate cache key for operations."""
        key_data = f"{operation}_{kwargs}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def _get_from_cache(self, cache_key: str) -> Optional[Any]:
        """Retrieve data from cache if available and valid."""
        if not self.cache_config.enable_cache:
            return None
        
        cache_file = self._cache_dir / f"{cache_key}.pkl"
        
        try:
            if cache_file.exists():
                # Check if cache is still valid
                file_age = time.time() - cache_file.stat().st_mtime
                if file_age < self.cache_config.cache_ttl_seconds:
                    with open(cache_file, 'rb') as f:
                        data = pickle.load(f)
                        logger.debug(f"Cache hit: {cache_key}")
                        return data
                else:
                    # Cache expired, remove file
                    cache_file.unlink()
                    logger.debug(f"Cache expired: {cache_key}")
        except Exception as e:
            logger.warning(f"Error reading cache {cache_key}: {e}")
        
        return None
    
    def _save_to_cache(self, cache_key: str, data: Any):
        """Save data to cache."""
        if not self.cache_config.enable_cache:
            return
        
        cache_file = self._cache_dir / f"{cache_key}.pkl"
        
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(data, f)
                logger.debug(f"Cache saved: {cache_key}")
        except Exception as e:
            logger.warning(f"Error saving cache {cache_key}: {e}")
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def _api_call(self, func, *args, **kwargs):
        """Make API call with retry logic."""
        try:
            return func(*args, **kwargs)
        except spotipy.SpotifyException as e:
            if e.http_status == 429:  # Rate limited
                logger.warning("Rate limited, retrying...")
                raise
            elif e.http_status == 401:  # Unauthorized
                logger.error("Authentication error")
                raise AuthenticationError("Spotify authentication expired")
            else:
                logger.error(f"Spotify API error: {e}")
                raise SpotifyAPIError(f"API call failed: {e}")
        except Exception as e:
            logger.error(f"Unexpected error in API call: {e}")
            raise SpotifyAPIError(f"Unexpected error: {e}")
    
    def search_tracks(self, query: str, limit: int = 20, offset: int = 0) -> List[Track]:
        """Search for tracks with caching."""
        cache_key = self._get_cache_key("search", query=query, limit=limit, offset=offset)
        
        # Check cache first
        cached_result = self._get_from_cache(cache_key)
        if cached_result:
            return cached_result
        
        try:
            results = self._api_call(
                self.sp.search,
                q=query,
                type='track',
                limit=limit,
                offset=offset
            )
            
            tracks = []
            for item in results['tracks']['items']:
                track = self._convert_to_track(item)
                if track:
                    tracks.append(track)
            
            # Cache the result
            self._save_to_cache(cache_key, tracks)
            
            logger.info(f"Found {len(tracks)} tracks for query: {query}")
            return tracks
            
        except Exception as e:
            logger.error(f"Error searching tracks: {e}")
            raise SpotifyAPIError(f"Search failed: {e}")
    
    def get_track_features(self, track_id: str) -> Optional[Dict[str, float]]:
        """Get audio features for a track with caching."""
        cache_key = self._get_cache_key("features", track_id=track_id)
        
        # Check cache first
        cached_result = self._get_from_cache(cache_key)
        if cached_result:
            return cached_result
        
        try:
            features = self._api_call(self.sp.audio_features, track_id)
            
            if features and features[0]:
                feature_data = features[0]
                # Extract only the features we need
                relevant_features = {
                    'danceability': feature_data['danceability'],
                    'energy': feature_data['energy'],
                    'valence': feature_data['valence'],
                    'acousticness': feature_data['acousticness'],
                    'instrumentalness': feature_data['instrumentalness'],
                    'speechiness': feature_data['speechiness'],
                    'tempo': feature_data['tempo'],
                    'loudness': feature_data['loudness'],
                    'mode': float(feature_data['mode']),
                    'key': float(feature_data['key'])
                }
                
                # Cache the result
                self._save_to_cache(cache_key, relevant_features)
                
                return relevant_features
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting track features for {track_id}: {e}")
            return None
    
    def get_tracks_with_features(self, track_ids: List[str]) -> List[Track]:
        """Get tracks with their audio features in batch."""
        tracks = []
        
        # Get track details in batches of 50 (Spotify API limit)
        batch_size = 50
        for i in range(0, len(track_ids), batch_size):
            batch_ids = track_ids[i:i + batch_size]
            
            try:
                # Get track details
                track_results = self._api_call(self.sp.tracks, batch_ids)
                
                # Get audio features
                features_results = self._api_call(self.sp.audio_features, batch_ids)
                
                # Combine track info with features
                for track_info, features in zip(track_results['tracks'], features_results):
                    if track_info and features:
                        track = self._convert_to_track(track_info, features)
                        if track:
                            tracks.append(track)
                
            except Exception as e:
                logger.error(f"Error getting batch tracks: {e}")
                continue
        
        logger.info(f"Retrieved {len(tracks)} tracks with features")
        return tracks
    
    def get_recommendations(self, seed_tracks: List[str] = None, 
                          seed_artists: List[str] = None,
                          seed_genres: List[str] = None,
                          target_features: Dict[str, float] = None,
                          limit: int = 20) -> List[Track]:
        """Get recommendations based on seeds and target features."""
        cache_key = self._get_cache_key(
            "recommendations",
            seed_tracks=seed_tracks,
            seed_artists=seed_artists, 
            seed_genres=seed_genres,
            target_features=target_features,
            limit=limit
        )
        
        # Check cache first
        cached_result = self._get_from_cache(cache_key)
        if cached_result:
            return cached_result
        
        try:
            kwargs = {'limit': limit}
            
            if seed_tracks:
                kwargs['seed_tracks'] = seed_tracks[:5]  # Max 5 seeds
            if seed_artists:
                kwargs['seed_artists'] = seed_artists[:5]
            if seed_genres:
                kwargs['seed_genres'] = seed_genres[:5]
            
            # Add target features if provided
            if target_features:
                for feature, value in target_features.items():
                    kwargs[f'target_{feature}'] = value
            
            results = self._api_call(self.sp.recommendations, **kwargs)
            
            tracks = []
            for item in results['tracks']:
                track = self._convert_to_track(item)
                if track:
                    tracks.append(track)
            
            # Cache the result
            self._save_to_cache(cache_key, tracks)
            
            logger.info(f"Got {len(tracks)} recommendations")
            return tracks
            
        except Exception as e:
            logger.error(f"Error getting recommendations: {e}")
            raise SpotifyAPIError(f"Recommendations failed: {e}")
    
    def get_popular_tracks_by_genre(self, genre: str, limit: int = 50) -> List[Track]:
        """Get popular tracks for a specific genre."""
        # Use search with genre filter
        query = f"genre:{genre}"
        return self.search_tracks(query, limit=limit)
    
    def get_diverse_seed_tracks(self, genres: List[str], tracks_per_genre: int = 20) -> List[Track]:
        """Get diverse tracks across multiple genres for seeding."""
        all_tracks = []
        
        for genre in genres:
            try:
                genre_tracks = self.get_popular_tracks_by_genre(genre, limit=tracks_per_genre)
                all_tracks.extend(genre_tracks)
                logger.info(f"Added {len(genre_tracks)} tracks for genre: {genre}")
            except Exception as e:
                logger.warning(f"Failed to get tracks for genre {genre}: {e}")
                continue
        
        logger.info(f"Retrieved {len(all_tracks)} diverse seed tracks")
        return all_tracks
    
    def _convert_to_track(self, track_data: Dict, features_data: Dict = None) -> Optional[Track]:
        """Convert Spotify API track data to our Track object."""
        try:
            # Get album art URL
            album_art_url = ""
            if track_data.get('album', {}).get('images'):
                # Get medium-sized image (usually index 1)
                images = track_data['album']['images']
                album_art_url = images[1]['url'] if len(images) > 1 else images[0]['url']
            
            # Extract artist name(s)
            artists = track_data.get('artists', [])
            artist_name = artists[0]['name'] if artists else "Unknown Artist"
            
            # Extract features if provided, otherwise None
            features = {}
            if features_data:
                features = {
                    'danceability': features_data['danceability'],
                    'energy': features_data['energy'],
                    'valence': features_data['valence'],
                    'acousticness': features_data['acousticness'],
                    'instrumentalness': features_data['instrumentalness'],
                    'speechiness': features_data['speechiness'],
                    'tempo': features_data['tempo'],
                    'loudness': features_data['loudness'],
                    'mode': float(features_data['mode']),
                    'key': float(features_data['key'])
                }
            
            # Extract year from release date
            year = None
            release_date = track_data.get('album', {}).get('release_date', '')
            if release_date:
                year = int(release_date.split('-')[0]) if release_date else None
            
            track = Track(
                id=track_data['id'],
                name=track_data['name'],
                artist=artist_name,
                album=track_data.get('album', {}).get('name', 'Unknown Album'),
                album_art_url=album_art_url,
                preview_url=track_data.get('preview_url'),
                popularity=track_data.get('popularity', 0),
                features=features,
                year=year
            )
            
            return track
            
        except Exception as e:
            logger.warning(f"Error converting track data: {e}")
            return None
    
    def clear_cache(self):
        """Clear all cached data."""
        try:
            for cache_file in self._cache_dir.glob("*.pkl"):
                cache_file.unlink()
            logger.info("Cache cleared successfully")
        except Exception as e:
            logger.error(f"Error clearing cache: {e}")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        try:
            cache_files = list(self._cache_dir.glob("*.pkl"))
            total_size = sum(f.stat().st_size for f in cache_files)
            
            return {
                'cached_items': len(cache_files),
                'total_size_mb': total_size / (1024 * 1024),
                'cache_enabled': self.cache_config.enable_cache,
                'cache_ttl_hours': self.cache_config.cache_ttl_seconds / 3600
            }
        except Exception as e:
            logger.error(f"Error getting cache stats: {e}")
            return {}


# Convenience function for getting client instance
_client_instance = None

def get_spotify_client() -> SpotifyClient:
    """Get singleton Spotify client instance."""
    global _client_instance
    
    if _client_instance is None:
        _client_instance = SpotifyClient()
    
    return _client_instance

def reset_spotify_client():
    """Reset the client instance (useful for testing or config changes)."""
    global _client_instance
    _client_instance = None