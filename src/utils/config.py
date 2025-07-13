"""
Configuration management for Spotify Binary Search Music Discovery app.
Handles environment variables, Streamlit secrets, and application settings.
"""

import os
import streamlit as st
from typing import Optional, Dict, Any, List
import logging
from pydantic_settings import SettingsConfigDict
from pathlib import Path
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings

# Setup logging
logger = logging.getLogger(__name__)

class SpotifyConfig(BaseSettings):
    """Spotify API configuration settings."""
    
    client_id: str = Field(..., description="Spotify API Client ID")
    client_secret: str = Field(..., description="Spotify API Client Secret")
    redirect_uri: str = Field(
        default="http://localhost:8501", 
        description="OAuth redirect URI"
    )
    scope: str = Field(
        default="user-read-private user-read-email user-library-read user-top-read",
        description="Spotify API scopes"
    )
    
    model_config = SettingsConfigDict(
        env_prefix="SPOTIFY_",
        case_sensitive=False
    )


class AlgorithmConfig(BaseSettings):
    """Binary search algorithm configuration."""
    
    max_iterations: int = Field(default=8, description="Maximum search iterations")
    min_songs_threshold: int = Field(default=10, description="Minimum songs to stop search")
    initial_sample_size: int = Field(default=100, description="Sample size for distance calculation")
    convergence_threshold: float = Field(default=0.1, description="Distance threshold for convergence")
    
    # Audio feature weights for distance calculation
    feature_weights: Dict[str, float] = Field(
        default={
            'danceability': 1.0,
            'energy': 1.0,
            'valence': 1.2,        # Emotional valence weighted higher
            'acousticness': 1.1,
            'instrumentalness': 0.8,
            'speechiness': 0.9,
            'tempo': 0.5,          # Normalized and weighted lower
            'loudness': 0.6,
            'mode': 0.4,
            'key': 0.3
        }
    )
    
    model_config = SettingsConfigDict(
        env_prefix="ALGORITHM_",
        case_sensitive=False
    )


class CacheConfig(BaseSettings):
    """Caching configuration settings."""
    
    enable_cache: bool = Field(default=True, description="Enable disk caching")
    cache_dir: Path = Field(default=Path("data/cache"), description="Cache directory")
    cache_ttl_seconds: int = Field(default=3600, description="Cache TTL in seconds")
    max_cache_size_mb: int = Field(default=500, description="Maximum cache size in MB")
    
    model_config = SettingsConfigDict(
        env_prefix="CACHE_",
        case_sensitive=False
    )


class DataConfig(BaseSettings):
    """Data management configuration."""
    
    seed_songs_path: Path = Field(default=Path("data/seed_songs"), description="Seed songs directory")
    user_sessions_path: Path = Field(default=Path("data/user_sessions"), description="User sessions directory")
    min_seed_songs: int = Field(default=1000, description="Minimum number of seed songs required")
    genres_to_include: List[str] = Field(
        default=[
            "pop", "rock", "hip-hop", "jazz", "classical", "electronic", 
            "country", "r&b", "indie", "folk", "metal", "reggae", "blues", "funk"
        ],
        description="Music genres to include in seed data"
    )
    
    model_config = SettingsConfigDict(
        env_prefix="DATA_",
        case_sensitive=False
    )


class UIConfig(BaseSettings):
    """User interface configuration."""
    
    app_title: str = Field(default="🎵 Spotify Music Discovery", description="App title")
    page_icon: str = Field(default="🎵", description="Page icon")
    layout: str = Field(default="wide", description="Streamlit layout")
    
    # Theme colors (Spotify-inspired)
    primary_color: str = Field(default="#1DB954", description="Primary color (Spotify green)")
    background_color: str = Field(default="#FFFFFF", description="Background color")
    secondary_bg_color: str = Field(default="#F0F2F6", description="Secondary background")
    text_color: str = Field(default="#262730", description="Text color")
    
    # Audio settings
    audio_volume: float = Field(default=0.5, description="Default audio volume")
    audio_autoplay: bool = Field(default=False, description="Auto-play audio previews")
    
    model_config = SettingsConfigDict(
        env_prefix="UI_",
        case_sensitive=False
    )


class AppConfig:
    """Main application configuration manager."""
    
    def __init__(self):
        self.spotify: Optional[SpotifyConfig] = None
        self.algorithm = AlgorithmConfig()
        self.cache = CacheConfig()
        self.data = DataConfig()
        self.ui = UIConfig()
        self._setup_directories()
        self._load_spotify_config()
    
    def _setup_directories(self):
        """Create necessary directories if they don't exist."""
        directories = [
            self.cache.cache_dir,
            self.data.seed_songs_path,
            self.data.user_sessions_path,
            Path("assets/images"),
            Path("assets/styles"),
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            logger.info(f"Ensured directory exists: {directory}")
    
    def _load_spotify_config(self):
        """Load Spotify configuration from Streamlit secrets or environment."""
        try:
            # Try Streamlit secrets first (for deployed apps) - but only if Streamlit is actually running
            try:
                # Check if we're actually in a Streamlit context
                if hasattr(st, 'secrets') and hasattr(st.secrets, 'get'):
                    spotify_secrets = {
                        'client_id': st.secrets.get('SPOTIFY_CLIENT_ID'),
                        'client_secret': st.secrets.get('SPOTIFY_CLIENT_SECRET'),
                        'redirect_uri': st.secrets.get('SPOTIFY_REDIRECT_URI', 'http://localhost:8501')
                    }
                    
                    # Check if all required secrets are present
                    if all(spotify_secrets.values()):
                        self.spotify = SpotifyConfig(**spotify_secrets)
                        logger.info("Loaded Spotify config from Streamlit secrets")
                        return
            except Exception:
                # Streamlit secrets not available or not in Streamlit context, fall back to env vars
                pass
            
            # Fall back to environment variables
            self.spotify = SpotifyConfig()
            logger.info("Loaded Spotify config from environment variables")
            
        except Exception as e:
            logger.error(f"Failed to load Spotify configuration: {e}")
            self.spotify = None
    
    def is_spotify_configured(self) -> bool:
        """Check if Spotify API is properly configured."""
        return (
            self.spotify is not None and 
            self.spotify.client_id and 
            self.spotify.client_secret
        )
    
    def get_streamlit_config(self) -> Dict[str, Any]:
        """Get configuration for Streamlit page setup."""
        return {
            'page_title': self.ui.app_title,
            'page_icon': self.ui.page_icon,
            'layout': self.ui.layout,
            'initial_sidebar_state': 'expanded'
        }
    
    def get_theme_config(self) -> Dict[str, str]:
        """Get theme configuration for custom CSS."""
        return {
            'primaryColor': self.ui.primary_color,
            'backgroundColor': self.ui.background_color,
            'secondaryBackgroundColor': self.ui.secondary_bg_color,
            'textColor': self.ui.text_color
        }
    
    def validate_configuration(self) -> Dict[str, bool]:
        """Validate all configuration components."""
        validation_results = {
            'spotify_configured': self.is_spotify_configured(),
            'cache_directory_writable': self._is_directory_writable(self.cache.cache_dir),
            'seed_data_directory_exists': self.data.seed_songs_path.exists(),
            'algorithm_weights_valid': self._validate_algorithm_weights(),
        }
        
        return validation_results
    
    def _is_directory_writable(self, directory: Path) -> bool:
        """Check if a directory is writable."""
        try:
            test_file = directory / '.write_test'
            test_file.touch()
            test_file.unlink()
            return True
        except Exception:
            return False
    
    def _validate_algorithm_weights(self) -> bool:
        """Validate algorithm feature weights."""
        required_features = {
            'danceability', 'energy', 'valence', 'acousticness', 
            'instrumentalness', 'speechiness', 'tempo', 'loudness', 'mode', 'key'
        }
        
        weights = set(self.algorithm.feature_weights.keys())
        return required_features.issubset(weights)
    
    def get_debug_info(self) -> Dict[str, Any]:
        """Get debug information about configuration."""
        return {
            'spotify_configured': self.is_spotify_configured(),
            'cache_enabled': self.cache.enable_cache,
            'cache_directory': str(self.cache.cache_dir),
            'max_iterations': self.algorithm.max_iterations,
            'seed_songs_path': str(self.data.seed_songs_path),
            'validation_results': self.validate_configuration()
        }


# Global configuration instance
config = AppConfig()


def get_config() -> AppConfig:
    """Get the global configuration instance."""
    return config


def reload_config():
    """Reload configuration (useful for development)."""
    global config
    config = AppConfig()
    logger.info("Configuration reloaded")


# Utility functions for common configuration access
def get_spotify_config() -> Optional[SpotifyConfig]:
    """Get Spotify configuration."""
    return config.spotify


def get_algorithm_config() -> AlgorithmConfig:
    """Get algorithm configuration."""
    return config.algorithm


def get_cache_config() -> CacheConfig:
    """Get cache configuration."""
    return config.cache


def get_data_config() -> DataConfig:
    """Get data configuration."""
    return config.data


def get_ui_config() -> UIConfig:
    """Get UI configuration."""
    return config.ui