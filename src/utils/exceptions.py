"""
Custom exception classes for Spotify Music Discovery application.
"""

class SpotifyMusicDiscoveryError(Exception):
    """Base exception for the application."""
    pass

class SpotifyAPIError(SpotifyMusicDiscoveryError):
    """Raised when Spotify API calls fail."""
    pass

class AuthenticationError(SpotifyMusicDiscoveryError):
    """Raised when authentication fails."""
    pass

class AlgorithmError(SpotifyMusicDiscoveryError):
    """Raised when algorithm operations fail."""
    pass

class DataError(SpotifyMusicDiscoveryError):
    """Raised when data operations fail."""
    pass

class CacheError(SpotifyMusicDiscoveryError):
    """Raised when cache operations fail."""
    pass