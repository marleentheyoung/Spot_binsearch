"""
Distance calculation functions for Spotify Binary Search Music Discovery.
Handles feature normalization, weighted distance calculations, and similarity metrics.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from enum import Enum
import logging
from functools import lru_cache

# Setup logging
logger = logging.getLogger(__name__)

class DistanceMetric(Enum):
    """Available distance calculation methods."""
    EUCLIDEAN = "euclidean"
    MANHATTAN = "manhattan"
    COSINE = "cosine"
    WEIGHTED_EUCLIDEAN = "weighted_euclidean"
    MAHALANOBIS = "mahalanobis"


@dataclass
class AudioFeatures:
    """Structured representation of Spotify audio features."""
    danceability: float
    energy: float
    valence: float
    acousticness: float
    instrumentalness: float
    speechiness: float
    tempo: float
    loudness: float
    mode: int
    key: int
    
    @classmethod
    def from_spotify_response(cls, spotify_features: Dict) -> 'AudioFeatures':
        """Create AudioFeatures from Spotify API response."""
        return cls(
            danceability=spotify_features.get('danceability', 0.0),
            energy=spotify_features.get('energy', 0.0),
            valence=spotify_features.get('valence', 0.0),
            acousticness=spotify_features.get('acousticness', 0.0),
            instrumentalness=spotify_features.get('instrumentalness', 0.0),
            speechiness=spotify_features.get('speechiness', 0.0),
            tempo=spotify_features.get('tempo', 120.0),
            loudness=spotify_features.get('loudness', -10.0),
            mode=spotify_features.get('mode', 1),
            key=spotify_features.get('key', 0)
        )
    
    def to_vector(self, normalize: bool = True) -> np.ndarray:
        """Convert to normalized feature vector."""
        if normalize:
            return normalize_features(self.__dict__)
        return np.array(list(self.__dict__.values()))


class FeatureNormalizer:
    """Handles normalization of different audio features to 0-1 scale."""
    
    # Spotify API feature ranges
    FEATURE_RANGES = {
        'danceability': (0.0, 1.0),
        'energy': (0.0, 1.0),
        'valence': (0.0, 1.0),
        'acousticness': (0.0, 1.0),
        'instrumentalness': (0.0, 1.0),
        'speechiness': (0.0, 1.0),
        'tempo': (0.0, 250.0),        # Extended range for edge cases
        'loudness': (-60.0, 5.0),     # Extended range for edge cases
        'mode': (0, 1),               # Categorical: 0=minor, 1=major
        'key': (0, 11)                # Categorical: 0=C, 1=C#, etc.
    }
    
    @classmethod
    def normalize_feature(cls, feature_name: str, value: Union[float, int]) -> float:
        """Normalize a single feature to 0-1 scale."""
        if feature_name not in cls.FEATURE_RANGES:
            logger.warning(f"Unknown feature: {feature_name}")
            return float(value)
        
        min_val, max_val = cls.FEATURE_RANGES[feature_name]
        
        # Clamp value to expected range
        clamped_value = max(min_val, min(max_val, value))
        
        # Handle categorical features specially
        if feature_name in ['mode', 'key']:
            return float(clamped_value) / max_val
        
        # Linear normalization for continuous features
        if max_val == min_val:
            return 0.5  # Default for degenerate ranges
        
        normalized = (clamped_value - min_val) / (max_val - min_val)
        return max(0.0, min(1.0, normalized))  # Ensure [0,1] bounds
    
    @classmethod
    def normalize_features_dict(cls, features: Dict[str, Union[float, int]]) -> Dict[str, float]:
        """Normalize all features in a dictionary."""
        normalized = {}
        for feature_name, value in features.items():
            if feature_name in cls.FEATURE_RANGES:
                normalized[feature_name] = cls.normalize_feature(feature_name, value)
            else:
                # Pass through unknown features unchanged
                normalized[feature_name] = float(value)
        return normalized


def normalize_features(features: Dict[str, Union[float, int]]) -> np.ndarray:
    """
    Normalize audio features to 0-1 scale for distance calculation.
    
    Args:
        features: Dictionary of raw audio features
        
    Returns:
        Normalized feature vector as numpy array
    """
    normalized_dict = FeatureNormalizer.normalize_features_dict(features)
    
    # Ensure consistent ordering
    feature_order = [
        'danceability', 'energy', 'valence', 'acousticness',
        'instrumentalness', 'speechiness', 'tempo', 'loudness', 'mode', 'key'
    ]
    
    vector = []
    for feature in feature_order:
        if feature in normalized_dict:
            vector.append(normalized_dict[feature])
        else:
            logger.warning(f"Missing feature {feature}, using default value 0.5")
            vector.append(0.5)
    
    return np.array(vector)


class WeightedDistanceCalculator:
    """Calculates weighted distances between songs based on audio features."""
    
    def __init__(self, feature_weights: Optional[Dict[str, float]] = None):
        """
        Initialize with custom feature weights.
        
        Args:
            feature_weights: Optional custom weights for features
        """
        self.feature_weights = feature_weights or self._get_default_weights()
        self._validate_weights()
    
    def _get_default_weights(self) -> Dict[str, float]:
        """Get default feature weights optimized for music discovery."""
        return {
            'danceability': 1.0,      # Core rhythmic preference
            'energy': 1.0,            # Energy level preference
            'valence': 1.2,           # Emotional tone (weighted higher)
            'acousticness': 1.1,      # Important for genre distinction
            'instrumentalness': 0.8,   # Less critical for most users
            'speechiness': 0.9,       # Moderate importance
            'tempo': 0.5,             # Often less important than feel
            'loudness': 0.6,          # Secondary characteristic
            'mode': 0.4,              # Minor vs major, subtle preference
            'key': 0.3                # Least important for most users
        }
    
    def _validate_weights(self):
        """Validate that weights are properly configured."""
        required_features = {
            'danceability', 'energy', 'valence', 'acousticness',
            'instrumentalness', 'speechiness', 'tempo', 'loudness', 'mode', 'key'
        }
        
        weight_features = set(self.feature_weights.keys())
        if not required_features.issubset(weight_features):
            missing = required_features - weight_features
            raise ValueError(f"Missing weights for features: {missing}")
        
        # Ensure all weights are positive
        for feature, weight in self.feature_weights.items():
            if weight < 0:
                raise ValueError(f"Negative weight for {feature}: {weight}")
    
    def calculate_distance(
        self, 
        features1: Union[Dict, AudioFeatures], 
        features2: Union[Dict, AudioFeatures],
        metric: DistanceMetric = DistanceMetric.WEIGHTED_EUCLIDEAN
    ) -> float:
        """
        Calculate weighted distance between two songs.
        
        Args:
            features1: First song's audio features
            features2: Second song's audio features
            metric: Distance metric to use
            
        Returns:
            Distance value (0 = identical, higher = more different)
        """
        # Convert to dictionaries if needed
        if isinstance(features1, AudioFeatures):
            features1 = features1.__dict__
        if isinstance(features2, AudioFeatures):
            features2 = features2.__dict__
        
        # Normalize features
        norm_features1 = normalize_features(features1)
        norm_features2 = normalize_features(features2)
        
        if metric == DistanceMetric.WEIGHTED_EUCLIDEAN:
            return self._weighted_euclidean_distance(norm_features1, norm_features2)
        elif metric == DistanceMetric.EUCLIDEAN:
            return self._euclidean_distance(norm_features1, norm_features2)
        elif metric == DistanceMetric.MANHATTAN:
            return self._manhattan_distance(norm_features1, norm_features2)
        elif metric == DistanceMetric.COSINE:
            return self._cosine_distance(norm_features1, norm_features2)
        else:
            raise ValueError(f"Unsupported distance metric: {metric}")
    
    def _weighted_euclidean_distance(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate weighted Euclidean distance."""
        feature_order = [
            'danceability', 'energy', 'valence', 'acousticness',
            'instrumentalness', 'speechiness', 'tempo', 'loudness', 'mode', 'key'
        ]
        
        weights = np.array([self.feature_weights[feature] for feature in feature_order])
        diff = vec1 - vec2
        weighted_squared_diff = weights * (diff ** 2)
        return np.sqrt(np.sum(weighted_squared_diff))
    
    def _euclidean_distance(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate standard Euclidean distance."""
        return np.linalg.norm(vec1 - vec2)
    
    def _manhattan_distance(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate Manhattan (L1) distance."""
        feature_order = [
            'danceability', 'energy', 'valence', 'acousticness',
            'instrumentalness', 'speechiness', 'tempo', 'loudness', 'mode', 'key'
        ]
        
        weights = np.array([self.feature_weights[feature] for feature in feature_order])
        diff = np.abs(vec1 - vec2)
        return np.sum(weights * diff)
    
    def _cosine_distance(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine distance (1 - cosine similarity)."""
        # Add small epsilon to avoid division by zero
        epsilon = 1e-8
        
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1) + epsilon
        norm2 = np.linalg.norm(vec2) + epsilon
        
        cosine_similarity = dot_product / (norm1 * norm2)
        return 1.0 - cosine_similarity
    
    def update_weights(self, new_weights: Dict[str, float]):
        """Update feature weights and validate."""
        self.feature_weights.update(new_weights)
        self._validate_weights()


class SimilarityAnalyzer:
    """Advanced similarity analysis for song recommendations."""
    
    def __init__(self, distance_calculator: WeightedDistanceCalculator):
        self.distance_calculator = distance_calculator
    
    def find_most_similar_songs(
        self, 
        target_song: Dict, 
        candidate_songs: List[Dict], 
        n_recommendations: int = 10
    ) -> List[Tuple[Dict, float]]:
        """
        Find the most similar songs to a target song.
        
        Args:
            target_song: Target song with 'features' key
            candidate_songs: List of candidate songs with 'features' key
            n_recommendations: Number of recommendations to return
            
        Returns:
            List of (song, distance) tuples, sorted by similarity
        """
        similarities = []
        
        for candidate in candidate_songs:
            if candidate.get('id') == target_song.get('id'):
                continue  # Skip the target song itself
            
            try:
                distance = self.distance_calculator.calculate_distance(
                    target_song['features'], 
                    candidate['features']
                )
                similarities.append((candidate, distance))
            except (KeyError, TypeError) as e:
                logger.warning(f"Skipping song due to missing features: {e}")
                continue
        
        # Sort by distance (lower = more similar)
        similarities.sort(key=lambda x: x[1])
        return similarities[:n_recommendations]
    
    def find_most_different_songs(
        self, 
        target_song: Dict, 
        candidate_songs: List[Dict], 
        n_songs: int = 10
    ) -> List[Tuple[Dict, float]]:
        """
        Find the most different songs from a target song.
        
        Args:
            target_song: Target song with 'features' key
            candidate_songs: List of candidate songs with 'features' key
            n_songs: Number of different songs to return
            
        Returns:
            List of (song, distance) tuples, sorted by difference (descending)
        """
        differences = []
        
        for candidate in candidate_songs:
            if candidate.get('id') == target_song.get('id'):
                continue  # Skip the target song itself
            
            try:
                distance = self.distance_calculator.calculate_distance(
                    target_song['features'], 
                    candidate['features']
                )
                differences.append((candidate, distance))
            except (KeyError, TypeError) as e:
                logger.warning(f"Skipping song due to missing features: {e}")
                continue
        
        # Sort by distance (higher = more different)
        differences.sort(key=lambda x: x[1], reverse=True)
        return differences[:n_songs]
    
    def calculate_feature_importance(
        self, 
        selected_songs: List[Dict], 
        rejected_songs: List[Dict]
    ) -> Dict[str, float]:
        """
        Calculate which features are most important for user preferences.
        
        Args:
            selected_songs: Songs the user has selected
            rejected_songs: Songs the user has rejected
            
        Returns:
            Dictionary of feature importance scores
        """
        if not selected_songs or not rejected_songs:
            return {}
        
        feature_order = [
            'danceability', 'energy', 'valence', 'acousticness',
            'instrumentalness', 'speechiness', 'tempo', 'loudness', 'mode', 'key'
        ]
        
        # Calculate mean feature values for selected and rejected songs
        selected_features = []
        rejected_features = []
        
        for song in selected_songs:
            try:
                features = normalize_features(song['features'])
                selected_features.append(features)
            except (KeyError, TypeError):
                continue
        
        for song in rejected_songs:
            try:
                features = normalize_features(song['features'])
                rejected_features.append(features)
            except (KeyError, TypeError):
                continue
        
        if not selected_features or not rejected_features:
            return {}
        
        selected_mean = np.mean(selected_features, axis=0)
        rejected_mean = np.mean(rejected_features, axis=0)
        
        # Calculate feature importance as absolute difference in means
        feature_differences = np.abs(selected_mean - rejected_mean)
        
        importance_dict = {}
        for i, feature in enumerate(feature_order):
            if i < len(feature_differences):
                importance_dict[feature] = float(feature_differences[i])
        
        return importance_dict


# Utility functions for common operations

def calculate_music_distance(
    song1_features: Dict, 
    song2_features: Dict, 
    custom_weights: Optional[Dict[str, float]] = None
) -> float:
    """
    Convenience function for calculating distance between two songs.
    
    Args:
        song1_features: First song's audio features
        song2_features: Second song's audio features
        custom_weights: Optional custom feature weights
        
    Returns:
        Distance value between the songs
    """
    calculator = WeightedDistanceCalculator(custom_weights)
    return calculator.calculate_distance(song1_features, song2_features)


@lru_cache(maxsize=1000)
def cached_distance_calculation(
    features1_tuple: Tuple, 
    features2_tuple: Tuple
) -> float:
    """
    Cached version of distance calculation for performance.
    
    Args:
        features1_tuple: First song's features as tuple
        features2_tuple: Second song's features as tuple
        
    Returns:
        Cached distance calculation result
    """
    # Convert tuples back to dictionaries
    feature_names = [
        'danceability', 'energy', 'valence', 'acousticness',
        'instrumentalness', 'speechiness', 'tempo', 'loudness', 'mode', 'key'
    ]
    
    features1 = dict(zip(feature_names, features1_tuple))
    features2 = dict(zip(feature_names, features2_tuple))
    
    return calculate_music_distance(features1, features2)


def features_dict_to_tuple(features: Dict[str, Union[float, int]]) -> Tuple:
    """Convert features dictionary to tuple for caching."""
    feature_order = [
        'danceability', 'energy', 'valence', 'acousticness',
        'instrumentalness', 'speechiness', 'tempo', 'loudness', 'mode', 'key'
    ]
    
    return tuple(features.get(feature, 0.0) for feature in feature_order)


def batch_distance_calculation(
    target_features: Dict, 
    song_list: List[Dict], 
    use_cache: bool = True
) -> List[float]:
    """
    Calculate distances from target song to a list of songs efficiently.
    
    Args:
        target_features: Target song's audio features
        song_list: List of songs with 'features' key
        use_cache: Whether to use caching for performance
        
    Returns:
        List of distances corresponding to input song list
    """
    distances = []
    
    if use_cache:
        target_tuple = features_dict_to_tuple(target_features)
        
        for song in song_list:
            try:
                song_tuple = features_dict_to_tuple(song['features'])
                distance = cached_distance_calculation(target_tuple, song_tuple)
                distances.append(distance)
            except (KeyError, TypeError):
                distances.append(float('inf'))  # Invalid song
    else:
        calculator = WeightedDistanceCalculator()
        
        for song in song_list:
            try:
                distance = calculator.calculate_distance(target_features, song['features'])
                distances.append(distance)
            except (KeyError, TypeError):
                distances.append(float('inf'))  # Invalid song
    
    return distances