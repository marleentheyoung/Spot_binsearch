"""
Binary search algorithm for Spotify Music Discovery.
Implements intelligent music space reduction based on user preferences.
"""

import numpy as np
import random
from typing import Dict, List, Tuple, Optional, Set, Any
from dataclasses import dataclass, field
from enum import Enum
import logging
from collections import defaultdict
import json
import time

from distance import (
    WeightedDistanceCalculator, 
    SimilarityAnalyzer, 
    calculate_music_distance,
    batch_distance_calculation
)

# Setup logging
logger = logging.getLogger(__name__)


class SearchState(Enum):
    """Current state of the binary search algorithm."""
    INITIALIZING = "initializing"
    SEARCHING = "searching"
    CONVERGED = "converged"
    FAILED = "failed"


class SelectionStrategy(Enum):
    """Strategy for selecting opposing song pairs."""
    MAXIMUM_DISTANCE = "maximum_distance"
    CLUSTER_REPRESENTATIVES = "cluster_representatives"
    WEIGHTED_SAMPLING = "weighted_sampling"
    ADAPTIVE = "adaptive"


@dataclass
class SearchIteration:
    """Represents a single iteration of the binary search."""
    iteration_number: int
    song_pair: Tuple[Dict, Dict]
    selected_song: Optional[Dict] = None
    rejected_song: Optional[Dict] = None
    space_size_before: int = 0
    space_size_after: int = 0
    distance_between_pair: float = 0.0
    user_choice_time: Optional[float] = None
    feature_importance: Dict[str, float] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


@dataclass
class SearchResult:
    """Final result of the binary search process."""
    recommended_songs: List[Dict]
    confidence_score: float
    total_iterations: int
    final_space_size: int
    user_preference_profile: Dict[str, float]
    search_history: List[SearchIteration]
    convergence_reason: str
    success: bool


class MusicSpaceBinarySearch:
    """
    Core binary search algorithm for music preference discovery.
    
    Uses iterative space reduction based on user choices between opposing songs
    to converge on personalized music recommendations.
    """
    
    def __init__(
        self,
        initial_song_pool: List[Dict],
        max_iterations: int = 8,
        min_songs_threshold: int = 10,
        convergence_threshold: float = 0.1,
        selection_strategy: SelectionStrategy = SelectionStrategy.ADAPTIVE,
        custom_weights: Optional[Dict[str, float]] = None
    ):
        """
        Initialize the binary search algorithm.
        
        Args:
            initial_song_pool: List of songs with audio features
            max_iterations: Maximum number of search iterations
            min_songs_threshold: Stop when space is reduced to this size
            convergence_threshold: Distance threshold for convergence detection
            selection_strategy: Strategy for selecting opposing pairs
            custom_weights: Custom feature weights for distance calculation
        """
        self.initial_song_pool = initial_song_pool.copy()
        self.current_space = initial_song_pool.copy()
        self.max_iterations = max_iterations
        self.min_songs_threshold = min_songs_threshold
        self.convergence_threshold = convergence_threshold
        self.selection_strategy = selection_strategy
        
        # Algorithm components
        self.distance_calculator = WeightedDistanceCalculator(custom_weights)
        self.similarity_analyzer = SimilarityAnalyzer(self.distance_calculator)
        
        # Search state
        self.current_iteration = 0
        self.search_history: List[SearchIteration] = []
        self.state = SearchState.INITIALIZING
        self.current_pair: Optional[Tuple[Dict, Dict]] = None
        
        # User preference learning
        self.selected_songs: List[Dict] = []
        self.rejected_songs: List[Dict] = []
        self.learned_weights: Dict[str, float] = {}
        
        # Performance tracking
        self.space_reduction_rate: List[float] = []
        self.decision_times: List[float] = []
        
        self._validate_initial_pool()
        
    def _validate_initial_pool(self):
        """Validate the initial song pool."""
        if not self.initial_song_pool:
            raise ValueError("Initial song pool cannot be empty")
        
        # Check that songs have required features
        required_keys = ['id', 'features']
        for song in self.initial_song_pool[:10]:  # Sample check
            if not all(key in song for key in required_keys):
                raise ValueError(f"Songs must have keys: {required_keys}")
            
            # Validate features structure
            features = song['features']
            required_features = [
                'danceability', 'energy', 'valence', 'acousticness',
                'instrumentalness', 'speechiness', 'tempo', 'loudness', 'mode', 'key'
            ]
            
            missing_features = [f for f in required_features if f not in features]
            if missing_features:
                logger.warning(f"Song {song.get('id', 'unknown')} missing features: {missing_features}")
        
        logger.info(f"Initialized with {len(self.initial_song_pool)} songs")
    
    def start_search(self) -> Tuple[Dict, Dict]:
        """
        Start the binary search process.
        
        Returns:
            Initial opposing song pair for user choice
        """
        self.state = SearchState.SEARCHING
        self.current_iteration = 0
        self.current_space = self.initial_song_pool.copy()
        
        # Get initial opposing pair
        self.current_pair = self._find_opposing_pair()
        
        if not self.current_pair:
            self.state = SearchState.FAILED
            raise RuntimeError("Failed to find initial opposing pair")
        
        # Create initial iteration record
        iteration = SearchIteration(
            iteration_number=self.current_iteration,
            song_pair=self.current_pair,
            space_size_before=len(self.current_space),
            distance_between_pair=self._calculate_pair_distance(self.current_pair)
        )
        
        logger.info(f"Started search with {len(self.current_space)} songs")
        logger.info(f"Initial pair distance: {iteration.distance_between_pair:.3f}")
        
        return self.current_pair
    
    def make_choice(
        self, 
        selected_song: Dict, 
        choice_time: Optional[float] = None
    ) -> Optional[Tuple[Dict, Dict]]:
        """
        Process user's choice and return next song pair, or None if converged.
        
        Args:
            selected_song: Song chosen by the user
            choice_time: Time taken for user to make choice (seconds)
            
        Returns:
            Next opposing pair, or None if search is complete
        """
        if self.state != SearchState.SEARCHING:
            raise RuntimeError(f"Cannot make choice in state: {self.state}")
        
        if not self.current_pair:
            raise RuntimeError("No current pair to choose from")
        
        song1, song2 = self.current_pair
        
        # Determine which song was selected and rejected
        if selected_song['id'] == song1['id']:
            rejected_song = song2
        elif selected_song['id'] == song2['id']:
            rejected_song = song1
        else:
            raise ValueError("Selected song is not one of the current pair")
        
        # Record the choice
        self.selected_songs.append(selected_song)
        self.rejected_songs.append(rejected_song)
        
        # Update current iteration record
        current_iteration = self.search_history[-1] if self.search_history else SearchIteration(
            iteration_number=self.current_iteration,
            song_pair=self.current_pair,
            space_size_before=len(self.current_space),
            distance_between_pair=self._calculate_pair_distance(self.current_pair)
        )
        
        current_iteration.selected_song = selected_song
        current_iteration.rejected_song = rejected_song
        current_iteration.user_choice_time = choice_time
        
        # Reduce the search space
        old_space_size = len(self.current_space)
        self._reduce_space(selected_song, rejected_song)
        new_space_size = len(self.current_space)
        
        current_iteration.space_size_after = new_space_size
        
        # Calculate space reduction rate
        reduction_rate = (old_space_size - new_space_size) / old_space_size
        self.space_reduction_rate.append(reduction_rate)
        
        # Learn feature importance
        if len(self.selected_songs) >= 2 and len(self.rejected_songs) >= 2:
            current_iteration.feature_importance = self.similarity_analyzer.calculate_feature_importance(
                self.selected_songs, self.rejected_songs
            )
            self._update_learned_weights(current_iteration.feature_importance)
        
        # Add to history
        if self.search_history and self.search_history[-1].iteration_number == self.current_iteration:
            self.search_history[-1] = current_iteration
        else:
            self.search_history.append(current_iteration)
        
        # Record decision time
        if choice_time:
            self.decision_times.append(choice_time)
        
        # Check convergence
        if self._check_convergence():
            self.state = SearchState.CONVERGED
            logger.info(f"Search converged after {self.current_iteration + 1} iterations")
            return None
        
        # Move to next iteration
        self.current_iteration += 1
        
        # Get next opposing pair
        try:
            self.current_pair = self._find_opposing_pair()
            
            if not self.current_pair:
                logger.warning("Could not find next opposing pair, ending search")
                self.state = SearchState.CONVERGED
                return None
            
            # Create next iteration record
            next_iteration = SearchIteration(
                iteration_number=self.current_iteration,
                song_pair=self.current_pair,
                space_size_before=len(self.current_space),
                distance_between_pair=self._calculate_pair_distance(self.current_pair)
            )
            
            logger.info(f"Iteration {self.current_iteration}: {len(self.current_space)} songs remaining")
            logger.info(f"Next pair distance: {next_iteration.distance_between_pair:.3f}")
            
            return self.current_pair
            
        except Exception as e:
            logger.error(f"Error finding next pair: {e}")
            self.state = SearchState.FAILED
            return None
    
    def _find_opposing_pair(self) -> Optional[Tuple[Dict, Dict]]:
        """Find the most opposing pair of songs in the current space."""
        if len(self.current_space) < 2:
            return None
        
        if self.selection_strategy == SelectionStrategy.MAXIMUM_DISTANCE:
            return self._find_maximum_distance_pair()
        elif self.selection_strategy == SelectionStrategy.CLUSTER_REPRESENTATIVES:
            return self._find_cluster_representatives()
        elif self.selection_strategy == SelectionStrategy.WEIGHTED_SAMPLING:
            return self._find_weighted_sampling_pair()
        elif self.selection_strategy == SelectionStrategy.ADAPTIVE:
            return self._find_adaptive_pair()
        else:
            return self._find_maximum_distance_pair()
    
    def _find_maximum_distance_pair(self) -> Optional[Tuple[Dict, Dict]]:
        """Find pair with maximum distance in current space."""
        max_distance = 0.0
        best_pair = None
        
        # For performance, sample if space is large
        sample_size = min(100, len(self.current_space))
        sampled_songs = random.sample(self.current_space, sample_size) if len(self.current_space) > sample_size else self.current_space
        
        for i, song1 in enumerate(sampled_songs):
            for song2 in sampled_songs[i+1:]:
                try:
                    distance = self.distance_calculator.calculate_distance(
                        song1['features'], song2['features']
                    )
                    
                    if distance > max_distance:
                        max_distance = distance
                        best_pair = (song1, song2)
                        
                except Exception as e:
                    logger.warning(f"Error calculating distance: {e}")
                    continue
        
        return best_pair
    
    def _find_cluster_representatives(self) -> Optional[Tuple[Dict, Dict]]:
        """Find representatives from different clusters."""
        # Simple clustering approach: divide space into regions and pick representatives
        if len(self.current_space) < 2:
            return None
        
        # For simplicity, use k=2 clustering based on principal features
        try:
            from sklearn.cluster import KMeans
            
            # Extract feature vectors
            feature_vectors = []
            for song in self.current_space:
                features = song['features']
                vector = [
                    features.get('valence', 0.5),
                    features.get('energy', 0.5),
                    features.get('danceability', 0.5),
                    features.get('acousticness', 0.5)
                ]
                feature_vectors.append(vector)
            
            # Cluster into 2 groups
            kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
            labels = kmeans.fit_predict(feature_vectors)
            
            # Find representatives from each cluster
            cluster_0_songs = [song for i, song in enumerate(self.current_space) if labels[i] == 0]
            cluster_1_songs = [song for i, song in enumerate(self.current_space) if labels[i] == 1]
            
            if not cluster_0_songs or not cluster_1_songs:
                # Fall back to maximum distance
                return self._find_maximum_distance_pair()
            
            # Pick random representatives
            song1 = random.choice(cluster_0_songs)
            song2 = random.choice(cluster_1_songs)
            
            return (song1, song2)
            
        except ImportError:
            logger.warning("scikit-learn not available, falling back to maximum distance")
            return self._find_maximum_distance_pair()
        except Exception as e:
            logger.warning(f"Clustering failed: {e}, falling back to maximum distance")
            return self._find_maximum_distance_pair()
    
    def _find_weighted_sampling_pair(self) -> Optional[Tuple[Dict, Dict]]:
        """Find pair using weighted sampling based on diversity."""
        if len(self.current_space) < 2:
            return None
        
        # Calculate diversity scores for each song
        diversity_scores = []
        
        for song in self.current_space:
            # Calculate average distance to all other songs
            distances = batch_distance_calculation(
                song['features'], 
                [s for s in self.current_space if s['id'] != song['id']]
            )
            avg_distance = np.mean([d for d in distances if d != float('inf')])
            diversity_scores.append(avg_distance)
        
        # Sample two songs with probability proportional to diversity
        if sum(diversity_scores) == 0:
            return self._find_maximum_distance_pair()
        
        weights = np.array(diversity_scores) / sum(diversity_scores)
        
        try:
            indices = np.random.choice(
                len(self.current_space), 
                size=2, 
                replace=False, 
                p=weights
            )
            return (self.current_space[indices[0]], self.current_space[indices[1]])
        except:
            return self._find_maximum_distance_pair()
    
    def _find_adaptive_pair(self) -> Optional[Tuple[Dict, Dict]]:
        """Adaptive strategy that changes based on search progress."""
        progress = self.current_iteration / self.max_iterations
        
        if progress < 0.3:
            # Early stage: use maximum distance for broad exploration
            return self._find_maximum_distance_pair()
        elif progress < 0.7:
            # Middle stage: use clustering for structured exploration
            return self._find_cluster_representatives()
        else:
            # Late stage: use weighted sampling for fine-tuning
            return self._find_weighted_sampling_pair()
    
    def _reduce_space(self, selected_song: Dict, rejected_song: Dict):
        """Reduce search space to songs closer to selected song."""
        new_space = []
        
        for song in self.current_space:
            try:
                dist_to_selected = self.distance_calculator.calculate_distance(
                    song['features'], selected_song['features']
                )
                dist_to_rejected = self.distance_calculator.calculate_distance(
                    song['features'], rejected_song['features']
                )
                
                # Keep songs that are closer to selected than rejected
                if dist_to_selected <= dist_to_rejected:
                    new_space.append(song)
                    
            except Exception as e:
                logger.warning(f"Error calculating distance for space reduction: {e}")
                # Keep song if we can't calculate distance
                new_space.append(song)
        
        # Ensure we don't reduce space to nothing
        if len(new_space) == 0:
            logger.warning("Space reduction resulted in empty space, keeping original")
            return
        
        self.current_space = new_space
        logger.debug(f"Space reduced to {len(self.current_space)} songs")
    
    def _check_convergence(self) -> bool:
        """Check if search has converged."""
        # Check minimum songs threshold
        if len(self.current_space) <= self.min_songs_threshold:
            return True
        
        # Check maximum iterations
        if self.current_iteration >= self.max_iterations - 1:
            return True
        
        # Check if space is not reducing enough
        if len(self.space_reduction_rate) >= 2:
            recent_reductions = self.space_reduction_rate[-2:]
            if all(rate < 0.1 for rate in recent_reductions):  # Less than 10% reduction
                logger.info("Space reduction rate too low, converging")
                return True
        
        # Check feature space convergence
        if len(self.current_space) > 5:
            return self._check_feature_space_convergence()
        
        return False
    
    def _check_feature_space_convergence(self) -> bool:
        """Check if remaining songs are similar in feature space."""
        if len(self.current_space) < 2:
            return True
        
        # Calculate pairwise distances among remaining songs
        distances = []
        sample_size = min(20, len(self.current_space))
        sampled_songs = random.sample(self.current_space, sample_size)
        
        for i, song1 in enumerate(sampled_songs):
            for song2 in sampled_songs[i+1:]:
                try:
                    distance = self.distance_calculator.calculate_distance(
                        song1['features'], song2['features']
                    )
                    distances.append(distance)
                except:
                    continue
        
        if not distances:
            return False
        
        # Check if all distances are below threshold
        max_distance = max(distances)
        return max_distance < self.convergence_threshold
    
    def _calculate_pair_distance(self, pair: Tuple[Dict, Dict]) -> float:
        """Calculate distance between two songs in a pair."""
        try:
            return self.distance_calculator.calculate_distance(
                pair[0]['features'], pair[1]['features']
            )
        except:
            return 0.0
    
    def _update_learned_weights(self, feature_importance: Dict[str, float]):
        """Update learned weights based on feature importance."""
        learning_rate = 0.1
        
        for feature, importance in feature_importance.items():
            if feature in self.learned_weights:
                self.learned_weights[feature] = (
                    (1 - learning_rate) * self.learned_weights[feature] + 
                    learning_rate * importance
                )
            else:
                self.learned_weights[feature] = importance
        
        # Update distance calculator weights
        current_weights = self.distance_calculator.feature_weights.copy()
        for feature, learned_weight in self.learned_weights.items():
            if feature in current_weights:
                # Blend learned weights with defaults
                current_weights[feature] *= (1 + learned_weight)
        
        self.distance_calculator.update_weights(current_weights)
    
    def get_recommendations(self, n_recommendations: int = 10) -> SearchResult:
        """
        Get final recommendations based on the search results.
        
        Args:
            n_recommendations: Number of songs to recommend
            
        Returns:
            SearchResult containing recommendations and metadata
        """
        if self.state not in [SearchState.CONVERGED, SearchState.FAILED]:
            logger.warning("Getting recommendations before search completion")
        
        # Get final recommendations
        recommended_songs = self.current_space[:n_recommendations]
        
        # Calculate confidence score
        confidence_score = self._calculate_confidence_score()
        
        # Generate user preference profile
        user_preference_profile = self._generate_preference_profile()
        
        # Determine convergence reason
        convergence_reason = self._get_convergence_reason()
        
        return SearchResult(
            recommended_songs=recommended_songs,
            confidence_score=confidence_score,
            total_iterations=self.current_iteration + 1,
            final_space_size=len(self.current_space),
            user_preference_profile=user_preference_profile,
            search_history=self.search_history.copy(),
            convergence_reason=convergence_reason,
            success=(self.state == SearchState.CONVERGED)
        )
    
    def _calculate_confidence_score(self) -> float:
        """Calculate confidence in the recommendations."""
        factors = []
        
        # Factor 1: Number of iterations completed
        iteration_factor = min(1.0, self.current_iteration / (self.max_iterations * 0.8))
        factors.append(iteration_factor)
        
        # Factor 2: Space reduction effectiveness
        if self.space_reduction_rate:
            avg_reduction = np.mean(self.space_reduction_rate)
            reduction_factor = min(1.0, avg_reduction / 0.5)  # Target 50% reduction per iteration
            factors.append(reduction_factor)
        
        # Factor 3: Final space size appropriateness
        final_size_factor = 1.0 if len(self.current_space) <= self.min_songs_threshold * 2 else 0.5
        factors.append(final_size_factor)
        
        # Factor 4: Consistency of choices (if we have decision times)
        if len(self.decision_times) >= 3:
            time_consistency = 1.0 - (np.std(self.decision_times) / np.mean(self.decision_times))
            time_factor = max(0.0, min(1.0, time_consistency))
            factors.append(time_factor)
        
        return np.mean(factors)
    
    def _generate_preference_profile(self) -> Dict[str, float]:
        """Generate user preference profile from selected songs."""
        if not self.selected_songs:
            return {}
        
        # Calculate average features of selected songs
        feature_sums = defaultdict(float)
        feature_counts = defaultdict(int)
        
        for song in self.selected_songs:
            for feature, value in song['features'].items():
                feature_sums[feature] += value
                feature_counts[feature] += 1
        
        preference_profile = {}
        for feature in feature_sums:
            if feature_counts[feature] > 0:
                preference_profile[feature] = feature_sums[feature] / feature_counts[feature]
        
        return preference_profile
    
    def _get_convergence_reason(self) -> str:
        """Get the reason why the search converged."""
        if len(self.current_space) <= self.min_songs_threshold:
            return f"Reached minimum songs threshold ({self.min_songs_threshold})"
        elif self.current_iteration >= self.max_iterations - 1:
            return f"Reached maximum iterations ({self.max_iterations})"
        elif len(self.space_reduction_rate) >= 2 and all(rate < 0.1 for rate in self.space_reduction_rate[-2:]):
            return "Space reduction rate too low"
        elif self._check_feature_space_convergence():
            return "Feature space convergence achieved"
        else:
            return "Unknown convergence reason"
    
    def reset(self):
        """Reset the algorithm for a new search."""
        self.current_space = self.initial_song_pool.copy()
        self.current_iteration = 0
        self.search_history.clear()
        self.state = SearchState.INITIALIZING
        self.current_pair = None
        self.selected_songs.clear()
        self.rejected_songs.clear()
        self.learned_weights.clear()
        self.space_reduction_rate.clear()
        self.decision_times.clear()
        
        logger.info("Algorithm reset for new search")
    
    def get_debug_info(self) -> Dict[str, Any]:
        """Get debug information about the current search state."""
        return {
            'state': self.state.value,
            'current_iteration': self.current_iteration,
            'current_space_size': len(self.current_space),
            'initial_space_size': len(self.initial_song_pool),
            'space_reduction_rate': self.space_reduction_rate,
            'selected_songs_count': len(self.selected_songs),
            'rejected_songs_count': len(self.rejected_songs),
            'learned_weights': self.learned_weights,
            'current_pair_distance': self._calculate_pair_distance(self.current_pair) if self.current_pair else None,
            'selection_strategy': self.selection_strategy.value,
            'average_decision_time': np.mean(self.decision_times) if self.decision_times else None
        }