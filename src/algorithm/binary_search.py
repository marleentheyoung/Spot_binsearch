"""
Simplified binary search algorithm using basic music metadata.
Works without audio features - uses genres, popularity, year, etc.
"""

import random
import logging
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass
from enum import Enum
import time

logger = logging.getLogger(__name__)

class SearchState(Enum):
    """Current state of the binary search algorithm."""
    INITIALIZING = "initializing"
    SEARCHING = "searching"
    CONVERGED = "converged"
    FAILED = "failed"

@dataclass
class SearchIteration:
    """Represents a single iteration of the binary search."""
    iteration_number: int
    song_pair: Tuple[Dict, Dict]
    selected_song: Optional[Dict] = None
    rejected_song: Optional[Dict] = None
    space_size_before: int = 0
    space_size_after: int = 0
    user_choice_time: Optional[float] = None
    timestamp: float = None

@dataclass
class SearchResult:
    """Final result of the binary search process."""
    recommended_songs: List[Dict]
    total_iterations: int
    final_space_size: int
    user_preferences: Dict[str, any]
    search_history: List[SearchIteration]
    convergence_reason: str
    success: bool

class SimplifiedMusicBinarySearch:
    """
    Simplified binary search using basic music metadata.
    Uses genres, popularity, release year, and duration instead of audio features.
    """
    
    def __init__(
        self,
        initial_song_pool: List[Dict],
        max_iterations: int = 6,
        min_songs_threshold: int = 10
    ):
        self.initial_song_pool = initial_song_pool.copy()
        self.current_space = initial_song_pool.copy()
        self.max_iterations = max_iterations
        self.min_songs_threshold = min_songs_threshold
        
        # Search state
        self.current_iteration = 0
        self.search_history: List[SearchIteration] = []
        self.state = SearchState.INITIALIZING
        self.current_pair: Optional[Tuple[Dict, Dict]] = None
        
        # User preference learning
        self.selected_songs: List[Dict] = []
        self.rejected_songs: List[Dict] = []
        self.learned_preferences = {
            'preferred_genres': set(),
            'avoided_genres': set(),
            'popularity_preference': None,  # 'high', 'low', or None
            'era_preference': None,  # decade preference
            'duration_preference': None  # 'short', 'long', or None
        }
        
        logger.info(f"Initialized with {len(self.initial_song_pool)} songs")
    
    def start_search(self) -> Tuple[Dict, Dict]:
        """Start the binary search process."""
        self.state = SearchState.SEARCHING
        self.current_iteration = 0
        self.current_space = self.initial_song_pool.copy()
        
        # Get initial opposing pair
        self.current_pair = self._find_opposing_pair()
        
        if not self.current_pair:
            self.state = SearchState.FAILED
            raise RuntimeError("Failed to find initial opposing pair")
        
        logger.info(f"Started search with {len(self.current_space)} songs")
        return self.current_pair
    
    def make_choice(self, selected_song: Dict, choice_time: Optional[float] = None) -> Optional[Tuple[Dict, Dict]]:
        """Process user's choice and return next song pair, or None if converged."""
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
        
        # Create iteration record
        iteration = SearchIteration(
            iteration_number=self.current_iteration,
            song_pair=self.current_pair,
            selected_song=selected_song,
            rejected_song=rejected_song,
            space_size_before=len(self.current_space),
            user_choice_time=choice_time,
            timestamp=time.time()
        )
        
        # Learn from the choice
        self._update_preferences(selected_song, rejected_song)
        
        # Reduce the search space
        old_space_size = len(self.current_space)
        self._reduce_space(selected_song, rejected_song)
        
        iteration.space_size_after = len(self.current_space)
        self.search_history.append(iteration)
        
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
            
            logger.info(f"Iteration {self.current_iteration}: {len(self.current_space)} songs remaining")
            return self.current_pair
            
        except Exception as e:
            logger.error(f"Error finding next pair: {e}")
            self.state = SearchState.FAILED
            return None
    
    def _find_opposing_pair(self) -> Optional[Tuple[Dict, Dict]]:
        """Find the most opposing pair based on metadata differences."""
        if len(self.current_space) < 2:
            return None
        
        max_difference = 0
        best_pair = None
        
        # Sample for performance if space is large
        sample_size = min(50, len(self.current_space))
        sampled_songs = random.sample(self.current_space, sample_size)
        
        for i, song1 in enumerate(sampled_songs):
            for song2 in sampled_songs[i+1:]:
                difference = self._calculate_song_difference(song1, song2)
                
                if difference > max_difference:
                    max_difference = difference
                    best_pair = (song1, song2)
        
        return best_pair
    
    def _calculate_song_difference(self, song1: Dict, song2: Dict) -> float:
        """Calculate difference between two songs based on metadata."""
        difference = 0.0
        
        # Genre difference (most important)
        genre_diff = self._calculate_genre_difference(song1.get('genres', []), song2.get('genres', []))
        difference += genre_diff * 3.0
        
        # Popularity difference
        pop1 = song1.get('popularity', 50)
        pop2 = song2.get('popularity', 50)
        pop_diff = abs(pop1 - pop2) / 100.0
        difference += pop_diff * 2.0
        
        # Year difference
        year1 = song1.get('year', 2000)
        year2 = song2.get('year', 2000)
        if year1 and year2:
            year_diff = abs(year1 - year2) / 50.0  # Normalize by ~50 years
            difference += min(year_diff, 1.0) * 1.5
        
        # Duration difference
        dur1 = song1.get('duration_ms', 180000) / 1000  # Convert to seconds
        dur2 = song2.get('duration_ms', 180000) / 1000
        dur_diff = abs(dur1 - dur2) / 300.0  # Normalize by 5 minutes
        difference += min(dur_diff, 1.0) * 0.5
        
        # Explicit content difference
        if song1.get('explicit', False) != song2.get('explicit', False):
            difference += 0.5
        
        return difference
    
    def _calculate_genre_difference(self, genres1: List[str], genres2: List[str]) -> float:
        """Calculate difference between two genre lists."""
        if not genres1 or not genres2:
            return 0.5  # Moderate difference for missing genres
        
        # Convert to sets for easier comparison
        set1 = set(g.lower() for g in genres1)
        set2 = set(g.lower() for g in genres2)
        
        # Calculate Jaccard distance (1 - Jaccard similarity)
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        
        if union == 0:
            return 0.0
        
        jaccard_similarity = intersection / union
        return 1.0 - jaccard_similarity
    
    def _reduce_space(self, selected_song: Dict, rejected_song: Dict):
        """Reduce search space to songs more similar to selected song."""
        new_space = []
        
        for song in self.current_space:
            # Calculate similarity to selected vs rejected
            sim_to_selected = 1.0 - self._calculate_song_difference(song, selected_song)
            sim_to_rejected = 1.0 - self._calculate_song_difference(song, rejected_song)
            
            # Keep songs more similar to selected than rejected
            if sim_to_selected >= sim_to_rejected:
                new_space.append(song)
        
        # Ensure we don't reduce space to nothing
        if len(new_space) == 0:
            logger.warning("Space reduction resulted in empty space, keeping original")
            return
        
        self.current_space = new_space
        logger.debug(f"Space reduced to {len(self.current_space)} songs")
    
    def _update_preferences(self, selected_song: Dict, rejected_song: Dict):
        """Learn user preferences from their choice."""
        # Genre preferences
        selected_genres = set(g.lower() for g in selected_song.get('genres', []))
        rejected_genres = set(g.lower() for g in rejected_song.get('genres', []))
        
        self.learned_preferences['preferred_genres'].update(selected_genres)
        self.learned_preferences['avoided_genres'].update(rejected_genres)
        
        # Popularity preference
        selected_pop = selected_song.get('popularity', 50)
        rejected_pop = rejected_song.get('popularity', 50)
        
        if selected_pop > rejected_pop + 20:
            self.learned_preferences['popularity_preference'] = 'high'
        elif selected_pop < rejected_pop - 20:
            self.learned_preferences['popularity_preference'] = 'low'
        
        # Era preference (by decade)
        selected_year = selected_song.get('year')
        rejected_year = rejected_song.get('year')
        
        if selected_year and rejected_year:
            selected_decade = (selected_year // 10) * 10
            rejected_decade = (rejected_year // 10) * 10
            
            if selected_decade != rejected_decade:
                self.learned_preferences['era_preference'] = selected_decade
        
        # Duration preference
        selected_duration = selected_song.get('duration_ms', 0) / 1000
        rejected_duration = rejected_song.get('duration_ms', 0) / 1000
        
        if selected_duration < rejected_duration - 60:  # More than 1 minute shorter
            self.learned_preferences['duration_preference'] = 'short'
        elif selected_duration > rejected_duration + 60:  # More than 1 minute longer
            self.learned_preferences['duration_preference'] = 'long'
    
    def _check_convergence(self) -> bool:
        """Check if search has converged."""
        # Check minimum songs threshold
        if len(self.current_space) <= self.min_songs_threshold:
            return True
        
        # Check maximum iterations
        if self.current_iteration >= self.max_iterations - 1:
            return True
        
        return False
    
    def get_recommendations(self, n_recommendations: int = 10) -> SearchResult:
        """Get final recommendations based on search results."""
        recommended_songs = self.current_space[:n_recommendations]
        
        # Sort by estimated preference
        recommended_songs = self._rank_songs_by_preference(recommended_songs)
        
        # Generate user preference summary
        user_preferences = self._generate_preference_summary()
        
        # Determine convergence reason
        convergence_reason = self._get_convergence_reason()
        
        return SearchResult(
            recommended_songs=recommended_songs[:n_recommendations],
            total_iterations=self.current_iteration + 1,
            final_space_size=len(self.current_space),
            user_preferences=user_preferences,
            search_history=self.search_history.copy(),
            convergence_reason=convergence_reason,
            success=(self.state == SearchState.CONVERGED)
        )
    
    def _rank_songs_by_preference(self, songs: List[Dict]) -> List[Dict]:
        """Rank songs by learned user preferences."""
        def preference_score(song):
            score = 0.0
            
            # Genre preference
            song_genres = set(g.lower() for g in song.get('genres', []))
            preferred_genres = self.learned_preferences['preferred_genres']
            avoided_genres = self.learned_preferences['avoided_genres']
            
            if song_genres.intersection(preferred_genres):
                score += 2.0
            if song_genres.intersection(avoided_genres):
                score -= 1.0
            
            # Popularity preference
            popularity = song.get('popularity', 50)
            if self.learned_preferences['popularity_preference'] == 'high' and popularity > 70:
                score += 1.0
            elif self.learned_preferences['popularity_preference'] == 'low' and popularity < 40:
                score += 1.0
            
            # Era preference
            year = song.get('year')
            preferred_era = self.learned_preferences['era_preference']
            if year and preferred_era:
                decade = (year // 10) * 10
                if decade == preferred_era:
                    score += 1.0
            
            # Add some randomness to avoid identical rankings
            score += random.uniform(-0.1, 0.1)
            
            return score
        
        return sorted(songs, key=preference_score, reverse=True)
    
    def _generate_preference_summary(self) -> Dict[str, any]:
        """Generate a summary of learned user preferences."""
        return {
            'preferred_genres': list(self.learned_preferences['preferred_genres']),
            'avoided_genres': list(self.learned_preferences['avoided_genres']),
            'popularity_preference': self.learned_preferences['popularity_preference'],
            'era_preference': self.learned_preferences['era_preference'],
            'duration_preference': self.learned_preferences['duration_preference'],
            'total_choices': len(self.selected_songs)
        }
    
    def _get_convergence_reason(self) -> str:
        """Get the reason why the search converged."""
        if len(self.current_space) <= self.min_songs_threshold:
            return f"Reached minimum songs threshold ({self.min_songs_threshold})"
        elif self.current_iteration >= self.max_iterations - 1:
            return f"Reached maximum iterations ({self.max_iterations})"
        else:
            return "Unknown convergence reason"
    
    def get_debug_info(self) -> Dict[str, any]:
        """Get debug information about the current search state."""
        return {
            'state': self.state.value,
            'current_iteration': self.current_iteration,
            'current_space_size': len(self.current_space),
            'initial_space_size': len(self.initial_song_pool),
            'selected_songs_count': len(self.selected_songs),
            'rejected_songs_count': len(self.rejected_songs),
            'learned_preferences': self.learned_preferences,
            'max_iterations': self.max_iterations
        }