"""
Complete Spotify Music Discovery App - Simplified Version
Works without audio features using genres, popularity, year, and other metadata.
"""

import streamlit as st
import sys
import time
import logging
import traceback
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

# Add src to path for imports
project_root = Path(__file__).parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import our simplified modules
try:
    from spotify_code.client import get_spotify_client, reset_spotify_client
    from algorithm.binary_search import SimplifiedMusicBinarySearch, SearchState
except ImportError as e:
    st.error(f"Import error: {e}")
    st.error("Please make sure all required files are in place")
    st.stop()

def apply_custom_css():
    """Apply custom CSS styling."""
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    .song-card {
        border: 2px solid #e9ecef;
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        transition: all 0.3s ease;
    }
    
    .song-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 35px rgba(29, 185, 84, 0.15);
        border-color: #1DB954;
    }
    
    .stButton > button {
        background: linear-gradient(90deg, #1DB954 0%, #1ed760 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(29, 185, 84, 0.4);
    }
    
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #1DB954 0%, #1ed760 100%);
    }
    
    .vs-text {
        text-align: center;
        font-size: 3rem;
        font-weight: 700;
        color: #1DB954;
        margin: 2rem 0;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.1);
    }
    
    .app-header {
        background: linear-gradient(135deg, #1DB954 0%, #1ed760 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        text-align: center;
        color: white;
        box-shadow: 0 8px 25px rgba(29, 185, 84, 0.3);
    }
    
    .metadata-chip {
        display: inline-block;
        background: linear-gradient(90deg, #1DB954 0%, #1ed760 100%);
        color: white;
        padding: 0.25rem 0.75rem;
        border-radius: 15px;
        font-size: 0.8rem;
        margin: 0.25rem;
        font-weight: 500;
    }
    
    .genre-tag {
        background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
        color: #1976d2;
        padding: 0.2rem 0.6rem;
        border-radius: 12px;
        font-size: 0.75rem;
        margin: 0.1rem;
        display: inline-block;
        font-weight: 500;
    }
    </style>
    """, unsafe_allow_html=True)

def initialize_app():
    """Initialize the Streamlit app."""
    st.set_page_config(
        page_title="🎵 Spotify Music Discovery",
        page_icon="🎵",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    apply_custom_css()
    
    # Initialize session state
    if "app_initialized" not in st.session_state:
        st.session_state.app_initialized = True
        st.session_state.spotify_client = None
        st.session_state.binary_search = None
        st.session_state.current_pair = None
        st.session_state.search_state = SearchState.INITIALIZING
        st.session_state.error_message = None
        st.session_state.seed_songs = []

def check_spotify_setup():
    """Check if Spotify API is configured."""
    try:
        # Try to get credentials
        client_id = None
        client_secret = None
        
        try:
            if hasattr(st, 'secrets'):
                client_id = st.secrets.get('SPOTIFY_CLIENT_ID')
                client_secret = st.secrets.get('SPOTIFY_CLIENT_SECRET')
        except:
            pass
        
        if not client_id or not client_secret:
            import os
            client_id = os.getenv('SPOTIFY_CLIENT_ID')
            client_secret = os.getenv('SPOTIFY_CLIENT_SECRET')
        
        if not client_id or not client_secret:
            st.error("🔑 Spotify API not configured!")
            st.markdown("""
            **Setup Instructions:**
            1. Get Spotify API credentials from [Spotify Developer Dashboard](https://developer.spotify.com/dashboard)
            2. Create a `.env` file in the project root with:
               ```
               SPOTIFY_CLIENT_ID=your_client_id
               SPOTIFY_CLIENT_SECRET=your_client_secret
               ```
            3. Or set these in Streamlit secrets for deployed apps
            """)
            return False
        
        return True
        
    except Exception as e:
        st.error(f"Configuration check failed: {e}")
        return False

def initialize_spotify_client():
    """Initialize and cache the Spotify client."""
    if st.session_state.spotify_client is None:
        try:
            with st.spinner("🔌 Connecting to Spotify..."):
                client = get_spotify_client()
                if client.is_authenticated():
                    st.session_state.spotify_client = client
                    st.success("✅ Connected to Spotify!")
                    return True
                else:
                    st.error("❌ Failed to authenticate with Spotify")
                    return False
        except Exception as e:
            st.error(f"🔐 Connection Error: {e}")
            logger.error(f"Spotify client initialization failed: {e}")
            return False
    
    return True

def load_seed_songs():
    """Load initial song pool for the algorithm."""
    if not st.session_state.seed_songs:
        try:
            with st.spinner("🎵 Loading diverse music collection..."):
                client = st.session_state.spotify_client
                
                # Get diverse tracks
                all_songs = client.get_diverse_tracks(limit=80)
                
                if len(all_songs) < 20:
                    st.warning("⚠️ Limited song database loaded. Recommendations may be less diverse.")
                
                # Convert to dictionaries for the algorithm
                song_dicts = []
                for song in all_songs:
                    song_dict = {
                        'id': song.id,
                        'name': song.name,
                        'artist': song.artist,
                        'album': song.album,
                        'album_art_url': song.album_art_url,
                        'preview_url': song.preview_url,
                        'popularity': song.popularity,
                        'year': song.year,
                        'genres': song.genres,
                        'duration_ms': song.duration_ms,
                        'explicit': song.explicit
                    }
                    song_dicts.append(song_dict)
                
                st.session_state.seed_songs = song_dicts
                st.success(f"✅ Loaded {len(song_dicts)} diverse songs")
                
        except Exception as e:
            st.error(f"❌ Failed to load songs: {e}")
            logger.error(f"Seed songs loading failed: {e}")
            return False
    
    return len(st.session_state.seed_songs) > 0

def initialize_binary_search():
    """Initialize the binary search algorithm."""
    if st.session_state.binary_search is None and st.session_state.seed_songs:
        try:
            st.session_state.binary_search = SimplifiedMusicBinarySearch(
                initial_song_pool=st.session_state.seed_songs,
                max_iterations=6,
                min_songs_threshold=8
            )
            
            # Start the search and get initial pair
            st.session_state.current_pair = st.session_state.binary_search.start_search()
            st.session_state.search_state = SearchState.SEARCHING
            
            return True
            
        except Exception as e:
            st.error(f"❌ Failed to initialize search algorithm: {e}")
            logger.error(f"Binary search initialization failed: {e}")
            return False
    
    return st.session_state.binary_search is not None

def display_header():
    """Display the main application header."""
    st.markdown("""
    <div class="app-header">
        <h1 style="margin: 0; font-size: 2.5rem;">🎵 Spotify Music Discovery</h1>
        <p style="margin: 0.5rem 0 0 0; font-size: 1.2rem; opacity: 0.9;">
            Discover your music taste through smart choices
        </p>
    </div>
    """, unsafe_allow_html=True)

def display_song_card(song_dict: Dict, card_id: str, key: str) -> bool:
    """Display a song card with metadata and selection button."""
    # Extract song information
    song_name = song_dict.get('name', 'Unknown Song')
    artist = song_dict.get('artist', 'Unknown Artist')
    album = song_dict.get('album', 'Unknown Album')
    album_art_url = song_dict.get('album_art_url', '')
    preview_url = song_dict.get('preview_url')
    popularity = song_dict.get('popularity', 0)
    year = song_dict.get('year')
    genres = song_dict.get('genres', [])
    duration_ms = song_dict.get('duration_ms', 0)
    explicit = song_dict.get('explicit', False)
    
    # Album art with fallback
    if album_art_url:
        st.image(album_art_url, width=250)
    else:
        st.markdown("""
        <div style="
            width: 250px; 
            height: 250px; 
            background: linear-gradient(45deg, #1DB954, #1ed760);
            display: flex;
            align-items: center;
            justify-content: center;
            border-radius: 10px;
            color: white;
            font-size: 2rem;
            margin-bottom: 1rem;
        ">🎵</div>
        """, unsafe_allow_html=True)
    
    # Song information
    st.markdown(f"### {song_name}")
    st.markdown(f"**Artist:** {artist}")
    st.markdown(f"**Album:** {album}")
    
    # Metadata chips
    metadata_html = ""
    
    if popularity:
        metadata_html += f'<span class="metadata-chip">🔥 {popularity}% Popular</span>'
    
    if year:
        metadata_html += f'<span class="metadata-chip">📅 {year}</span>'
    
    if duration_ms:
        duration_min = duration_ms // 60000
        duration_sec = (duration_ms % 60000) // 1000
        metadata_html += f'<span class="metadata-chip">⏱️ {duration_min}:{duration_sec:02d}</span>'
    
    if explicit:
        metadata_html += f'<span class="metadata-chip">🔞 Explicit</span>'
    
    if metadata_html:
        st.markdown(metadata_html, unsafe_allow_html=True)
    
    # Genres
    if genres:
        genre_html = "**Genres:** "
        for genre in genres[:3]:  # Show max 3 genres
            genre_html += f'<span class="genre-tag">{genre.title()}</span>'
        st.markdown(genre_html, unsafe_allow_html=True)
    
    # Audio preview
    if preview_url:
        st.audio(preview_url, format='audio/mp3')
    else:
        st.info("🔇 No preview available")
    
    # Selection button
    return st.button(
        f"🎵 Choose This Song", 
        key=f"select_{key}_{card_id}", 
        use_container_width=True,
        type="primary"
    )

def handle_song_choice(selected_song_dict: Dict, rejected_song_dict: Dict):
    """Handle user's song selection and advance the algorithm."""
    try:
        choice_time = time.time() - st.session_state.get('choice_start_time', time.time())
        
        # Make choice in algorithm
        next_pair = st.session_state.binary_search.make_choice(
            selected_song_dict, 
            choice_time=choice_time
        )
        
        # Update session state
        st.session_state.current_pair = next_pair
        
        # Show user feedback
        st.success(f"✅ You chose: **{selected_song_dict['name']}** by {selected_song_dict['artist']}")
        
        # Check if search is complete
        if next_pair is None:
            st.session_state.search_state = SearchState.CONVERGED
            st.balloons()
            time.sleep(1)
        
        # Force rerun to update UI
        time.sleep(0.5)
        st.rerun()
        
    except Exception as e:
        st.error(f"❌ Error processing your choice: {e}")
        logger.error(f"Choice handling failed: {e}")

def show_song_comparison():
    """Display the main song comparison interface."""
    if not st.session_state.current_pair:
        st.error("No song pair available. Please restart the discovery process.")
        return
    
    song1_dict, song2_dict = st.session_state.current_pair
    
    st.markdown("### 🎵 Which song better matches your taste?")
    st.markdown("Choose the song you prefer based on the artist, genre, era, and any other factors that matter to you:")
    
    # Record when choice starts
    st.session_state.choice_start_time = time.time()
    
    # Two-column layout with VS divider
    col1, col_vs, col2 = st.columns([5, 1, 5])
    
    with col1:
        st.markdown('<div class="song-card">', unsafe_allow_html=True)
        choice1 = display_song_card(song1_dict, "song1", key="song1_card")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col_vs:
        st.markdown('<div class="vs-text">VS</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="song-card">', unsafe_allow_html=True)
        choice2 = display_song_card(song2_dict, "song2", key="song2_card")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Handle choices
    if choice1:
        handle_song_choice(song1_dict, song2_dict)
    elif choice2:
        handle_song_choice(song2_dict, song1_dict)

def show_sidebar():
    """Display sidebar with progress and controls."""
    with st.sidebar:
        st.header("🎯 Discovery Progress")
        
        if st.session_state.binary_search:
            debug_info = st.session_state.binary_search.get_debug_info()
            
            # Progress bar
            progress = debug_info['current_iteration'] / debug_info['max_iterations']
            st.progress(progress)
            
            # Current stats
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Step", f"{debug_info['current_iteration']}/{debug_info['max_iterations']}")
            with col2:
                st.metric("Songs Left", debug_info['current_space_size'])
            
            # Learned preferences
            if debug_info['selected_songs_count'] > 0:
                st.markdown("### 🎵 Your Preferences")
                
                learned_prefs = debug_info['learned_preferences']
                
                if learned_prefs['preferred_genres']:
                    st.markdown("**Liked Genres:**")
                    for genre in list(learned_prefs['preferred_genres'])[:3]:
                        st.markdown(f"• {genre.title()}")
                
                if learned_prefs['popularity_preference']:
                    pref = learned_prefs['popularity_preference']
                    emoji = "🔥" if pref == "high" else "🎭"
                    st.markdown(f"**Popularity:** {emoji} {pref.title()}")
                
                if learned_prefs['era_preference']:
                    decade = learned_prefs['era_preference']
                    st.markdown(f"**Era:** 📅 {decade}s")
        
        st.markdown("---")
        
        # Control buttons
        if st.button("🔄 Start New Discovery", use_container_width=True):
            restart_discovery()
        
        if st.button("⚙️ Reset Connection", use_container_width=True):
            reset_connection()
        
        # Show status
        st.markdown("### 📡 Status")
        if st.session_state.spotify_client:
            st.success("✅ Spotify Connected")
        else:
            st.error("❌ Spotify Disconnected")
        
        if st.session_state.seed_songs:
            st.info(f"🎵 {len(st.session_state.seed_songs)} songs loaded")

def restart_discovery():
    """Restart the music discovery process."""
    st.session_state.binary_search = None
    st.session_state.current_pair = None
    st.session_state.search_state = SearchState.INITIALIZING
    
    st.success("🔄 Discovery process restarted!")
    time.sleep(1)
    st.rerun()

def reset_connection():
    """Reset Spotify connection."""
    reset_spotify_client()
    st.session_state.spotify_client = None
    st.session_state.seed_songs = []
    st.success("🔌 Connection reset. Reconnecting...")
    time.sleep(1)
    st.rerun()

def show_final_results():
    """Display final recommendations."""
    if st.session_state.binary_search:
        search_result = st.session_state.binary_search.get_recommendations(n_recommendations=8)
        
        st.markdown("## 🎉 Your Personalized Music Recommendations")
        
        # Success metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Steps Completed", search_result.total_iterations)
        with col2:
            st.metric("Final Pool", search_result.final_space_size)
        with col3:
            st.metric("Recommendations", len(search_result.recommended_songs))
        
        st.markdown("---")
        
        # Display recommendations
        st.markdown("### 🎵 Your Top Picks")
        
        if search_result.recommended_songs:
            # Show recommendations in pairs
            for i in range(0, len(search_result.recommended_songs), 2):
                col1, col2 = st.columns(2)
                
                # First song
                with col1:
                    if i < len(search_result.recommended_songs):
                        song = search_result.recommended_songs[i]
                        display_recommendation_card(song, i + 1)
                
                # Second song
                with col2:
                    if i + 1 < len(search_result.recommended_songs):
                        song = search_result.recommended_songs[i + 1]
                        display_recommendation_card(song, i + 2)
        
        # User preference profile
        st.markdown("---")
        st.markdown("### 📊 Your Music Profile")
        
        preferences = search_result.user_preferences
        
        col1, col2 = st.columns(2)
        
        with col1:
            if preferences['preferred_genres']:
                st.markdown("**🎼 Favorite Genres:**")
                for genre in preferences['preferred_genres'][:5]:
                    st.markdown(f"• {genre.title()}")
            
            if preferences['popularity_preference']:
                pref = preferences['popularity_preference']
                emoji = "🔥" if pref == "high" else "🎭"
                st.markdown(f"**{emoji} Popularity Preference:** {pref.title()}")
        
        with col2:
            if preferences['era_preference']:
                decade = preferences['era_preference']
                st.markdown(f"**📅 Preferred Era:** {decade}s")
            
            if preferences['duration_preference']:
                dur = preferences['duration_preference']
                emoji = "⚡" if dur == "short" else "🎭"
                st.markdown(f"**{emoji} Song Length:** {dur.title()}")
        
        # Action buttons
        st.markdown("---")
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🔄 Discover More Music", use_container_width=True):
                restart_discovery()
        
        with col2:
            if st.button("📊 View Discovery Journey", use_container_width=True):
                show_search_history(search_result.search_history)
    
    else:
        st.error("No search results available.")

def display_recommendation_card(song: Dict, rank: int):
    """Display a recommendation card."""
    st.markdown(f"### {rank}. {song['name']}")
    st.markdown(f"**{song['artist']}** - {song['album']}")
    
    # Album art
    if song.get('album_art_url'):
        st.image(song['album_art_url'], width=200)
    
    # Metadata
    metadata_parts = []
    if song.get('year'):
        metadata_parts.append(f"📅 {song['year']}")
    if song.get('popularity'):
        metadata_parts.append(f"🔥 {song['popularity']}% popular")
    
    if metadata_parts:
        st.markdown(" • ".join(metadata_parts))
    
    # Genres
    if song.get('genres'):
        genre_html = ""
        for genre in song['genres'][:2]:
            genre_html += f'<span class="genre-tag">{genre.title()}</span>'
        st.markdown(genre_html, unsafe_allow_html=True)
    
    # Audio preview
    if song.get('preview_url'):
        st.audio(song['preview_url'], format='audio/mp3')

def show_search_history(search_history: List):
    """Display search history in an expander."""
    with st.expander("🔍 Your Discovery Journey", expanded=False):
        for i, iteration in enumerate(search_history, 1):
            col1, col2, col3 = st.columns([2, 1, 2])
            
            with col1:
                if iteration.selected_song:
                    st.markdown(f"**Step {i} - ✅ Chosen:**")
                    st.markdown(f"🎵 {iteration.selected_song['name']}")
                    st.markdown(f"👤 {iteration.selected_song['artist']}")
            
            with col2:
                st.markdown("**vs**")
            
            with col3:
                if iteration.rejected_song:
                    st.markdown(f"**Step {i} - ❌ Rejected:**")
                    st.markdown(f"🎵 {iteration.rejected_song['name']}")
                    st.markdown(f"👤 {iteration.rejected_song['artist']}")
            
            # Show space reduction
            if iteration.space_size_after:
                reduction = iteration.space_size_before - iteration.space_size_after
                st.markdown(f"📉 Narrowed by {reduction} songs → {iteration.space_size_after} remaining")
            
            st.markdown("---")

def main():
    """Main application logic."""
    try:
        # Initialize app
        initialize_app()
        
        # Display header
        display_header()
        
        # Check Spotify setup
        if not check_spotify_setup():
            return
        
        # Initialize Spotify client
        if not initialize_spotify_client():
            return
        
        # Load seed songs
        if not load_seed_songs():
            return
        
        # Show sidebar
        show_sidebar()
        
        # Main content based on search state
        if st.session_state.search_state == SearchState.CONVERGED:
            show_final_results()
        else:
            # Initialize binary search if needed
            if initialize_binary_search():
                show_song_comparison()
            else:
                st.error("Failed to initialize music discovery algorithm.")
    
    except Exception as e:
        st.error(f"❌ Application error: {e}")
        logger.error(f"Main application error: {e}")
        st.markdown("**Debug Information:**")
        st.code(traceback.format_exc())
        
        if st.button("🔄 Restart Application"):
            # Clear all session state
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

if __name__ == "__main__":
    main()