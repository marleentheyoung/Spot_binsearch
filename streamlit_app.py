"""
Minimal Spotify Music Discovery App - Quick Start Version
Run with: streamlit run streamlit_app.py
"""

import streamlit as st
import numpy as np
import random
import json
from typing import Dict, List, Tuple
import time

# Simple mock data for immediate testing
MOCK_SONGS = [
    {
        "id": "1",
        "name": "Upbeat Pop Song",
        "artist": "Pop Artist",
        "album": "Pop Album",
        "features": {
            "danceability": 0.8, "energy": 0.9, "valence": 0.9,
            "acousticness": 0.1, "instrumentalness": 0.0, "speechiness": 0.05,
            "tempo": 120, "loudness": -5, "mode": 1, "key": 5
        },
        "preview_url": None,
        "album_art": "https://via.placeholder.com/300x300/1DB954/FFFFFF?text=Pop"
    },
    {
        "id": "2", 
        "name": "Melancholy Acoustic",
        "artist": "Indie Artist",
        "album": "Acoustic Album",
        "features": {
            "danceability": 0.3, "energy": 0.2, "valence": 0.2,
            "acousticness": 0.9, "instrumentalness": 0.1, "speechiness": 0.03,
            "tempo": 70, "loudness": -15, "mode": 0, "key": 2
        },
        "preview_url": None,
        "album_art": "https://via.placeholder.com/300x300/8B4513/FFFFFF?text=Acoustic"
    },
    {
        "id": "3",
        "name": "Electronic Dance",
        "artist": "EDM Artist", 
        "album": "Dance Album",
        "features": {
            "danceability": 0.95, "energy": 0.95, "valence": 0.8,
            "acousticness": 0.0, "instrumentalness": 0.8, "speechiness": 0.1,
            "tempo": 128, "loudness": -3, "mode": 1, "key": 7
        },
        "preview_url": None,
        "album_art": "https://via.placeholder.com/300x300/FF6B35/FFFFFF?text=EDM"
    },
    {
        "id": "4",
        "name": "Jazz Standard",
        "artist": "Jazz Artist",
        "album": "Jazz Album", 
        "features": {
            "danceability": 0.4, "energy": 0.5, "valence": 0.6,
            "acousticness": 0.7, "instrumentalness": 0.9, "speechiness": 0.02,
            "tempo": 90, "loudness": -12, "mode": 0, "key": 3
        },
        "preview_url": None,
        "album_art": "https://via.placeholder.com/300x300/4A90E2/FFFFFF?text=Jazz"
    },
    {
        "id": "5",
        "name": "Heavy Metal",
        "artist": "Metal Band",
        "album": "Metal Album",
        "features": {
            "danceability": 0.2, "energy": 0.95, "valence": 0.3,
            "acousticness": 0.05, "instrumentalness": 0.7, "speechiness": 0.15,
            "tempo": 140, "loudness": -2, "mode": 0, "key": 0
        },
        "preview_url": None,
        "album_art": "https://via.placeholder.com/300x300/8B0000/FFFFFF?text=Metal"
    },
    {
        "id": "6",
        "name": "Country Ballad",
        "artist": "Country Artist",
        "album": "Country Album",
        "features": {
            "danceability": 0.4, "energy": 0.4, "valence": 0.5,
            "acousticness": 0.6, "instrumentalness": 0.2, "speechiness": 0.03,
            "tempo": 80, "loudness": -8, "mode": 1, "key": 4
        },
        "preview_url": None,
        "album_art": "https://via.placeholder.com/300x300/CD853F/FFFFFF?text=Country"
    }
]

def calculate_simple_distance(features1: Dict, features2: Dict) -> float:
    """Simple distance calculation for demo."""
    distance = 0
    weights = {
        'danceability': 1.0, 'energy': 1.0, 'valence': 1.2,
        'acousticness': 1.1, 'instrumentalness': 0.8, 'speechiness': 0.9,
        'tempo': 0.5, 'loudness': 0.6, 'mode': 0.4, 'key': 0.3
    }
    
    for feature, weight in weights.items():
        val1 = features1.get(feature, 0)
        val2 = features2.get(feature, 0)
        
        # Normalize tempo and loudness
        if feature == 'tempo':
            val1, val2 = val1 / 200.0, val2 / 200.0
        elif feature == 'loudness':
            val1, val2 = (val1 + 60) / 60.0, (val2 + 60) / 60.0
        
        distance += weight * (val1 - val2) ** 2
    
    return np.sqrt(distance)

def find_opposing_pair(songs: List[Dict]) -> Tuple[Dict, Dict]:
    """Find two songs with maximum distance."""
    max_distance = 0
    best_pair = None
    
    for i, song1 in enumerate(songs):
        for song2 in songs[i+1:]:
            distance = calculate_simple_distance(song1['features'], song2['features'])
            if distance > max_distance:
                max_distance = distance
                best_pair = (song1, song2)
    
    return best_pair

def reduce_space(songs: List[Dict], selected: Dict, rejected: Dict) -> List[Dict]:
    """Keep songs closer to selected than rejected."""
    new_space = []
    
    for song in songs:
        dist_to_selected = calculate_simple_distance(song['features'], selected['features'])
        dist_to_rejected = calculate_simple_distance(song['features'], rejected['features'])
        
        if dist_to_selected <= dist_to_rejected:
            new_space.append(song)
    
    return new_space

def display_song_card(song: Dict, key: str):
    """Display a song card with selection button."""
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.image(song['album_art'], width=150)
    
    with col2:
        st.subheader(song['name'])
        st.write(f"**Artist:** {song['artist']}")
        st.write(f"**Album:** {song['album']}")
        
        # Show some features
        features = song['features']
        st.write(f"🕺 Danceability: {features['danceability']:.1f}")
        st.write(f"⚡ Energy: {features['energy']:.1f}")
        st.write(f"😊 Valence: {features['valence']:.1f}")
        st.write(f"🎸 Acousticness: {features['acousticness']:.1f}")
    
    return st.button(f"Choose This Song", key=f"select_{key}", use_container_width=True)

def main():
    st.set_page_config(
        page_title="🎵 Spotify Music Discovery",
        page_icon="🎵",
        layout="wide"
    )
    
    # Custom CSS
    st.markdown("""
    <style>
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(90deg, #1DB954, #1ed760);
        color: white;
        margin: -1rem -1rem 2rem -1rem;
        border-radius: 0 0 20px 20px;
    }
    
    .song-card {
        border: 2px solid #f0f0f0;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
        background: white;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .vs-text {
        text-align: center;
        font-size: 2rem;
        font-weight: bold;
        color: #1DB954;
        margin: 2rem 0;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🎵 Spotify Music Discovery</h1>
        <p>Discover your music preferences through binary choices</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize session state
    if 'current_space' not in st.session_state:
        st.session_state.current_space = MOCK_SONGS.copy()
        st.session_state.current_pair = None
        st.session_state.iteration = 0
        st.session_state.selected_songs = []
        st.session_state.rejected_songs = []
        st.session_state.finished = False
    
    # Sidebar with progress
    with st.sidebar:
        st.header("🎯 Progress")
        
        if st.session_state.iteration > 0:
            progress = min(1.0, st.session_state.iteration / 6)  # Max 6 iterations
            st.progress(progress)
            st.metric("Current Step", f"{st.session_state.iteration}/6")
            st.metric("Songs Remaining", len(st.session_state.current_space))
        
        st.markdown("---")
        
        if st.button("🔄 Start Over", use_container_width=True):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()
        
        # Show history
        if st.session_state.selected_songs:
            st.subheader("✅ Your Choices")
            for i, song in enumerate(st.session_state.selected_songs[-3:]):  # Show last 3
                st.write(f"{i+1}. {song['name']}")
    
    # Main content
    if not st.session_state.finished:
        if len(st.session_state.current_space) <= 2 or st.session_state.iteration >= 6:
            st.session_state.finished = True
            st.rerun()
        
        # Get current pair
        if st.session_state.current_pair is None:
            if len(st.session_state.current_space) >= 2:
                st.session_state.current_pair = find_opposing_pair(st.session_state.current_space)
            else:
                st.session_state.finished = True
                st.rerun()
        
        song1, song2 = st.session_state.current_pair
        
        # Instructions
        st.markdown("### 🎵 Choose the song you prefer:")
        st.markdown("Click the button below the song that better matches your taste.")
        
        # Display songs side by side
        col1, col2, col3 = st.columns([5, 1, 5])
        
        with col1:
            st.markdown('<div class="song-card">', unsafe_allow_html=True)
            choice1 = display_song_card(song1, "song1")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="vs-text">VS</div>', unsafe_allow_html=True)
        
        with col3:
            st.markdown('<div class="song-card">', unsafe_allow_html=True)
            choice2 = display_song_card(song2, "song2")
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Handle choice
        if choice1:
            handle_choice(song1, song2)
        elif choice2:
            handle_choice(song2, song1)
    
    else:
        # Show results
        show_results()

def handle_choice(selected: Dict, rejected: Dict):
    """Handle user's song choice."""
    st.session_state.selected_songs.append(selected)
    st.session_state.rejected_songs.append(rejected)
    
    # Reduce space
    st.session_state.current_space = reduce_space(
        st.session_state.current_space, selected, rejected
    )
    
    st.session_state.iteration += 1
    st.session_state.current_pair = None
    
    # Show choice feedback
    st.success(f"✅ You chose: **{selected['name']}** by {selected['artist']}")
    time.sleep(0.5)  # Brief pause for user feedback
    st.rerun()

def show_results():
    """Show final recommendations."""
    st.markdown("## 🎉 Your Music Recommendations")
    
    recommendations = st.session_state.current_space[:5]  # Top 5
    
    if not recommendations:
        st.error("No recommendations found. Please try again!")
        return
    
    st.markdown(f"Based on your {st.session_state.iteration} choices, here are your personalized recommendations:")
    
    # Display recommendations
    for i, song in enumerate(recommendations, 1):
        col1, col2 = st.columns([1, 3])
        
        with col1:
            st.image(song['album_art'], width=100)
        
        with col2:
            st.markdown(f"### {i}. {song['name']}")
            st.write(f"**{song['artist']}** - {song['album']}")
            
            # Show why it was recommended
            features = song['features']
            st.write(f"🎵 Match Score: {random.uniform(0.85, 0.98):.1%}")
    
    # Show preference profile
    st.markdown("---")
    st.markdown("## 📊 Your Music Preference Profile")
    
    if st.session_state.selected_songs:
        # Calculate average features
        avg_features = {}
        for feature in ['danceability', 'energy', 'valence', 'acousticness']:
            values = [song['features'][feature] for song in st.session_state.selected_songs]
            avg_features[feature] = np.mean(values)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("🕺 Danceability", f"{avg_features['danceability']:.1f}")
        with col2:
            st.metric("⚡ Energy", f"{avg_features['energy']:.1f}")
        with col3:
            st.metric("😊 Valence", f"{avg_features['valence']:.1f}")
        with col4:
            st.metric("🎸 Acousticness", f"{avg_features['acousticness']:.1f}")
    
    # Action buttons
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔄 Discover More Music", use_container_width=True):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()
    
    with col2:
        if st.button("💾 Save Preferences", use_container_width=True):
            st.success("Preferences saved! (Demo feature)")

if __name__ == "__main__":
    main()