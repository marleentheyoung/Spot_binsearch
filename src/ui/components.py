"""
UI Components for Spotify Music Discovery Streamlit App
"""

import streamlit as st
from typing import Dict, Any, List, Optional
import time


def display_song_card(song_dict: Dict, card_id: str, key: str) -> bool:
    """
    Display a song card with audio preview and selection button.
    
    Args:
        song_dict: Dictionary containing song information
        card_id: Unique identifier for the card
        key: Streamlit key for the button
    
    Returns:
        bool: True if song was selected
    """
    # Extract song information
    song_name = song_dict.get('name', 'Unknown Song')
    artist = song_dict.get('artist', 'Unknown Artist')
    album = song_dict.get('album', 'Unknown Album')
    album_art_url = song_dict.get('album_art_url', song_dict.get('album_art', ''))
    preview_url = song_dict.get('preview_url')
    popularity = song_dict.get('popularity', 0)
    features = song_dict.get('features', {})
    
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
    
    if popularity:
        st.markdown(f"**Popularity:** {popularity}/100")
    
    # Audio preview
    if preview_url:
        st.audio(preview_url, format='audio/mp3')
    else:
        st.info("🔇 No preview available for this song")
    
    # Audio features visualization
    if features:
        st.markdown("**Musical Features:**")
        
        # Key features to display
        display_features = {
            'danceability': '🕺 Danceability',
            'energy': '⚡ Energy', 
            'valence': '😊 Positivity',
            'acousticness': '🎸 Acousticness'
        }
        
        for feature_key, feature_label in display_features.items():
            if feature_key in features:
                value = features[feature_key]
                # Create a visual bar
                bar_width = int(value * 100)
                st.markdown(f"{feature_label}: {'█' * (bar_width // 10)}{'░' * (10 - bar_width // 10)} {value:.1f}")
    
    # Selection button
    button_style = f"""
    <style>
    div.stButton > button:first-child {{
        background: linear-gradient(90deg, #1DB954 0%, #1ed760 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.75rem 1.5rem;
        font-size: 1.1rem;
        font-weight: bold;
        width: 100%;
        transition: all 0.3s ease;
    }}
    div.stButton > button:first-child:hover {{
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(29, 185, 84, 0.4);
    }}
    </style>
    """
    st.markdown(button_style, unsafe_allow_html=True)
    
    return st.button(
        f"🎵 Choose This Song", 
        key=f"select_{key}_{card_id}", 
        use_container_width=True,
        type="primary"
    )


def show_progress_sidebar(binary_search_instance):
    """Display progress information in the sidebar."""
    debug_info = binary_search_instance.get_debug_info()
    
    # Progress bar
    progress = debug_info['current_iteration'] / binary_search_instance.max_iterations
    st.progress(progress)
    
    # Current stats
    col1, col2 = st.columns(2)
    with col1:
        st.metric(
            "Step", 
            f"{debug_info['current_iteration']}/{binary_search_instance.max_iterations}"
        )
    with col2:
        st.metric("Songs Left", debug_info['current_space_size'])
    
    # Search state
    state_emoji = {
        'initializing': '🔄',
        'searching': '🔍', 
        'converged': '✅',
        'failed': '❌'
    }
    
    current_state = debug_info['state']
    st.markdown(f"**Status:** {state_emoji.get(current_state, '❓')} {current_state.title()}")
    
    # Space reduction visualization
    if debug_info['current_space_size'] > 0:
        initial_size = len(binary_search_instance.initial_song_pool)
        reduction_percent = (1 - debug_info['current_space_size'] / initial_size) * 100
        st.markdown(f"**Space Reduced:** {reduction_percent:.1f}%")
    
    # Recent choices
    if debug_info['selected_songs_count'] > 0:
        st.markdown("### 🎵 Your Recent Choices")
        
        # Show last few selected songs
        for i, iteration in enumerate(binary_search_instance.search_history[-3:]):
            if iteration.selected_song:
                song = iteration.selected_song
                st.markdown(f"{i+1}. **{song['name']}** by {song['artist']}")
    
    # Performance metrics
    if debug_info['average_decision_time']:
        st.markdown(f"**Avg. Decision Time:** {debug_info['average_decision_time']:.1f}s")


def show_results_page(search_result):
    """Display the final results page with recommendations."""
    st.markdown("## 🎉 Your Personalized Music Recommendations")
    
    if not search_result.success:
        st.error("❌ Search did not complete successfully. Please try again.")
        return
    
    # Success metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Iterations", search_result.total_iterations)
    with col2:
        st.metric("Final Songs", search_result.final_space_size)
    with col3:
        st.metric("Confidence", f"{search_result.confidence_score:.1%}")
    with col4:
        convergence_emoji = "🎯" if "threshold" in search_result.convergence_reason else "⏱️"
        st.metric("Convergence", f"{convergence_emoji} Complete")
    
    st.markdown("---")
    
    # Display recommendations
    st.markdown("### 🎵 Your Top Recommendations")
    
    if not search_result.recommended_songs:
        st.warning("No specific recommendations found. Please try the discovery process again.")
        return
    
    # Show top recommendations in a grid
    for i, song in enumerate(search_result.recommended_songs[:6], 1):
        with st.container():
            col1, col2, col3 = st.columns([1, 3, 1])
            
            with col1:
                # Album art
                album_art = song.get('album_art_url', song.get('album_art', ''))
                if album_art:
                    st.image(album_art, width=100)
                else:
                    st.markdown(f"""
                    <div style="
                        width: 100px; height: 100px; 
                        background: linear-gradient(45deg, #1DB954, #1ed760);
                        border-radius: 10px; display: flex; align-items: center; 
                        justify-content: center; color: white; font-size: 1.5rem;
                    ">🎵</div>
                    """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"### {i}. {song['name']}")
                st.markdown(f"**{song['artist']}** - {song['album']}")
                
                # Show key features that influenced the recommendation
                features = song.get('features', {})
                if features:
                    feature_text = []
                    if features.get('valence', 0) > 0.7:
                        feature_text.append("😊 Upbeat")
                    elif features.get('valence', 0) < 0.3:
                        feature_text.append("😔 Melancholic")
                    
                    if features.get('energy', 0) > 0.7:
                        feature_text.append("⚡ High Energy")
                    elif features.get('energy', 0) < 0.3:
                        feature_text.append("🕯️ Chill")
                    
                    if features.get('danceability', 0) > 0.7:
                        feature_text.append("🕺 Danceable")
                    
                    if features.get('acousticness', 0) > 0.7:
                        feature_text.append("🎸 Acoustic")
                    
                    if feature_text:
                        st.markdown(f"*{' • '.join(feature_text)}*")
                
                # Audio preview
                preview_url = song.get('preview_url')
                if preview_url:
                    st.audio(preview_url, format='audio/mp3')
            
            with col3:
                # Match score (simplified calculation)
                match_score = max(0.7, min(0.99, 0.85 + (i * -0.05)))
                st.metric("Match", f"{match_score:.0%}")
        
        st.markdown("---")
    
    # User preference profile
    st.markdown("### 📊 Your Music Preference Profile")
    
    profile = search_result.user_preference_profile
    if profile:
        # Create preference radar chart visualization
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Your Preferences:**")
            
            preference_labels = {
                'valence': ('😊 Mood', 'Happy' if profile.get('valence', 0.5) > 0.5 else 'Melancholic'),
                'energy': ('⚡ Energy', 'High' if profile.get('energy', 0.5) > 0.5 else 'Low'),
                'danceability': ('🕺 Danceability', 'Danceable' if profile.get('danceability', 0.5) > 0.5 else 'Non-danceable'),
                'acousticness': ('🎸 Sound', 'Acoustic' if profile.get('acousticness', 0.5) > 0.5 else 'Electronic')
            }
            
            for feature, (label, preference) in preference_labels.items():
                value = profile.get(feature, 0.5)
                st.markdown(f"**{label}:** {preference} ({value:.1f})")
        
        with col2:
            st.markdown("**Feature Analysis:**")
            
            # Show what makes this profile unique
            high_features = [k for k, v in profile.items() if v > 0.7]
            low_features = [k for k, v in profile.items() if v < 0.3]
            
            if high_features:
                st.markdown(f"**Strong preferences:** {', '.join(high_features)}")
            if low_features:
                st.markdown(f"**Avoids:** {', '.join(low_features)}")
            
            # Convergence info
            st.markdown(f"**Reason for completion:** {search_result.convergence_reason}")
    
    # Action buttons
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🔄 Discover More Music", use_container_width=True):
            # Reset for new discovery
            for key in ['binary_search', 'current_pair', 'search_state']:
                if key in st.session_state:
                    del st.session_state[key]
            st.rerun()
    
    with col2:
        if st.button("📊 View Search History", use_container_width=True):
            show_search_history(search_result.search_history)
    
    with col3:
        if st.button("💾 Export Results", use_container_width=True):
            export_results(search_result)


def show_search_history(search_history: List):
    """Display the search history in an expandable section."""
    with st.expander("🔍 Search History Details", expanded=False):
        st.markdown("### Your Discovery Journey")
        
        for i, iteration in enumerate(search_history, 1):
            col1, col2, col3 = st.columns([2, 1, 2])
            
            with col1:
                if iteration.selected_song:
                    st.markdown(f"**Step {i} - Selected:**")
                    st.markdown(f"🎵 {iteration.selected_song['name']}")
                    st.markdown(f"👤 {iteration.selected_song['artist']}")
            
            with col2:
                st.markdown("**vs**")
                if iteration.distance_between_pair:
                    st.markdown(f"Distance: {iteration.distance_between_pair:.2f}")
            
            with col3:
                if iteration.rejected_song:
                    st.markdown(f"**Step {i} - Rejected:**")
                    st.markdown(f"🎵 {iteration.rejected_song['name']}")
                    st.markdown(f"👤 {iteration.rejected_song['artist']}")
            
            # Show space reduction
            if iteration.space_size_before and iteration.space_size_after:
                reduction = iteration.space_size_before - iteration.space_size_after
                st.markdown(f"📉 Reduced space by {reduction} songs ({iteration.space_size_after} remaining)")
            
            if iteration.user_choice_time:
                st.markdown(f"⏱️ Decision time: {iteration.user_choice_time:.1f}s")
            
            st.markdown("---")


def export_results(search_result):
    """Export search results for the user."""
    try:
        import json
        
        # Create exportable data
        export_data = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'recommendations': [
                {
                    'name': song['name'],
                    'artist': song['artist'],
                    'album': song['album'],
                    'spotify_id': song.get('id', ''),
                }
                for song in search_result.recommended_songs[:10]
            ],
            'preference_profile': search_result.user_preference_profile,
            'discovery_stats': {
                'total_iterations': search_result.total_iterations,
                'final_space_size': search_result.final_space_size,
                'confidence_score': search_result.confidence_score,
                'convergence_reason': search_result.convergence_reason
            }
        }
        
        # Convert to JSON
        json_data = json.dumps(export_data, indent=2)
        
        # Download button
        st.download_button(
            label="📄 Download Results (JSON)",
            data=json_data,
            file_name=f"spotify_discovery_{int(time.time())}.json",
            mime="application/json"
        )
        
        st.success("✅ Results ready for download!")
        
    except Exception as e:
        st.error(f"❌ Export failed: {e}")


def apply_custom_css():
    """Apply custom CSS styling to the Streamlit app."""
    st.markdown("""
    <style>
    /* Custom styling for the Spotify app */
    
    .main > div {
        padding-top: 2rem;
    }
    
    .song-card {
        border: 2px solid #f0f0f0;
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        transition: all 0.3s ease;
    }
    
    .song-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 25px rgba(29, 185, 84, 0.15);
        border-color: #1DB954;
    }
    
    /* Spotify-themed colors */
    .stSelectbox > div > div {
        background-color: #f8f9fa;
    }
    
    .stTextInput > div > div > input {
        background-color: #f8f9fa;
    }
    
    /* Progress bar styling */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #1DB954 0%, #1ed760 100%);
    }
    
    /* Metric styling */
    .metric-container {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #1DB954;
    }
    
    /* Audio player customization */
    audio {
        width: 100%;
        height: 40px;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #f8f9fa 0%, #e9ecef 100%);
    }
    
    /* Button hover effects */
    .stButton > button {
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
    }
    
    /* Custom animations */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .fade-in {
        animation: fadeIn 0.5s ease-out;
    }
    
    /* Responsive design */
    @media (max-width: 768px) {
        .song-card {
            margin: 0.5rem 0;
            padding: 1rem;
        }
    }
    
    /* Success/Error message styling */
    .stAlert > div {
        border-radius: 10px;
    }
    
    /* Feature bar styling */
    .feature-bar {
        background: linear-gradient(90deg, #1DB954 0%, #1ed760 100%);
        height: 8px;
        border-radius: 4px;
        margin: 2px 0;
    }
    
    </style>
    """, unsafe_allow_html=True)