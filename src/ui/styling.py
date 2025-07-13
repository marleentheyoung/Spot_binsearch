"""
UI Styling and theming for Spotify Music Discovery app.
"""

import streamlit as st
from typing import Dict, Optional


def apply_custom_css():
    """Apply custom CSS styling to the Streamlit app."""
    st.markdown("""
    <style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global styles */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    .main > div {
        padding-top: 1rem;
        max-width: 1200px;
        margin: 0 auto;
    }
    
    /* Hide Streamlit default elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Custom song card styling */
    .song-card {
        border: 2px solid #e9ecef;
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    
    .song-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        background: linear-gradient(90deg, #1DB954 0%, #1ed760 100%);
        transform: scaleX(0);
        transition: transform 0.3s ease;
    }
    
    .song-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 35px rgba(29, 185, 84, 0.15);
        border-color: #1DB954;
    }
    
    .song-card:hover::before {
        transform: scaleX(1);
    }
    
    /* Spotify-themed progress bar */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #1DB954 0%, #1ed760 100%);
        border-radius: 10px;
    }
    
    /* Custom button styling */
    .stButton > button {
        background: linear-gradient(90deg, #1DB954 0%, #1ed760 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(29, 185, 84, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(29, 185, 84, 0.4);
        background: linear-gradient(90deg, #1ed760 0%, #1DB954 100%);
    }
    
    .stButton > button:active {
        transform: translateY(0);
        box-shadow: 0 2px 10px rgba(29, 185, 84, 0.3);
    }
    
    /* Secondary button styling */
    .stButton > button[data-testid="secondary"] {
        background: transparent;
        color: #1DB954;
        border: 2px solid #1DB954;
        box-shadow: none;
    }
    
    .stButton > button[data-testid="secondary"]:hover {
        background: #1DB954;
        color: white;
    }
    
    /* Metric container styling */
    .metric-container {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        padding: 1.5rem;
        border-radius: 15px;
        border-left: 5px solid #1DB954;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
        margin: 0.5rem 0;
    }
    
    /* Audio player styling */
    audio {
        width: 100%;
        height: 50px;
        border-radius: 10px;
        margin: 1rem 0;
    }
    
    audio::-webkit-media-controls-panel {
        background-color: #f8f9fa;
        border-radius: 10px;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #f8f9fa 0%, #e9ecef 100%);
        border-right: 2px solid #1DB954;
    }
    
    .css-1d391kg .css-1v0mbdj {
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    
    /* Alert styling */
    .stAlert > div {
        border-radius: 15px;
        border-left: 5px solid;
        font-weight: 500;
    }
    
    .stSuccess > div {
        border-left-color: #1DB954;
        background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
    }
    
    .stError > div {
        border-left-color: #dc3545;
        background: linear-gradient(135deg, #f8d7da 0%, #f5c6cb 100%);
    }
    
    .stWarning > div {
        border-left-color: #ffc107;
        background: linear-gradient(135deg, #fff3cd 0%, #ffeaa7 100%);
    }
    
    .stInfo > div {
        border-left-color: #17a2b8;
        background: linear-gradient(135deg, #d1ecf1 0%, #bee5eb 100%);
    }
    
    /* Custom animations */
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.05); }
        100% { transform: scale(1); }
    }
    
    .fade-in-up {
        animation: fadeInUp 0.6s ease-out;
    }
    
    .pulse {
        animation: pulse 2s infinite;
    }
    
    /* VS text styling */
    .vs-text {
        text-align: center;
        font-size: 3rem;
        font-weight: 700;
        color: #1DB954;
        margin: 2rem 0;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.1);
        animation: pulse 3s infinite;
    }
    
    /* Header gradient */
    .app-header {
        background: linear-gradient(135deg, #1DB954 0%, #1ed760 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        text-align: center;
        color: white;
        box-shadow: 0 8px 25px rgba(29, 185, 84, 0.3);
    }
    
    .app-header h1 {
        margin: 0;
        font-size: 2.5rem;
        font-weight: 700;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.2);
    }
    
    .app-header p {
        margin: 0.5rem 0 0 0;
        font-size: 1.2rem;
        opacity: 0.9;
        font-weight: 400;
    }
    
    /* Feature visualization bars */
    .feature-bar {
        background: linear-gradient(90deg, #1DB954 0%, #1ed760 100%);
        height: 8px;
        border-radius: 4px;
        margin: 4px 0;
        transition: all 0.3s ease;
    }
    
    .feature-bar:hover {
        height: 10px;
        box-shadow: 0 2px 8px rgba(29, 185, 84, 0.4);
    }
    
    /* Responsive design */
    @media (max-width: 768px) {
        .main > div {
            padding: 1rem;
        }
        
        .song-card {
            margin: 0.5rem 0;
            padding: 1rem;
        }
        
        .app-header h1 {
            font-size: 2rem;
        }
        
        .app-header p {
            font-size: 1rem;
        }
        
        .vs-text {
            font-size: 2rem;
            margin: 1rem 0;
        }
    }
    
    /* Loading spinner */
    .loading-spinner {
        border: 4px solid #f3f3f3;
        border-top: 4px solid #1DB954;
        border-radius: 50%;
        width: 40px;
        height: 40px;
        animation: spin 1s linear infinite;
        margin: 20px auto;
    }
    
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: #f1f1f1;
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(180deg, #1DB954 0%, #1ed760 100%);
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(180deg, #1ed760 0%, #1DB954 100%);
    }
    
    /* Results page styling */
    .recommendation-card {
        background: white;
        border: 1px solid #e9ecef;
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    
    .recommendation-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        width: 4px;
        height: 100%;
        background: linear-gradient(180deg, #1DB954 0%, #1ed760 100%);
    }
    
    .recommendation-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 20px rgba(29, 185, 84, 0.2);
    }
    
    /* Match score styling */
    .match-score {
        background: linear-gradient(135deg, #1DB954 0%, #1ed760 100%);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.9rem;
        text-align: center;
        box-shadow: 0 2px 8px rgba(29, 185, 84, 0.3);
    }
    
    /* Preference profile styling */
    .preference-item {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border-left: 4px solid #1DB954;
        transition: all 0.3s ease;
    }
    
    .preference-item:hover {
        transform: translateX(5px);
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
    }
    
    </style>
    """, unsafe_allow_html=True)


def get_spotify_theme_colors() -> Dict[str, str]:
    """Get Spotify-themed color palette."""
    return {
        'primary': '#1DB954',      # Spotify Green
        'primary_light': '#1ed760', # Lighter Spotify Green
        'secondary': '#191414',    # Spotify Black
        'background': '#FFFFFF',   # White background
        'surface': '#F8F9FA',     # Light surface
        'text': '#191414',        # Dark text
        'text_secondary': '#6C757D', # Secondary text
        'border': '#E9ECEF',      # Border color
        'success': '#28A745',     # Success green
        'warning': '#FFC107',     # Warning yellow
        'error': '#DC3545',       # Error red
        'info': '#17A2B8'         # Info blue
    }


def create_song_card_html(song_data: Dict, card_id: str) -> str:
    """Create HTML for a song card with custom styling."""
    colors = get_spotify_theme_colors()
    
    return f"""
    <div class="song-card fade-in-up" id="card-{card_id}">
        <div class="card-header">
            <img src="{song_data.get('album_art_url', '')}" 
                 alt="Album Art" 
                 style="width: 100%; border-radius: 10px; margin-bottom: 1rem;">
        </div>
        <div class="card-content">
            <h3 style="margin: 0 0 0.5rem 0; color: {colors['text']};">
                {song_data.get('name', 'Unknown Song')}
            </h3>
            <p style="margin: 0 0 0.25rem 0; color: {colors['text_secondary']}; font-weight: 500;">
                {song_data.get('artist', 'Unknown Artist')}
            </p>
            <p style="margin: 0 0 1rem 0; color: {colors['text_secondary']}; font-size: 0.9rem;">
                {song_data.get('album', 'Unknown Album')}
            </p>
        </div>
    </div>
    """


def create_progress_indicator(current: int, total: int, label: str = "Progress") -> str:
    """Create a custom progress indicator."""
    colors = get_spotify_theme_colors()
    progress_percent = (current / total) * 100
    
    return f"""
    <div style="margin: 1rem 0;">
        <div style="
            display: flex; 
            justify-content: space-between; 
            align-items: center; 
            margin-bottom: 0.5rem;
        ">
            <span style="font-weight: 600; color: {colors['text']};">{label}</span>
            <span style="color: {colors['text_secondary']};">{current}/{total}</span>
        </div>
        <div style="
            width: 100%; 
            height: 12px; 
            background-color: {colors['surface']}; 
            border-radius: 6px; 
            overflow: hidden;
            box-shadow: inset 0 2px 4px rgba(0,0,0,0.1);
        ">
            <div style="
                width: {progress_percent}%; 
                height: 100%; 
                background: linear-gradient(90deg, {colors['primary']} 0%, {colors['primary_light']} 100%);
                transition: width 0.3s ease;
                border-radius: 6px;
            "></div>
        </div>
    </div>
    """


def create_feature_bar(feature_name: str, value: float, max_value: float = 1.0) -> str:
    """Create a visual feature bar."""
    colors = get_spotify_theme_colors()
    percentage = (value / max_value) * 100
    
    return f"""
    <div style="margin: 0.5rem 0;">
        <div style="
            display: flex; 
            justify-content: space-between; 
            margin-bottom: 0.25rem;
        ">
            <span style="font-size: 0.9rem; color: {colors['text']};">{feature_name}</span>
            <span style="font-size: 0.9rem; color: {colors['text_secondary']};">{value:.1f}</span>
        </div>
        <div style="
            width: 100%; 
            height: 6px; 
            background-color: {colors['surface']}; 
            border-radius: 3px; 
            overflow: hidden;
        ">
            <div style="
                width: {percentage}%; 
                height: 100%; 
                background: linear-gradient(90deg, {colors['primary']} 0%, {colors['primary_light']} 100%);
                border-radius: 3px;
                transition: width 0.3s ease;
            "></div>
        </div>
    </div>
    """


def create_vs_divider() -> str:
    """Create a VS divider between songs."""
    colors = get_spotify_theme_colors()
    
    return f"""
    <div style="
        text-align: center; 
        margin: 2rem 0; 
        position: relative;
    ">
        <div style="
            display: inline-block;
            padding: 1rem 2rem;
            background: linear-gradient(135deg, {colors['primary']} 0%, {colors['primary_light']} 100%);
            color: white;
            border-radius: 50px;
            font-size: 1.5rem;
            font-weight: 700;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
            box-shadow: 0 4px 15px rgba(29, 185, 84, 0.3);
            animation: pulse 3s infinite;
        ">
            VS
        </div>
    </div>
    """


def show_loading_spinner(message: str = "Loading...") -> None:
    """Display a loading spinner with message."""
    st.markdown(f"""
    <div style="text-align: center; padding: 2rem;">
        <div class="loading-spinner"></div>
        <p style="margin-top: 1rem; color: #6C757D; font-weight: 500;">{message}</p>
    </div>
    """, unsafe_allow_html=True)


def create_success_message(title: str, message: str) -> str:
    """Create a styled success message."""
    colors = get_spotify_theme_colors()
    
    return f"""
    <div style="
        background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
        border: 1px solid {colors['success']};
        border-left: 5px solid {colors['success']};
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    ">
        <h4 style="
            margin: 0 0 0.5rem 0; 
            color: {colors['success']}; 
            font-weight: 600;
        ">✅ {title}</h4>
        <p style="margin: 0; color: #155724; font-weight: 500;">{message}</p>
    </div>
    """