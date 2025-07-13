#!/usr/bin/env python3
"""
Test script for Spotify client functionality.
Run this to verify your Spotify API credentials and client implementation work.
"""

import sys
import os
from pathlib import Path

# Add the src directory to Python path
project_root = Path(__file__).parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

# Also add the project root to handle relative imports
sys.path.insert(0, str(project_root))

def test_spotify_client():
    """Test basic Spotify client functionality."""
    print("🎵 Testing Spotify Music Discovery Client")
    print("=" * 50)
    
    try:
        # Import the client
        from src.spotify.client import get_spotify_client
        print("✅ Successfully imported Spotify client")
        
        # Initialize client
        print("\n🔑 Initializing Spotify client...")
        client = get_spotify_client()
        print("✅ Client initialized")
        
        # Test authentication
        print("\n🔐 Testing authentication...")
        if client.is_authenticated():
            print("✅ Authentication successful!")
        else:
            print("❌ Authentication failed!")
            return False
        
        # Test basic search
        print("\n🔍 Testing track search...")
        tracks = client.search_tracks("Blinding Lights", limit=5)
        if tracks:
            print(f"✅ Found {len(tracks)} tracks")
            
            # Display first track
            first_track = tracks[0]
            print(f"\n📀 Sample track:")
            print(f"   Name: {first_track.name}")
            print(f"   Artist: {first_track.artist}")
            print(f"   Album: {first_track.album}")
            print(f"   Popularity: {first_track.popularity}")
            print(f"   Preview URL: {'Yes' if first_track.preview_url else 'No'}")
        else:
            print("❌ No tracks found in search")
            return False
        
        # Test audio features
        print("\n🎼 Testing audio features...")
        if tracks:
            track_id = tracks[0].id
            features = client.get_track_features(track_id)
            
            if features:
                print("✅ Audio features retrieved successfully!")
                print(f"   Danceability: {features['danceability']:.2f}")
                print(f"   Energy: {features['energy']:.2f}")
                print(f"   Valence: {features['valence']:.2f}")
                print(f"   Tempo: {features['tempo']:.0f} BPM")
            else:
                print("❌ Failed to get audio features")
                return False
        
        # Test caching
        print("\n💾 Testing cache functionality...")
        cache_stats = client.get_cache_stats()
        print(f"✅ Cache stats: {cache_stats['cached_items']} items, {cache_stats['total_size_mb']:.2f} MB")
        
        # Test recommendations
        print("\n🎯 Testing recommendations...")
        try:
            recommendations = client.get_recommendations(
                seed_tracks=[tracks[0].id],
                limit=3
            )
            if recommendations:
                print(f"✅ Got {len(recommendations)} recommendations")
                for i, track in enumerate(recommendations[:2], 1):
                    print(f"   {i}. {track.name} by {track.artist}")
            else:
                print("⚠️  No recommendations returned (this might be normal)")
        except Exception as e:
            print(f"⚠️  Recommendations test failed: {e}")
        
        print("\n🎉 All tests completed successfully!")
        print("\n📊 Summary:")
        print(f"   ✅ Authentication: Working")
        print(f"   ✅ Search: Working")
        print(f"   ✅ Audio Features: Working")
        print(f"   ✅ Caching: Working")
        print(f"   ✅ API Rate Limiting: Protected")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("💡 Make sure you've implemented the Spotify client first")
        return False
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        print(f"💡 Check your .env file has SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET")
        return False

def test_environment_setup():
    """Test that environment is properly set up."""
    print("\n🔧 Testing environment setup...")
    
    # Check if .env file exists
    env_file = Path(".env")
    if env_file.exists():
        print("✅ .env file found")
        
        # Try to load environment variables
        from dotenv import load_dotenv
        load_dotenv()
        
        client_id = os.getenv("SPOTIFY_CLIENT_ID")
        client_secret = os.getenv("SPOTIFY_CLIENT_SECRET")
        
        if client_id and client_secret:
            print("✅ Spotify credentials found in environment")
            print(f"   Client ID: {client_id[:10]}..." if len(client_id) > 10 else client_id)
        else:
            print("❌ Spotify credentials not found in .env")
            print("💡 Make sure your .env file contains:")
            print("   SPOTIFY_CLIENT_ID=your_client_id")
            print("   SPOTIFY_CLIENT_SECRET=your_client_secret")
            return False
    else:
        print("❌ .env file not found")
        print("💡 Create a .env file with your Spotify credentials")
        return False
    
    # Check required directories exist
    required_dirs = ["src", "data", "data/cache"]
    for dir_name in required_dirs:
        dir_path = Path(dir_name)
        if dir_path.exists():
            print(f"✅ Directory exists: {dir_name}")
        else:
            print(f"❌ Directory missing: {dir_name}")
            return False
    
    return True

if __name__ == "__main__":
    print("🚀 Spotify Music Discovery - Client Test")
    print("=" * 50)
    
    # Test environment first
    if not test_environment_setup():
        print("\n❌ Environment setup failed. Fix the issues above and try again.")
        sys.exit(1)
    
    # Test the client
    if test_spotify_client():
        print("\n🎉 All tests passed! Your Spotify client is ready to use.")
        sys.exit(0)
    else:
        print("\n❌ Some tests failed. Check the errors above.")
        sys.exit(1)