#!/usr/bin/env python3
"""
Test script for Spotify client functionality.
This version works when run from project root as: python scripts/test_spotify_connection.py
"""

import sys
import os
from pathlib import Path

# Get the actual current working directory (where the command was run)
project_root = Path.cwd()  # This will be the directory you're in when you run the command
src_path = project_root / "src"

# Add src to Python path
sys.path.insert(0, str(src_path))

def test_environment_setup():
    """Test that environment is properly set up."""
    print("🔧 Testing environment setup...")
    
    # Check if .env file exists in project root
    env_file = project_root / ".env"
    if env_file.exists():
        print("✅ .env file found")
        
        # Try to load environment variables
        from dotenv import load_dotenv
        load_dotenv(env_file)
        
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
        print(f"❌ .env file not found at {env_file}")
        print("💡 Create a .env file with your Spotify credentials")
        return False
    
    # Check required directories exist
    required_dirs = ["src", "data", "data/cache"]
    for dir_name in required_dirs:
        dir_path = project_root / dir_name
        if dir_path.exists():
            print(f"✅ Directory exists: {dir_name}")
        else:
            print(f"❌ Directory missing: {dir_name}")
            return False
    
    # Check what's actually in src
    print(f"\n📁 Contents of src directory:")
    src_contents = list(src_path.iterdir()) if src_path.exists() else []
    for item in src_contents:
        if item.is_dir():
            print(f"   📂 {item.name}/")
        else:
            print(f"   📄 {item.name}")
    
    return True

def test_spotify_client():
    """Test basic Spotify client functionality."""
    print("\n🎵 Testing Spotify Music Discovery Client")
    print("=" * 50)
    
    try:
        # First, let's see what's in the spotify directory
        spotify_dirs = [d for d in src_path.iterdir() if d.is_dir() and 'spotify' in d.name.lower()]
        if not spotify_dirs:
            print("❌ No spotify directory found in src/")
            print("Available directories:")
            for d in src_path.iterdir():
                if d.is_dir():
                    print(f"   - {d.name}")
            return False
        
        spotify_dir = spotify_dirs[0]  # Take the first one found
        print(f"📁 Using spotify directory: {spotify_dir.name}")
        
        # Check for client.py
        client_file = spotify_dir / "client.py"
        if not client_file.exists():
            print(f"❌ client.py not found in {spotify_dir}")
            print("Contents of spotify directory:")
            for item in spotify_dir.iterdir():
                print(f"   - {item.name}")
            return False
        
        print(f"✅ Found client.py ({client_file.stat().st_size} bytes)")
        
        # Import the client using the actual directory name
        module_name = spotify_dir.name
        
        if module_name == "spotify":
            from spotify.client import get_spotify_client
        elif module_name == "spotify_code":
            from spotify_code.client import get_spotify_client
        else:
            print(f"❌ Unexpected spotify directory name: {module_name}")
            return False
            
        print(f"✅ Successfully imported from {module_name}")
        
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
        tracks = client.search_tracks("Blinding Lights", limit=3)
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
        
        print("\n🎉 Basic tests completed successfully!")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("💡 Make sure you've implemented the Spotify client code")
        return False
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Spotify Music Discovery - Client Test")
    print(f"Running from: {project_root}")
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