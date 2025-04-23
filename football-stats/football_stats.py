import os
import requests
import json
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# === CONFIGURATION ===
API_KEY = os.getenv("FOOTBALL_API_KEY")
if not API_KEY:
    raise ValueError("Missing API key. Please set FOOTBALL_API_KEY in .env file")

API_HOST = "v3.football.api-sports.io"
LEAGUE_ID = 39        # Premier League
SEASON = 2023         # 2023-24 season which has complete data
TIMEZONE = "Europe/London"
DEBUG = True          # Enable debug output

HEADERS = {
    "x-apisports-key": API_KEY,
    "Host": API_HOST
}

# === PLAYER LIST ===
# Replace the IDs with the correct ones from your API dashboard or lookup via /players/profiles
players = {
    "Myles Lewis-Skelly": 467918,
    "Ismaïla Sarr": 16185,
    "Eddie Nketiah": 35547,
    "Martin Ødegaard": 20179,
    "Gabriel Martinelli": 31303,
    "Thomas Partey": 19211,
    "Declan Rice": 30197,
    "Jurrien Timber": 49673,
    "Ethan Nwaneri": 700456
}

def debug_print(message, data=None, truncate=True):
    """Print debug information if DEBUG is enabled."""
    if DEBUG:
        print(f"DEBUG: {message}")
        if data is not None:
            try:
                if truncate:
                    json_str = json.dumps(data, indent=2, default=str)
                    print(json_str[:1000] + "..." if len(json_str) > 1000 else json_str)
                else:
                    print(json.dumps(data, indent=2, default=str))
            except:
                print(f"[Unable to serialize data to JSON: {type(data)}]")
                print(data)

def get_player_logs(player_id, count=10):
    """Fetch the last `count` matches for a player and return key stats."""
    url = f"https://{API_HOST}/players"
    params = {
        "league": LEAGUE_ID,
        "season": SEASON,
        "player": player_id,
        "timezone": TIMEZONE,
        "page": 1
    }
    
    debug_print(f"Fetching player data for ID {player_id}", params)
    
    try:
        resp = requests.get(url, headers=HEADERS, params=params)
        debug_print(f"Response status: {resp.status_code}")
        
        resp.raise_for_status()  # Raise exception for 4XX/5XX responses
        
        # Print full raw response for debugging
        debug_print("Full API response", resp.json(), truncate=False)
        
        data = resp.json()
        if "errors" in data and data["errors"]:
            debug_print("API returned errors", data["errors"])
            print(f"API Error for player {player_id}: {data['errors']}")
            return []
        
        if "response" not in data or not data["response"]:
            print(f"No data available for player {player_id}")
            return []
            
        # Sort fixtures by date descending
        fixtures = sorted(data["response"], key=lambda x: x["fixture"]["date"], reverse=True)[:count]
        
        logs = []
        for entry in fixtures:
            fx = entry["fixture"]
            stats = entry["statistics"][0]
            logs.append({
                "date": fx["date"][:10],
                "opponent": stats["team"]["name"],
                "result": f"{fx['score']['fulltime']['home']}-{fx['score']['fulltime']['away']}",
                "minutes": stats["games"]["minutes"],
                "fouls": stats.get("fouls", {}).get("committed", 0),
                "shots": stats.get("shots", {}).get("total", 0),
                "tackles": stats.get("tackles", {}).get("total", 0),
            })
        return logs
        
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data for player {player_id}: {e}")
        debug_print("Request exception", str(e))
        return []
    except Exception as e:
        print(f"Unexpected error processing player {player_id}: {e}")
        debug_print("Exception", str(e))
        return []

def get_arsenal_btss(count=10):
    """Fetch Arsenal's last `count` PL fixtures and compute BTTS."""
    url = f"https://{API_HOST}/fixtures"
    params = {
        "league": LEAGUE_ID,
        "season": SEASON,
        "team": 33,  # Arsenal FC ID
        "last": count,
        "timezone": TIMEZONE
    }
    
    debug_print("Fetching Arsenal fixtures", params)
    
    try:
        resp = requests.get(url, headers=HEADERS, params=params)
        debug_print(f"Response status: {resp.status_code}")
        
        resp.raise_for_status()
        
        # Print full raw response for debugging
        debug_print("Full API response", resp.json(), truncate=False)
        
        data = resp.json()
        if "errors" in data and data["errors"]:
            debug_print("API returned errors", data["errors"])
            print(f"API Error for Arsenal fixtures: {data['errors']}")
            return []
        
        if "response" not in data or not data["response"]:
            print("No fixture data available for Arsenal")
            return []
            
        fixtures = data["response"]
        
        results = []
        for fx in fixtures:
            home = fx["goals"]["home"]
            away = fx["goals"]["away"]
            opp = fx["teams"]["away"]["name"] if fx["teams"]["home"]["id"] == 33 else fx["teams"]["home"]["name"]
            results.append({
                "date": fx["fixture"]["date"][:10],
                "opponent": opp,
                "score": f"{home}-{away}",
                "BTTS": "Yes" if home > 0 and away > 0 else "No"
            })
        return results
        
    except requests.exceptions.RequestException as e:
        print(f"Error fetching Arsenal fixtures: {e}")
        debug_print("Request exception", str(e))
        return []
    except Exception as e:
        print(f"Unexpected error processing Arsenal fixtures: {e}")
        debug_print("Exception", str(e))
        return []

def test_api_connection():
    """Test the API connection with a simple leagues request."""
    print("\n=== Testing API Connection ===")
    url = f"https://{API_HOST}/leagues"
    params = {
        "id": LEAGUE_ID
    }
    
    try:
        debug_print("Testing API with leagues endpoint", params)
        resp = requests.get(url, headers=HEADERS, params=params)
        debug_print(f"Response status: {resp.status_code}")
        
        if resp.status_code != 200:
            print(f"API test failed with status code: {resp.status_code}")
            return False
            
        data = resp.json()
        debug_print("API test response", data, truncate=False)
        
        if "errors" in data and data["errors"]:
            print(f"API returned errors: {data['errors']}")
            return False
            
        if "response" in data and data["response"]:
            if len(data["response"]) > 0:
                league = data["response"][0].get("league", {})
                print(f"Successfully connected to API and found league: {league.get('name', 'Unknown')}")
                return True
                
        print("API connection successful but no league data returned")
        return True
    
    except Exception as e:
        print(f"API test failed with error: {e}")
        debug_print("Exception", str(e))
        return False

if __name__ == "__main__":
    print(f"Using API key: {API_KEY[:5]}...{API_KEY[-3:]}")
    print(f"Looking for data from the {SEASON}-{SEASON+1} season")
    
    # Test API status
    try:
        status_url = f"https://{API_HOST}/status"
        status_resp = requests.get(status_url, headers=HEADERS)
        status_data = status_resp.json()
        
        print("\n=== API Status Check ===")
        debug_print("Status response", status_data, truncate=False)
        
        if status_resp.status_code == 200:
            if "errors" in status_data and status_data["errors"]:
                print(f"API Status errors: {status_data['errors']}")
            else:
                resp = status_data.get("response", {})
                print(f"API Status: {resp.get('status', 'Unknown')}")
                print(f"Account: {resp.get('account', {}).get('name', 'Unknown')}")
                print(f"Requests remaining today: {resp.get('requests', {}).get('remaining', 'Unknown')}")
                print(f"Subscription: {resp.get('subscription', {}).get('name', 'Unknown')}")
        else:
            print(f"API Status check failed: {status_resp.status_code}")
            debug_print("API status response", status_data)
    except Exception as e:
        print(f"Error checking API status: {e}")
        debug_print("Exception", str(e))
    
    # Test basic API connection
    api_working = test_api_connection()
    
    if not api_working:
        print("\nAPI tests failed. Please check your API key and subscription.")
        print("Stopping execution to avoid unnecessary failed requests.")
        exit(1)
    
    # Process player data
    for name, pid in players.items():
        print(f"\n{name} - Last 10 PL Matches:")
        logs = get_player_logs(pid)
        if logs:
            for log in logs:
                print(log)
        else:
            print("No data available")
    
    # Process Arsenal BTTS data
    print("\nArsenal - BTTS Over Last 10 PL Matches:")
    matches = get_arsenal_btss()
    if matches:
        for match in matches:
            print(match)
    else:
        print("No data available")