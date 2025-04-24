import os
import requests
import json
from dotenv import load_dotenv
from datetime import datetime, timedelta
import time
import concurrent.futures

# Load environment variables from .env file
load_dotenv()

# === CONFIGURATION ===
API_KEY = os.getenv("FOOTBALL_API_KEY")
if not API_KEY:
    raise ValueError("Missing API key. Please set FOOTBALL_API_KEY in .env file")

API_HOST = "v3.football.api-sports.io"
LEAGUE_ID = 39        # Premier League
SEASON = 2023         # 2023-24 season
DEBUG = False         # Enable debug output
SLEEP_TIME = 6        # Sleep between requests to avoid rate limiting (10 per minute)

HEADERS = {
    "x-apisports-key": API_KEY,
    "Content-Type": "application/json"
}

# === PLAYER LIST ===
# Arsenal squad - key players
players = {
    "Martin Ødegaard": 1772,     # Updated IDs based on the players endpoint
    "Bukayo Saka": 1161,
    "Gabriel Jesus": 1164,
    "Declan Rice": 2413,
    "Gabriel Martinelli": 25874,
    "William Saliba": 24263,
    "Aaron Ramsdale": 2928,
    "Leandro Trossard": 2335
}

# Mapping of team names to API team IDs
TEAM_IDS = {
    "arsenal": 42,
    "manchester city": 50,
    "manchester united": 33,
    "liverpool": 40,
    "chelsea": 49,
    "tottenham": 47,
    "newcastle": 34,
    "aston villa": 66,
    "west ham": 48,
    "brighton": 51,
    "everton": 45,
    "wolves": 39,
    "crystal palace": 52,
    "brentford": 55,
    "fulham": 36,
    "bournemouth": 35,
    "nottingham forest": 65,
    "luton": 1359,
    "burnley": 44,
    "sheffield united": 62
}

class FootballAPIClient:
    """Client for interacting with the Football API."""
    
    def __init__(self):
        """Initialize the Football API client."""
        self.api_key = API_KEY
        self.api_host = API_HOST
        self.headers = HEADERS
        self.season = SEASON
        self.league_id = LEAGUE_ID
        self.debug = DEBUG
        self.sleep_time = SLEEP_TIME
        self.request_count = 0
        self.last_request_time = None
    
    def debug_print(self, message, data=None, truncate=True):
        """Print debug information if DEBUG is enabled."""
        if self.debug:
            print(f"DEBUG: {message}")
            if data is not None:
                try:
                    if truncate and isinstance(data, dict):
                        json_str = json.dumps(data, indent=2, default=str)
                        print(json_str[:1000] + "..." if len(json_str) > 1000 else json_str)
                    else:
                        print(json.dumps(data, indent=2, default=str) if isinstance(data, dict) else data)
                except:
                    print(f"[Unable to serialize data to JSON: {type(data)}]")
                    print(data)
    
    def _rate_limit_request(self):
        """
        Manage rate limiting to avoid API throttling.
        Free tier is limited to 10 requests per minute.
        """
        now = datetime.now()
        
        if self.last_request_time:
            # Calculate time since last request
            elapsed = (now - self.last_request_time).total_seconds()
            
            # If less than sleep_time seconds have passed, sleep
            if elapsed < self.sleep_time:
                time.sleep(self.sleep_time - elapsed)
        
        self.last_request_time = datetime.now()
        self.request_count += 1
    
    def make_request(self, endpoint, params=None):
        """
        Make an API request with rate limiting.
        
        Args:
            endpoint: API endpoint to request
            params: Dictionary of query parameters
            
        Returns:
            JSON response data
        """
        url = f"https://{self.api_host}/{endpoint}"
        self.debug_print(f"Making request to {url}", params)
        
        # Apply rate limiting
        self._rate_limit_request()
        
        try:
            # Add a timeout to prevent hanging requests
            response = requests.get(url, headers=self.headers, params=params, timeout=10)
            self.debug_print(f"Response status: {response.status_code}")
            
            if response.status_code != 200:
                self.debug_print("Error response", response.text)
                return {"error": f"API returned status code {response.status_code}"}
            
            data = response.json()
            
            # Check for API errors
            if "errors" in data and data["errors"]:
                self.debug_print("API returned errors", data["errors"])
                return {"error": data["errors"]}
            
            return data
            
        except requests.exceptions.Timeout:
            self.debug_print("Request timed out")
            return {"error": "API request timed out. Please try again later."}
        except requests.exceptions.ConnectionError:
            self.debug_print("Connection error")
            return {"error": "Could not connect to the API. Please check your internet connection."}
        except Exception as e:
            self.debug_print(f"Request failed: {str(e)}")
            return {"error": str(e)}
    
    def get_players_info(self, team_id=None):
        """
        Fetch general player information for a team.
        
        Args:
            team_id: Team ID to fetch players for (default: Arsenal)
            
        Returns:
            List of player data
        """
        if team_id is None:
            team_id = 42  # Default to Arsenal
            
        params = {
            "team": team_id,
            "season": self.season,
            "page": 1
        }
        
        data = self.make_request("players", params)
        
        if "error" in data:
            return []
            
        if "response" not in data:
            return []
            
        player_info = []
        for player in data["response"]:
            player_info.append({
                "id": player["player"]["id"],
                "name": player["player"]["name"],
                "age": player["player"]["age"],
                "nationality": player["player"]["nationality"],
                "position": player["statistics"][0]["games"]["position"] if player["statistics"] else "Unknown"
            })
            
        return player_info
    
    def get_team_id(self, team_name):
        """
        Get the API team ID from a team name.
        
        Args:
            team_name: Name of the team
            
        Returns:
            Team ID if found, None otherwise
        """
        # Normalize team name
        team_name = team_name.lower().strip()
        
        # Direct lookup in our mapping
        if team_name in TEAM_IDS:
            return TEAM_IDS[team_name]
            
        # Try to find a partial match
        for known_name, team_id in TEAM_IDS.items():
            if team_name in known_name or known_name in team_name:
                return team_id
                
        # If no match found, try to search in the API
        params = {
            "name": team_name,
            "league": self.league_id,
            "season": self.season
        }
        
        data = self.make_request("teams", params)
        
        if "error" in data:
            return None
            
        if "response" in data and data["response"]:
            return data["response"][0]["team"]["id"]
            
        return None
    
    def get_team_fixtures(self, team_id, limit=10):
        """
        Fetch team fixtures using date parameters.
        
        Args:
            team_id: Team ID to fetch fixtures for
            limit: Maximum number of fixtures to return
            
        Returns:
            List of fixture data
        """
        # Use the actual 2023-24 season date range
        end_date = "2024-05-19"
        start_date = "2024-02-01"  # Get fixtures from February 2024 onward
        
        params = {
            "league": self.league_id,
            "season": self.season,
            "team": team_id,
            "from": start_date,
            "to": end_date
        }
        
        data = self.make_request("fixtures", params)
        
        if "error" in data or "response" not in data:
            return []
            
        # Sort by date (newest first) and limit to requested number
        fixtures = sorted(data["response"], key=lambda x: x["fixture"]["date"], reverse=True)[:limit]
        
        results = []
        for fx in fixtures:
            home_score = fx["goals"]["home"]
            away_score = fx["goals"]["away"]
            home_team = fx["teams"]["home"]["name"]
            away_team = fx["teams"]["away"]["name"]
            
            if home_score is None or away_score is None:  # Skip fixtures without scores (not played yet)
                continue
                
            is_home = fx["teams"]["home"]["id"] == team_id
            opp = away_team if is_home else home_team
            
            results.append({
                "date": fx["fixture"]["date"][:10],
                "opponent": opp,
                "score": {
                    "home": home_score, 
                    "away": away_score
                },
                "is_home": is_home
            })
            
        return results
    
    def get_h2h_fixtures(self, team1_id, team2_id, limit=10):
        """
        Fetch head-to-head fixtures between two teams.
        
        Args:
            team1_id: First team ID
            team2_id: Second team ID
            limit: Maximum number of fixtures to return
            
        Returns:
            List of fixture data
        """
        params = {
            "h2h": f"{team1_id}-{team2_id}",
            "season": self.season
        }
        
        data = self.make_request("fixtures/headtohead", params)
        
        if "error" in data or "response" not in data:
            return []
            
        # Sort by date (newest first) and limit to requested number
        fixtures = sorted(data["response"], key=lambda x: x["fixture"]["date"], reverse=True)[:limit]
        
        results = []
        for fx in fixtures:
            home_score = fx["goals"]["home"]
            away_score = fx["goals"]["away"]
            
            if home_score is None or away_score is None:  # Skip fixtures without scores
                continue
                
            results.append({
                "date": fx["fixture"]["date"][:10],
                "teams": {
                    "home": fx["teams"]["home"]["name"],
                    "away": fx["teams"]["away"]["name"]
                },
                "score": {
                    "home": home_score, 
                    "away": away_score
                }
            })
            
        return results
    
    def get_team_standings(self):
        """
        Get current league standings.
        
        Returns:
            List of team standings
        """
        params = {
            "league": self.league_id,
            "season": self.season
        }
        
        data = self.make_request("standings", params)
        
        if "error" in data or "response" not in data:
            return []
            
        # Extract league standings from the first league
        if not data["response"] or not data["response"][0]["league"]["standings"]:
            return []
            
        standings = data["response"][0]["league"]["standings"][0]
        
        table = []
        for team in standings:
            table.append({
                "position": team["rank"],
                "team": {
                    "id": team["team"]["id"],
                    "name": team["team"]["name"]
                },
                "played": team["all"]["played"],
                "won": team["all"]["win"],
                "drawn": team["all"]["draw"],
                "lost": team["all"]["lose"],
                "points": team["points"],
                "goalDiff": team["goalsDiff"]
            })
            
        return table

    def batch_get_team_fixtures(self, team_ids, limit=10):
        """
        Fetch fixtures for multiple teams in parallel.
        Args:
            team_ids: List of team IDs
            limit: Number of fixtures per team
        Returns:
            Dict mapping team_id to fixtures list
        """
        results = {}
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_to_team = {
                executor.submit(self.get_team_fixtures, team_id, limit): team_id
                for team_id in team_ids
            }
            for future in concurrent.futures.as_completed(future_to_team):
                team_id = future_to_team[future]
                try:
                    results[team_id] = future.result()
                except Exception as e:
                    results[team_id] = []
        return results

    def batch_get_players_info(self, team_ids):
        """
        Fetch player info for multiple teams in parallel.
        Args:
            team_ids: List of team IDs
        Returns:
            Dict mapping team_id to player info list
        """
        results = {}
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_to_team = {
                executor.submit(self.get_players_info, team_id): team_id
                for team_id in team_ids
            }
            for future in concurrent.futures.as_completed(future_to_team):
                team_id = future_to_team[future]
                try:
                    results[team_id] = future.result()
                except Exception as e:
                    results[team_id] = []
        return results

def debug_print(message, data=None, truncate=True):
    """Print debug information if DEBUG is enabled."""
    if DEBUG:
        print(f"DEBUG: {message}")
        if data is not None:
            try:
                if truncate and isinstance(data, dict):
                    json_str = json.dumps(data, indent=2, default=str)
                    print(json_str[:1000] + "..." if len(json_str) > 1000 else json_str)
                else:
                    print(json.dumps(data, indent=2, default=str) if isinstance(data, dict) else data)
            except:
                print(f"[Unable to serialize data to JSON: {type(data)}]")
                print(data)

def get_players_info():
    """Fetch general player information for Arsenal."""
    url = f"https://{API_HOST}/players"
    
    # For Free plan, need to use player search with a team query
    params = {
        "team": 42,         # Arsenal FC ID in this API version
        "season": SEASON,
        "page": 1
    }
    
    debug_print("Fetching Arsenal players info", params)
    
    try:
        resp = requests.get(url, headers=HEADERS, params=params)
        debug_print(f"Response status: {resp.status_code}")
        
        resp.raise_for_status()
        
        data = resp.json()
        if "errors" in data and data["errors"]:
            debug_print("API returned errors", data["errors"])
            print(f"API Error: {data['errors']}")
            return []
        
        if "response" not in data or not data["response"]:
            print("No player data available")
            return []
        
        player_info = []
        for player in data["response"]:
            player_info.append({
                "id": player["player"]["id"],
                "name": player["player"]["name"],
                "age": player["player"]["age"],
                "nationality": player["player"]["nationality"],
                "position": player["statistics"][0]["games"]["position"] if player["statistics"] else "Unknown"
            })
            
        return player_info
            
    except Exception as e:
        print(f"Error fetching players: {e}")
        return []

def get_team_fixtures(team_id, limit=10):
    """Fetch Arsenal fixtures using the date parameter instead of 'last'."""
    url = f"https://{API_HOST}/fixtures"
    
    # Use the actual 2023-24 season date range instead of calculating from current date
    # The 2023-24 Premier League season ended on May 19, 2024
    end_date = "2024-05-19"
    start_date = "2024-02-01"  # Get fixtures from February 2024 onward
    
    params = {
        "league": LEAGUE_ID,
        "season": SEASON,
        "team": team_id,
        "from": start_date,
        "to": end_date
    }
    
    debug_print("Fetching team fixtures with date range", params)
    
    try:
        resp = requests.get(url, headers=HEADERS, params=params)
        debug_print(f"Response status: {resp.status_code}")
        
        resp.raise_for_status()
        
        data = resp.json()
        if "errors" in data and data["errors"]:
            debug_print("API returned errors", data["errors"])
            print(f"API Error: {data['errors']}")
            return []
        
        if "response" not in data or not data["response"]:
            print("No fixture data available")
            return []
            
        # Sort by date (newest first) and limit to requested number
        fixtures = sorted(data["response"], key=lambda x: x["fixture"]["date"], reverse=True)[:limit]
        
        results = []
        for fx in fixtures:
            home = fx["goals"]["home"]
            away = fx["goals"]["away"]
            home_team = fx["teams"]["home"]["name"]
            away_team = fx["teams"]["away"]["name"]
            
            if home is None or away is None:  # Skip fixtures without scores (not played yet)
                continue
                
            opp = away_team if home_team == "Arsenal" else home_team
            results.append({
                "date": fx["fixture"]["date"][:10],
                "opponent": opp,
                "score": f"{home}-{away}",
                "BTTS": "Yes" if home > 0 and away > 0 else "No"
            })
        return results
        
    except Exception as e:
        print(f"Error fetching fixtures: {e}")
        debug_print("Exception", str(e))
        return []

def get_team_standings():
    """Get current Premier League standings."""
    url = f"https://{API_HOST}/standings"
    
    params = {
        "league": LEAGUE_ID,
        "season": SEASON
    }
    
    debug_print("Fetching league standings", params)
    
    try:
        resp = requests.get(url, headers=HEADERS, params=params)
        debug_print(f"Response status: {resp.status_code}")
        
        resp.raise_for_status()
        
        data = resp.json()
        if "errors" in data and data["errors"]:
            debug_print("API returned errors", data["errors"])
            print(f"API Error: {data['errors']}")
            return None
            
        if "response" not in data or not data["response"]:
            print("No standings data available")
            return None
            
        # Extract league standings from the first league
        standings = data["response"][0]["league"]["standings"][0]
        
        table = []
        for team in standings:
            table.append({
                "position": team["rank"],
                "team": team["team"]["name"],
                "played": team["all"]["played"],
                "won": team["all"]["win"],
                "drawn": team["all"]["draw"],
                "lost": team["all"]["lose"],
                "points": team["points"],
                "goalDiff": team["goalsDiff"]
            })
            
        return table
        
    except Exception as e:
        print(f"Error fetching standings: {e}")
        debug_print("Exception", str(e))
        return None

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
        
        if status_resp.status_code == 200:
            if "errors" in status_data and status_data["errors"]:
                print(f"API Status errors: {status_data['errors']}")
            else:
                resp = status_data.get("response", {})
                print(f"Account: {resp.get('account', {}).get('firstname', 'Unknown')} {resp.get('account', {}).get('lastname', 'Unknown')}")
                print(f"Subscription: {resp.get('subscription', {}).get('plan', 'Unknown')}")
                print(f"Requests: {resp.get('requests', {}).get('current', 'Unknown')}/{resp.get('requests', {}).get('limit_day', 'Unknown')} (daily)")
        else:
            print(f"API Status check failed: {status_resp.status_code}")
    except Exception as e:
        print(f"Error checking API status: {e}")
    
    # Test basic API connection
    api_working = test_api_connection()
    
    if not api_working:
        print("\nAPI tests failed. Please check your API key and subscription.")
        print("Stopping execution to avoid unnecessary failed requests.")
        exit(1)
    
    # Get team standings
    print("\n=== Premier League Table ===")
    standings = get_team_standings()
    if standings:
        print(f"{'Pos':<5}{'Team':<25}{'P':<5}{'W':<5}{'D':<5}{'L':<5}{'Pts':<5}{'GD':<5}")
        print("-" * 60)
        for team in standings:
            print(f"{team['position']:<5}{team['team']:<25}{team['played']:<5}{team['won']:<5}{team['drawn']:<5}{team['lost']:<5}{team['points']:<5}{team['goalDiff']:<5}")
    
    # Get Arsenal player info
    print("\n=== Arsenal Squad ===")
    players_info = get_players_info()
    if players_info:
        print(f"{'Name':<30}{'Position':<15}{'Age':<5}{'Nationality':<20}")
        print("-" * 70)
        for player in players_info:
            print(f"{player['name']:<30}{player['position']:<15}{player['age']:<5}{player['nationality']:<20}")
    
    # Get Arsenal fixtures (BTTS)
    print("\n=== Arsenal - Recent Fixtures & BTTS ===")
    arsenal_fixtures = get_team_fixtures(42)  # Arsenal ID in the API
    if arsenal_fixtures:
        print(f"{'Date':<12}{'Opponent':<20}{'Score':<10}{'BTTS':<5}")
        print("-" * 47)
        for match in arsenal_fixtures:
            print(f"{match['date']:<12}{match['opponent']:<20}{match['score']:<10}{match['BTTS']:<5}")
    else:
        print("No fixtures data available")