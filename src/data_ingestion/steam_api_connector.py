"""
Steam API Connector
Connects to Steam API for real player data (optional alternative to synthetic data)
"""

import requests
import time
from typing import List, Dict, Optional
import pandas as pd
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SteamAPIConnector:
    """Connector for Steam Web API"""
    
    def __init__(self, api_key: str):
        """
        Initialize Steam API connector
        
        Args:
            api_key: Steam Web API key (get from https://steamcommunity.com/dev/apikey)
        """
        self.api_key = api_key
        self.base_url = "https://api.steampowered.com"
        self.rate_limit_delay = 1  # seconds between requests
        
    def get_player_summaries(self, steam_ids: List[str]) -> List[Dict]:
        """
        Get player summaries for given Steam IDs
        
        Args:
            steam_ids: List of Steam IDs
            
        Returns:
            List of player summary dictionaries
        """
        url = f"{self.base_url}/ISteamUser/GetPlayerSummaries/v0002/"
        
        # API allows max 100 IDs per request
        summaries = []
        
        for i in range(0, len(steam_ids), 100):
            batch = steam_ids[i:i+100]
            params = {
                'key': self.api_key,
                'steamids': ','.join(batch)
            }
            
            try:
                response = requests.get(url, params=params)
                response.raise_for_status()
                data = response.json()
                
                if 'response' in data and 'players' in data['response']:
                    summaries.extend(data['response']['players'])
                
                time.sleep(self.rate_limit_delay)
                
            except Exception as e:
                logger.error(f"Error fetching player summaries: {e}")
        
        return summaries
    
    def get_player_stats(self, steam_id: str, app_id: int) -> Optional[Dict]:
        """
        Get player statistics for a specific game
        
        Args:
            steam_id: Steam ID
            app_id: Steam App ID (game ID)
            
        Returns:
            Player stats dictionary or None
        """
        url = f"{self.base_url}/ISteamUserStats/GetUserStatsForGame/v0002/"
        
        params = {
            'key': self.api_key,
            'steamid': steam_id,
            'appid': app_id
        }
        
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            
            time.sleep(self.rate_limit_delay)
            
            return data.get('playerstats')
            
        except Exception as e:
            logger.error(f"Error fetching player stats for {steam_id}: {e}")
            return None
    
    def get_owned_games(self, steam_id: str) -> List[Dict]:
        """
        Get list of games owned by player
        
        Args:
            steam_id: Steam ID
            
        Returns:
            List of owned games
        """
        url = f"{self.base_url}/IPlayerService/GetOwnedGames/v0001/"
        
        params = {
            'key': self.api_key,
            'steamid': steam_id,
            'include_appinfo': 1,
            'include_played_free_games': 1
        }
        
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            
            time.sleep(self.rate_limit_delay)
            
            if 'response' in data and 'games' in data['response']:
                return data['response']['games']
            
        except Exception as e:
            logger.error(f"Error fetching owned games for {steam_id}: {e}")
        
        return []
    
    def get_friends_list(self, steam_id: str) -> List[Dict]:
        """
        Get friends list for a player
        
        Args:
            steam_id: Steam ID
            
        Returns:
            List of friends
        """
        url = f"{self.base_url}/ISteamUser/GetFriendList/v0001/"
        
        params = {
            'key': self.api_key,
            'steamid': steam_id,
            'relationship': 'friend'
        }
        
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            
            time.sleep(self.rate_limit_delay)
            
            if 'friendslist' in data and 'friends' in data['friendslist']:
                return data['friendslist']['friends']
            
        except Exception as e:
            logger.error(f"Error fetching friends list for {steam_id}: {e}")
        
        return []
    
    def get_recently_played_games(self, steam_id: str) -> List[Dict]:
        """
        Get recently played games for a player
        
        Args:
            steam_id: Steam ID
            
        Returns:
            List of recently played games
        """
        url = f"{self.base_url}/IPlayerService/GetRecentlyPlayedGames/v0001/"
        
        params = {
            'key': self.api_key,
            'steamid': steam_id
        }
        
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            
            time.sleep(self.rate_limit_delay)
            
            if 'response' in data and 'games' in data['response']:
                return data['response']['games']
            
        except Exception as e:
            logger.error(f"Error fetching recently played games for {steam_id}: {e}")
        
        return []
    
    def extract_player_features(self, steam_id: str, app_id: int) -> Dict:
        """
        Extract churn-relevant features from Steam API data
        
        Args:
            steam_id: Steam ID
            app_id: Game App ID
            
        Returns:
            Dictionary of extracted features
        """
        features = {
            'player_id': steam_id,
            'extraction_date': datetime.now().isoformat()
        }
        
        # Get player summary
        summary = self.get_player_summaries([steam_id])
        if summary:
            player = summary[0]
            features['account_created'] = player.get('timecreated')
            features['last_logoff'] = player.get('lastlogoff')
            features['profile_visibility'] = player.get('communityvisibilitystate')
        
        # Get friends
        friends = self.get_friends_list(steam_id)
        features['total_friends'] = len(friends)
        
        # Get owned games
        owned_games = self.get_owned_games(steam_id)
        features['total_games_owned'] = len(owned_games)
        
        # Find target game
        target_game = None
        for game in owned_games:
            if game.get('appid') == app_id:
                target_game = game
                break
        
        if target_game:
            features['playtime_total_mins'] = target_game.get('playtime_forever', 0)
            features['playtime_2weeks_mins'] = target_game.get('playtime_2weeks', 0)
        
        # Get game stats
        stats = self.get_player_stats(steam_id, app_id)
        if stats:
            features['achievements'] = len(stats.get('achievements', []))
            features['stats'] = stats.get('stats', {})
        
        return features
    
    def batch_extract_features(self, steam_ids: List[str], app_id: int) -> pd.DataFrame:
        """
        Extract features for multiple players
        
        Args:
            steam_ids: List of Steam IDs
            app_id: Game App ID
            
        Returns:
            DataFrame with extracted features
        """
        all_features = []
        
        for i, steam_id in enumerate(steam_ids):
            logger.info(f"Extracting features for player {i+1}/{len(steam_ids)}")
            features = self.extract_player_features(steam_id, app_id)
            all_features.append(features)
            
            # Be respectful of API rate limits
            if (i + 1) % 10 == 0:
                time.sleep(5)
        
        return pd.DataFrame(all_features)


if __name__ == "__main__":
    # Example usage
    # NOTE: You need a valid Steam API key to use this
    
    # api_key = "YOUR_STEAM_API_KEY"
    # connector = SteamAPIConnector(api_key)
    
    # Example: Extract features for CS:GO (App ID: 730)
    # steam_ids = ["76561198000000000"]  # Example Steam IDs
    # df = connector.batch_extract_features(steam_ids, app_id=730)
    # print(df.head())
    
    print("Steam API Connector initialized.")
    print("To use: provide a valid Steam API key and uncomment the example code.")
