import os
import cv2
import numpy as np
import pytesseract
from PIL import Image
import re
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from tabulate import tabulate
import json
from typing import List, Dict, Any, Tuple

# Check if running in a typical desktop environment or server
try:
    # For desktop environments
    pytesseract.pytesseract.tesseract_cmd = pytesseract.pytesseract.tesseract_cmd
except Exception:
    # For server environments, specify path
    # Modify this path as needed for your environment
    pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'

# Data directory structure
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOADS_DIR = os.path.join(BASE_DIR, 'uploads')
PROCESSED_DIR = os.path.join(BASE_DIR, 'processed')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')

# Ensure directories exist
for directory in [UPLOADS_DIR, PROCESSED_DIR, RESULTS_DIR]:
    if not os.path.exists(directory):
        os.makedirs(directory)

class BettingSlipAnalyzer:
    """Analyzes betting slips from images using OCR and historical data."""
    
    def __init__(self, api_client):
        """
        Initialize the analyzer with the API client.
        
        Args:
            api_client: Football API client for retrieving historical data
        """
        self.api_client = api_client
        # Common betting types and their variations in text
        self.bet_types = {
            'BTTS': ['btts', 'both teams to score', 'both to score', 'gole oba tima'],
            'Over 2.5': ['over 2.5', 'over 2,5', 'preko 2.5', 'o2.5', 'o 2.5'],
            'Under 2.5': ['under 2.5', 'under 2,5', 'ispod 2.5', 'u2.5', 'u 2.5'],
            'Home Win': ['1', 'home', 'home win', 'pobeda domaćin', 'pobjeda domacin'],
            'Away Win': ['2', 'away', 'away win', 'pobeda gost', 'pobjeda gost'],
            'Draw': ['x', 'draw', 'tie', 'remi', 'nereseno'],
            'Double Chance 1X': ['1x', 'home draw', 'double chance 1x'],
            'Double Chance X2': ['x2', 'away draw', 'double chance x2'],
            'First Half Over 0.5': ['fh over 0.5', 'first half over 0.5', '1h o0.5'],
            'Clean Sheet': ['clean sheet', 'cs', 'without conceding']
        }
        
        # Team name variations (common misspellings or short names)
        self.team_variations = {
            'arsenal': ['arsenal', 'ars', 'arsenal fc', 'the gunners'],
            'manchester united': ['manchester united', 'man united', 'man utd', 'united', 'mu'],
            'manchester city': ['manchester city', 'man city', 'city', 'man c', 'mc'],
            'liverpool': ['liverpool', 'liv', 'liverpool fc', 'the reds'],
            'chelsea': ['chelsea', 'che', 'chelsea fc', 'the blues'],
            'tottenham': ['tottenham', 'spurs', 'tottenham hotspur', 'tot'],
            # Add more teams as needed
        }
    
    def preprocess_image(self, image_path: str) -> np.ndarray:
        """
        Preprocess the image to enhance OCR accuracy.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Preprocessed image as numpy array
        """
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not open image at {image_path}")
        
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply thresholding
        _, threshold = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Noise removal and enhancement
        kernel = np.ones((1, 1), np.uint8)
        threshold = cv2.dilate(threshold, kernel, iterations=1)
        threshold = cv2.erode(threshold, kernel, iterations=1)
        
        # Apply adaptive thresholding
        adaptive_threshold = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        
        # Save the preprocessed image
        preprocessed_path = os.path.join(
            PROCESSED_DIR, 
            f"preprocessed_{os.path.basename(image_path)}"
        )
        cv2.imwrite(preprocessed_path, threshold)
        
        return threshold
    
    def extract_text_from_image(self, image_path: str) -> str:
        """
        Extract text from betting slip image using OCR.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Extracted text as string
        """
        # Preprocess the image
        preprocessed = self.preprocess_image(image_path)
        
        # Extract text using Tesseract
        config = '--psm 6 --oem 3'  # Page segmentation mode 6: Assume single uniform block of text
        text = pytesseract.image_to_string(preprocessed, config=config)
        
        # Save extracted text for debugging
        text_file_path = os.path.join(
            PROCESSED_DIR, 
            f"text_{os.path.basename(image_path).split('.')[0]}.txt"
        )
        with open(text_file_path, 'w', encoding='utf-8') as f:
            f.write(text)
            
        return text
    
    def parse_betting_slip(self, text: str) -> List[Dict[str, Any]]:
        """
        Parse the extracted text to identify matches and bet types.
        
        Args:
            text: Extracted text from the betting slip
            
        Returns:
            List of dictionaries containing match details and bet types
        """
        # Split into lines
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        
        selections = []
        current_match = None
        current_bet_type = None
        
        for line in lines:
            line = line.lower()  # Convert to lowercase for easier matching
            
            # Check for team names (this is simplistic and might need refinement)
            teams_found = self._find_teams(line)
            if teams_found:
                # Found possible team names, likely a match
                home_team, away_team = teams_found
                current_match = f"{home_team} vs {away_team}"
            
            # Check for bet types
            bet_type_found = self._find_bet_type(line)
            if bet_type_found:
                current_bet_type = bet_type_found
            
            # If we have both a match and a bet type, add a selection
            if current_match and current_bet_type:
                selections.append({
                    'match': current_match,
                    'bet_type': current_bet_type,
                    'original_text': line
                })
                # Reset for next selection
                current_match = None
                current_bet_type = None
        
        # Secondary pass to match orphaned bet types or matches
        # This helps when layout isn't consistent
        if not selections:
            for i, line in enumerate(lines):
                if i < len(lines) - 1:
                    combined = f"{line} {lines[i+1]}".lower()
                    teams_found = self._find_teams(combined)
                    bet_type_found = self._find_bet_type(combined)
                    
                    if teams_found and bet_type_found:
                        home_team, away_team = teams_found
                        selections.append({
                            'match': f"{home_team} vs {away_team}",
                            'bet_type': bet_type_found,
                            'original_text': combined
                        })
        
        return selections
    
    def _find_teams(self, text: str) -> Tuple[str, str] or None:
        """
        Find team names in text.
        This is a simplified implementation that would need to be enhanced
        with more robust team name recognition.
        
        Args:
            text: Text to search for team names
            
        Returns:
            Tuple of (home_team, away_team) if found, None otherwise
        """
        # Check for common team naming patterns
        vs_pattern = re.compile(r'([a-z\s]+)\s+(?:vs|v|-)\.?\s+([a-z\s]+)')
        match = vs_pattern.search(text)
        
        if match:
            home = match.group(1).strip()
            away = match.group(2).strip()
            return self._normalize_team_name(home), self._normalize_team_name(away)
            
        return None
    
    def _normalize_team_name(self, team_name: str) -> str:
        """
        Normalize team name to standard form.
        
        Args:
            team_name: Team name to normalize
            
        Returns:
            Normalized team name
        """
        team_name = team_name.lower().strip()
        
        for standard_name, variations in self.team_variations.items():
            if team_name in variations:
                return standard_name
                
        return team_name  # Return as is if no match found
    
    def _find_bet_type(self, text: str) -> str or None:
        """
        Find betting type in text.
        
        Args:
            text: Text to search for bet type
            
        Returns:
            Standardized bet type name if found, None otherwise
        """
        text = text.lower()
        
        for bet_type, variations in self.bet_types.items():
            for variation in variations:
                if variation in text:
                    return bet_type
                    
        return None
    
    def analyze_selections(self, selections: List[Dict[str, Any]], lookback: int = 10) -> Dict[str, Any]:
        """
        Analyze betting selections against historical data.
        
        Args:
            selections: List of betting selections
            lookback: Number of past matches to look at
            
        Returns:
            Dictionary with analysis results
        """
        results = {
            'selections': selections,
            'analyses': [],
            'summary': {
                'total_selections': len(selections),
                'successful_selections': 0,
                'unsuccessful_selections': 0,
                'avg_success_rate': 0
            },
            'generated': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        for selection in selections:
            match_text = selection['match']
            bet_type = selection['bet_type']
            
            # Extract team names
            match_parts = match_text.split(' vs ')
            if len(match_parts) != 2:
                # Try with 'v' instead of 'vs'
                match_parts = match_text.split(' v ')
            
            if len(match_parts) != 2:
                analysis = {
                    'match': match_text,
                    'bet_type': bet_type,
                    'error': 'Could not parse team names',
                    'success_rate': None,
                    'historical_matches': []
                }
            else:
                home_team, away_team = match_parts
                
                # Get historical data
                historical_matches = self._get_historical_data(home_team, away_team, lookback)
                
                # Calculate success rate for this bet type
                success_count = 0
                for match in historical_matches:
                    if self._bet_would_win(match, bet_type):
                        success_count += 1
                
                success_rate = success_count / len(historical_matches) if historical_matches else 0
                
                analysis = {
                    'match': match_text,
                    'bet_type': bet_type,
                    'success_rate': success_rate,
                    'historical_matches': historical_matches
                }
                
                # Update summary stats
                if success_rate >= 0.6:  # 60% success rate threshold
                    results['summary']['successful_selections'] += 1
                else:
                    results['summary']['unsuccessful_selections'] += 1
            
            results['analyses'].append(analysis)
        
        # Calculate overall success rate
        if results['summary']['total_selections'] > 0:
            results['summary']['avg_success_rate'] = (
                results['summary']['successful_selections'] / 
                results['summary']['total_selections']
            )
        
        return results
    
    def _get_historical_data(self, home_team: str, away_team: str, count: int) -> List[Dict[str, Any]]:
        """
        Get historical match data for the teams.
        
        Args:
            home_team: Home team name
            away_team: Away team name
            count: Number of matches to retrieve
            
        Returns:
            List of historical match data
        """
        try:
            # Try to get specific head-to-head data
            h2h_matches = self._get_h2h_matches(home_team, away_team, count)
            if h2h_matches:
                return h2h_matches
                
            # Get historical matches for each team
            home_matches = self._get_team_matches(home_team, count // 2)
            away_matches = self._get_team_matches(away_team, count // 2)
            
            return home_matches + away_matches
            
        except Exception as e:
            print(f"Error getting historical data: {e}")
            return []
    
    def _get_h2h_matches(self, home_team: str, away_team: str, count: int) -> List[Dict[str, Any]]:
        """
        Get head-to-head match history.
        
        Args:
            home_team: Home team name
            away_team: Away team name
            count: Number of matches to retrieve
            
        Returns:
            List of historical match data
        """
        # This would call the API client's method to get head-to-head data
        # For now, we'll return an empty list
        return []
    
    def _get_team_matches(self, team_name: str, count: int) -> List[Dict[str, Any]]:
        """
        Get historical matches for a team.
        
        Args:
            team_name: Team name
            count: Number of matches to retrieve
            
        Returns:
            List of historical match data
        """
        # Normalize team name
        team_name = self._normalize_team_name(team_name)
        
        # Use the API client to get team fixture data
        # The implementation depends on your specific API client
        fixtures = []
        
        try:
            # First try to find the team ID by name
            team_id = None
            if hasattr(self.api_client, 'get_team_id'):
                team_id = self.api_client.get_team_id(team_name)
            
            if team_id and hasattr(self.api_client, 'get_team_fixtures'):
                fixtures = self.api_client.get_team_fixtures(team_id, count)
                
            return fixtures
        except Exception as e:
            print(f"Error retrieving fixtures for {team_name}: {e}")
            return []
    
    def _bet_would_win(self, match: Dict[str, Any], bet_type: str) -> bool:
        """
        Determine if a bet would win based on match result.
        
        Args:
            match: Match data
            bet_type: Type of bet
            
        Returns:
            True if bet would win, False otherwise
        """
        # Extract relevant match data
        try:
            # These fields depend on your API data structure
            home_score = match.get('score', {}).get('home', 0)
            away_score = match.get('score', {}).get('away', 0)
            total_goals = home_score + away_score
            
            # Check against bet type
            if bet_type == 'BTTS':
                return home_score > 0 and away_score > 0
            elif bet_type == 'Over 2.5':
                return total_goals > 2.5
            elif bet_type == 'Under 2.5':
                return total_goals < 2.5
            elif bet_type == 'Home Win':
                return home_score > away_score
            elif bet_type == 'Away Win':
                return away_score > home_score
            elif bet_type == 'Draw':
                return home_score == away_score
            elif bet_type == 'Double Chance 1X':
                return home_score >= away_score
            elif bet_type == 'Double Chance X2':
                return away_score >= home_score
            elif bet_type == 'Clean Sheet':
                return home_score == 0 or away_score == 0
                
            return False
            
        except Exception as e:
            print(f"Error evaluating bet: {e}")
            return False
    
    def generate_report(self, analysis_results: Dict[str, Any]) -> Dict[str, str]:
        """
        Generate analysis reports in multiple formats.
        
        Args:
            analysis_results: Results from analyze_selections
            
        Returns:
            Dictionary with report paths
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_name = f"betting_analysis_{timestamp}"
        
        # Generate text report
        text_path = os.path.join(RESULTS_DIR, f"{report_name}.txt")
        self._generate_text_report(analysis_results, text_path)
        
        # Generate HTML report
        html_path = os.path.join(RESULTS_DIR, f"{report_name}.html")
        self._generate_html_report(analysis_results, html_path)
        
        # Generate JSON report
        json_path = os.path.join(RESULTS_DIR, f"{report_name}.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(analysis_results, f, indent=2)
        
        # Generate visualization
        viz_path = os.path.join(RESULTS_DIR, f"{report_name}_chart.png")
        self._generate_visualization(analysis_results, viz_path)
        
        return {
            'text': text_path,
            'html': html_path,
            'json': json_path,
            'viz': viz_path
        }
    
    def _generate_text_report(self, results: Dict[str, Any], output_path: str):
        """
        Generate a text report of the analysis results.
        
        Args:
            results: Analysis results
            output_path: Path to save the report
        """
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("===== BETTING SLIP ANALYSIS =====\n")
            f.write(f"Generated: {results['generated']}\n\n")
            
            f.write("===== SUMMARY =====\n")
            summary = results['summary']
            f.write(f"Total Selections: {summary['total_selections']}\n")
            f.write(f"Successful Selections: {summary['successful_selections']}\n")
            f.write(f"Unsuccessful Selections: {summary['unsuccessful_selections']}\n")
            f.write(f"Average Success Rate: {summary['avg_success_rate']:.1%}\n\n")
            
            f.write("===== SELECTION ANALYSES =====\n")
            for i, analysis in enumerate(results['analyses'], 1):
                f.write(f"Selection {i}: {analysis['match']} - {analysis['bet_type']}\n")
                
                if 'error' in analysis:
                    f.write(f"  Error: {analysis['error']}\n\n")
                    continue
                    
                f.write(f"  Success Rate: {analysis['success_rate']:.1%}\n")
                f.write("  Historical Matches:\n")
                
                for match in analysis['historical_matches']:
                    date = match.get('date', 'Unknown')
                    result = f"{match.get('score', {}).get('home', 0)}-{match.get('score', {}).get('away', 0)}"
                    opponent = match.get('opponent', 'Unknown')
                    f.write(f"    {date}: {opponent} {result}\n")
                
                f.write("\n")
    
    def _generate_html_report(self, results: Dict[str, Any], output_path: str):
        """
        Generate an HTML report of the analysis results.
        
        Args:
            results: Analysis results
            output_path: Path to save the report
        """
        html = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Betting Slip Analysis</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                h1, h2 { color: #2c3e50; }
                .summary { background-color: #f8f9fa; padding: 15px; border-radius: 5px; }
                .selection { margin-bottom: 20px; border: 1px solid #ddd; padding: 15px; border-radius: 5px; }
                .good { background-color: #d4edda; }
                .bad { background-color: #f8d7da; }
                table { border-collapse: collapse; width: 100%; }
                th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                th { background-color: #f2f2f2; }
            </style>
        </head>
        <body>
            <h1>Betting Slip Analysis</h1>
            <p>Generated: {generated}</p>
            
            <div class="summary">
                <h2>Summary</h2>
                <p>Total Selections: {total_selections}</p>
                <p>Successful Selections: {successful_selections}</p>
                <p>Unsuccessful Selections: {unsuccessful_selections}</p>
                <p>Average Success Rate: {avg_success_rate:.1%}</p>
            </div>
            
            <h2>Selection Analyses</h2>
            {selection_analyses}
            
            <p><em>Analysis powered by Football Stats API</em></p>
        </body>
        </html>
        """.format(
            generated=results['generated'],
            total_selections=results['summary']['total_selections'],
            successful_selections=results['summary']['successful_selections'],
            unsuccessful_selections=results['summary']['unsuccessful_selections'],
            avg_success_rate=results['summary']['avg_success_rate'],
            selection_analyses=self._generate_html_selections(results['analyses'])
        )
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html)
    
    def _generate_html_selections(self, analyses: List[Dict[str, Any]]) -> str:
        """
        Generate HTML content for selection analyses.
        
        Args:
            analyses: List of selection analysis results
            
        Returns:
            HTML content as string
        """
        html = ""
        
        for i, analysis in enumerate(analyses, 1):
            # Determine if it's a good or bad selection
            css_class = ""
            if 'error' not in analysis and analysis['success_rate'] is not None:
                css_class = "good" if analysis['success_rate'] >= 0.6 else "bad"
            
            html += f'<div class="selection {css_class}">'
            html += f'<h3>Selection {i}: {analysis["match"]} - {analysis["bet_type"]}</h3>'
            
            if 'error' in analysis:
                html += f'<p>Error: {analysis["error"]}</p>'
            else:
                html += f'<p>Success Rate: {analysis["success_rate"]:.1%}</p>'
                
                # Add historical matches table
                if analysis['historical_matches']:
                    html += '<table>'
                    html += '<tr><th>Date</th><th>Opponent</th><th>Result</th><th>Bet Result</th></tr>'
                    
                    for match in analysis['historical_matches']:
                        date = match.get('date', 'Unknown')
                        result = f"{match.get('score', {}).get('home', 0)}-{match.get('score', {}).get('away', 0)}"
                        opponent = match.get('opponent', 'Unknown')
                        
                        # Check if bet would have won
                        bet_result = "Win" if self._bet_would_win(match, analysis["bet_type"]) else "Loss"
                        bet_class = "good" if bet_result == "Win" else "bad"
                        
                        html += f'<tr>'
                        html += f'<td>{date}</td>'
                        html += f'<td>{opponent}</td>'
                        html += f'<td>{result}</td>'
                        html += f'<td class="{bet_class}">{bet_result}</td>'
                        html += f'</tr>'
                        
                    html += '</table>'
                else:
                    html += '<p>No historical data available</p>'
            
            html += '</div>'
            
        return html
    
    def _generate_visualization(self, results: Dict[str, Any], output_path: str):
        """
        Generate visualization of the analysis results.
        
        Args:
            results: Analysis results
            output_path: Path to save the visualization
        """
        plt.figure(figsize=(10, 8))
        
        # Extract success rates for each selection
        success_rates = []
        labels = []
        
        for analysis in results['analyses']:
            if 'error' not in analysis and analysis['success_rate'] is not None:
                success_rates.append(analysis['success_rate'])
                labels.append(f"{analysis['match']}\n{analysis['bet_type']}")
        
        if not success_rates:
            # No data to visualize
            plt.text(0.5, 0.5, "No data to visualize", 
                    horizontalalignment='center', verticalalignment='center',
                    transform=plt.gca().transAxes)
        else:
            # Create bar chart of success rates
            colors = ['green' if rate >= 0.6 else 'red' for rate in success_rates]
            bars = plt.bar(range(len(success_rates)), success_rates, color=colors)
            
            # Add labels and formatting
            plt.xlabel('Selection')
            plt.ylabel('Success Rate')
            plt.title('Betting Selections Success Rates')
            plt.xticks(range(len(labels)), labels, rotation=45, ha='right')
            plt.tight_layout()
            plt.ylim(0, 1.0)
            
            # Add a threshold line at 60%
            plt.axhline(y=0.6, color='black', linestyle='--', alpha=0.7)
            plt.text(len(success_rates) - 1, 0.61, '60% Threshold', va='bottom', ha='right')
            
            # Add success rate values above bars
            for i, bar in enumerate(bars):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05, 
                        f"{success_rates[i]:.1%}", 
                        ha='center', va='bottom')
        
        # Save the figure
        plt.savefig(output_path, bbox_inches='tight')
        plt.close()
    
    def process_slip(self, image_path: str, lookback: int = 10) -> Dict[str, str]:
        """
        Process a betting slip image and generate reports.
        
        Args:
            image_path: Path to the betting slip image
            lookback: Number of past matches to analyze
            
        Returns:
            Dictionary with report paths
        """
        # Extract text from image
        text = self.extract_text_from_image(image_path)
        
        # Parse betting slip to get selections
        selections = self.parse_betting_slip(text)
        
        # Analyze selections
        analysis_results = self.analyze_selections(selections, lookback)
        
        # Generate reports
        reports = self.generate_report(analysis_results)
        
        return reports