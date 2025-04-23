import os
import sys
import argparse
from datetime import datetime
from football_stats_free import FootballAPIClient
from slip_analyzer import BettingSlipAnalyzer

# Create necessary directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOADS_DIR = os.path.join(BASE_DIR, 'uploads')
PROCESSED_DIR = os.path.join(BASE_DIR, 'processed')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')

# Ensure directories exist
for directory in [UPLOADS_DIR, PROCESSED_DIR, RESULTS_DIR]:
    if not os.path.exists(directory):
        os.makedirs(directory)

def analyze_slip(image_path, lookback=10):
    """
    Analyze a betting slip image and generate reports.
    
    Args:
        image_path: Path to the betting slip image
        lookback: Number of past matches to analyze
        
    Returns:
        Dictionary with report paths
    """
    # Ensure the image file exists
    if not os.path.exists(image_path):
        print(f"Error: Image file not found at {image_path}")
        return None
    
    # Initialize the API client
    print("Initializing Football API client...")
    api_client = FootballAPIClient()
    
    # Initialize and run the betting slip analyzer
    print(f"Processing betting slip image: {os.path.basename(image_path)}")
    analyzer = BettingSlipAnalyzer(api_client)
    
    try:
        # Process the slip and get reports
        reports = analyzer.process_slip(image_path, lookback)
        
        print("\nAnalysis complete! Reports generated at:")
        for report_type, report_path in reports.items():
            print(f"- {report_type.capitalize()} report: {report_path}")
        
        return reports
    except Exception as e:
        print(f"Error analyzing betting slip: {e}")
        return None

def main():
    """Main entry point for the application."""
    parser = argparse.ArgumentParser(description="Betting Slip Analysis Tool")
    parser.add_argument("image_path", help="Path to the betting slip image")
    parser.add_argument("--lookback", type=int, default=10, 
                       help="Number of past matches to analyze (default: 10)")
    
    args = parser.parse_args()
    
    # Run the analysis
    analyze_slip(args.image_path, args.lookback)

if __name__ == "__main__":
    main()