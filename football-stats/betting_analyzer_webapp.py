import os
import sys
import uuid
import shutil
from flask import Flask, request, render_template, redirect, url_for, send_from_directory
from werkzeug.utils import secure_filename
from football_stats_free import FootballAPIClient
from slip_analyzer import BettingSlipAnalyzer

app = Flask(__name__)

# Configuration
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
app.config['PROCESSED_FOLDER'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'processed')
app.config['RESULTS_FOLDER'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif'}
app.config['MAX_CONTENT_LENGTH'] = 8 * 1024 * 1024  # 8MB max upload size

# Ensure required directories exist
for folder in [app.config['UPLOAD_FOLDER'], app.config['PROCESSED_FOLDER'], app.config['RESULTS_FOLDER']]:
    if not os.path.exists(folder):
        os.makedirs(folder)

# Create static and templates directories if they don't exist
static_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static')
templates_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'templates')
for folder in [static_folder, templates_folder]:
    if not os.path.exists(folder):
        os.makedirs(folder)

# Create basic templates
index_html = """
<!DOCTYPE html>
<html>
<head>
    <title>Betting Slip Analyzer</title>
    <link rel="stylesheet" href="{{ url_for('static', filename='style.css') }}">
</head>
<body>
    <div class="container">
        <h1>Betting Slip Analyzer</h1>
        <p>Upload a photo of your betting slip to analyze historical performance of your selections</p>
        
        {% if error %}
        <div class="error">{{ error }}</div>
        {% endif %}
        
        <form action="{{ url_for('upload_slip') }}" method="post" enctype="multipart/form-data">
            <div class="form-group">
                <label for="slip_image">Betting Slip Image:</label>
                <input type="file" id="slip_image" name="slip_image" accept="image/*" required>
            </div>
            
            <div class="form-group">
                <label for="lookback">Historical Games to Analyze:</label>
                <select id="lookback" name="lookback">
                    <option value="5">Last 5 matches</option>
                    <option value="10" selected>Last 10 matches</option>
                    <option value="15">Last 15 matches</option>
                    <option value="20">Last 20 matches</option>
                </select>
            </div>
            
            <button type="submit">Analyze Slip</button>
        </form>
        
        <div class="footer">
            <p>Powered by Football Stats API</p>
        </div>
    </div>
</body>
</html>
"""

results_html = """
<!DOCTYPE html>
<html>
<head>
    <title>Betting Analysis Results</title>
    <link rel="stylesheet" href="{{ url_for('static', filename='style.css') }}">
</head>
<body>
    <div class="container">
        <h1>Betting Slip Analysis Results</h1>
        
        <div class="summary-box">
            <h2>Summary</h2>
            <p>Total Selections: {{ summary.total_selections }}</p>
            <p>Successful Selections: {{ summary.successful_selections }}</p>
            <p>Unsuccessful Selections: {{ summary.unsuccessful_selections }}</p>
            <p>Average Success Rate: {{ "%.1f"|format(summary.avg_success_rate * 100) }}%</p>
        </div>
        
        <h2>Original Slip</h2>
        <div class="slip-image">
            <img src="{{ url_for('uploaded_file', filename=uploaded_file) }}" alt="Uploaded betting slip">
        </div>
        
        <h2>Analysis Results</h2>
        {% if analyses %}
        {% for analysis in analyses %}
        <div class="selection-box {% if analysis.success_rate >= 0.6 %}good{% else %}bad{% endif %}">
            <h3>{{ analysis.match }} - {{ analysis.bet_type }}</h3>
            {% if analysis.error %}
            <p>Error: {{ analysis.error }}</p>
            {% else %}
            <p>Success Rate: {{ "%.1f"|format(analysis.success_rate * 100) }}%</p>
            
            {% if analysis.historical_matches %}
            <table>
                <tr>
                    <th>Date</th>
                    <th>Opponent</th>
                    <th>Result</th>
                    <th>Bet Result</th>
                </tr>
                {% for match in analysis.historical_matches %}
                <tr>
                    <td>{{ match.date }}</td>
                    <td>{{ match.opponent }}</td>
                    <td>{{ match.score.home }}-{{ match.score.away }}</td>
                    <td class="{% if analysis.bet_would_win(match) %}win{% else %}loss{% endif %}">
                        {{ "Win" if analysis.bet_would_win(match) else "Loss" }}
                    </td>
                </tr>
                {% endfor %}
            </table>
            {% else %}
            <p>No historical match data available</p>
            {% endif %}
            {% endif %}
        </div>
        {% endfor %}
        {% else %}
        <p>No analysis results available.</p>
        {% endif %}
        
        <div class="visualization">
            <h2>Visualization</h2>
            <img src="{{ url_for('uploaded_file', filename=viz_image) }}" alt="Analysis visualization">
        </div>
        
        <div class="actions">
            <a href="{{ url_for('index') }}" class="button">Analyze Another Slip</a>
        </div>
        
        <div class="footer">
            <p>Powered by Football Stats API</p>
        </div>
    </div>
</body>
</html>
"""

# Create CSS
css_content = """
body {
    font-family: Arial, sans-serif;
    margin: 0;
    padding: 0;
    background-color: #f4f7f9;
    color: #333;
}

.container {
    max-width: 900px;
    margin: 0 auto;
    padding: 20px;
}

h1, h2 {
    color: #2c3e50;
}

h1 {
    text-align: center;
    margin-bottom: 30px;
}

.form-group {
    margin-bottom: 20px;
}

label {
    display: block;
    margin-bottom: 5px;
    font-weight: bold;
}

input[type="file"], select {
    width: 100%;
    padding: 10px;
    border: 1px solid #ddd;
    border-radius: 4px;
}

button, .button {
    background-color: #3498db;
    color: white;
    border: none;
    padding: 12px 20px;
    cursor: pointer;
    font-size: 16px;
    border-radius: 4px;
    display: inline-block;
    text-decoration: none;
    text-align: center;
}

button:hover, .button:hover {
    background-color: #2980b9;
}

.summary-box {
    background-color: #f8f9fa;
    padding: 15px;
    border-radius: 5px;
    margin-bottom: 20px;
    border-left: 5px solid #3498db;
}

.selection-box {
    margin-bottom: 20px;
    padding: 15px;
    border-radius: 5px;
    border: 1px solid #ddd;
}

.good {
    border-left: 5px solid #2ecc71;
    background-color: #d4edda;
}

.bad {
    border-left: 5px solid #e74c3c;
    background-color: #f8d7da;
}

.error {
    color: #721c24;
    background-color: #f8d7da;
    border: 1px solid #f5c6cb;
    padding: 10px;
    margin-bottom: 20px;
    border-radius: 4px;
}

table {
    width: 100%;
    border-collapse: collapse;
    margin: 15px 0;
}

th, td {
    padding: 10px;
    border: 1px solid #ddd;
    text-align: left;
}

th {
    background-color: #f2f2f2;
}

.win {
    color: #2ecc71;
    font-weight: bold;
}

.loss {
    color: #e74c3c;
}

.visualization img, .slip-image img {
    max-width: 100%;
    border: 1px solid #ddd;
    border-radius: 4px;
}

.actions {
    margin: 30px 0;
    text-align: center;
}

.footer {
    margin-top: 40px;
    text-align: center;
    color: #7f8c8d;
    font-size: 14px;
}
"""

# Create template files if they don't exist
with open(os.path.join(templates_folder, 'index.html'), 'w') as f:
    f.write(index_html)
    
with open(os.path.join(templates_folder, 'results.html'), 'w') as f:
    f.write(results_html)
    
with open(os.path.join(static_folder, 'style.css'), 'w') as f:
    f.write(css_content)

def allowed_file(filename):
    """Check if filename has an allowed extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/')
def index():
    """Render the index page."""
    return render_template('index.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    """Serve uploaded files."""
    # Check if the file is a processed one
    if filename.startswith('preprocessed_'):
        return send_from_directory(app.config['PROCESSED_FOLDER'], filename)
    elif filename.endswith('_chart.png'):
        return send_from_directory(app.config['RESULTS_FOLDER'], filename)
    else:
        return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/analyze', methods=['POST'])
def upload_slip():
    """Handle file upload and analysis."""
    # Check if file was uploaded
    if 'slip_image' not in request.files:
        return render_template('index.html', error='No file selected')
    
    file = request.files['slip_image']
    
    # Check if file was selected
    if file.filename == '':
        return render_template('index.html', error='No file selected')
    
    # Check if file is allowed
    if not allowed_file(file.filename):
        return render_template('index.html', error='Invalid file type. Only jpg, jpeg, png and gif allowed')
    
    # Get lookback parameter
    lookback = int(request.form.get('lookback', 10))
    
    # Save the file with a secure name
    unique_filename = str(uuid.uuid4()) + '_' + secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
    file.save(file_path)
    
    # Initialize API client
    api_client = FootballAPIClient()
    
    # Initialize analyzer
    analyzer = BettingSlipAnalyzer(api_client)
    
    try:
        # Extract text from image
        text = analyzer.extract_text_from_image(file_path)
        
        # Parse betting slip to get selections
        selections = analyzer.parse_betting_slip(text)
        
        # Analyze selections
        analysis_results = analyzer.analyze_selections(selections, lookback)
        
        # Generate reports
        reports = analyzer.generate_report(analysis_results)
        
        # Get the visualization image name (the chart)
        viz_image = os.path.basename(reports['viz'])
        
        # Add a helper function for the template to determine if a bet would win
        for analysis in analysis_results['analyses']:
            analysis['bet_would_win'] = lambda match, bet_type=analysis['bet_type']: analyzer._bet_would_win(match, bet_type)
        
        return render_template(
            'results.html',
            summary=analysis_results['summary'],
            analyses=analysis_results['analyses'],
            uploaded_file=unique_filename,
            viz_image=viz_image
        )
    
    except Exception as e:
        return render_template('index.html', error=f'Error during analysis: {str(e)}')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)