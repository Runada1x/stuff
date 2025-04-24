import os
import sys
import uuid
import shutil
import logging
from flask import Flask, request, render_template, redirect, url_for, send_from_directory, flash
from werkzeug.utils import secure_filename
from football_stats_free import FootballAPIClient
from slip_analyzer import BettingSlipAnalyzer
import concurrent.futures

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

# Setup logging
log_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'app.log')
logging.basicConfig(
    filename=log_path,
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s'
)

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
    logging.info('--- Received analyze request ---')
    if 'slip_image' not in request.files:
        logging.warning('No file part in request')
        return render_template('index.html', error='No file selected')
    file = request.files['slip_image']
    if file.filename == '':
        logging.warning('No file selected')
        return render_template('index.html', error='No file selected')
    if not allowed_file(file.filename):
        logging.warning('Invalid file type')
        return render_template('index.html', error='Invalid file type. Only jpg, jpeg, png and gif allowed')
    lookback = int(request.form.get('lookback', 10))
    unique_filename = str(uuid.uuid4()) + '_' + secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
    file.save(file_path)
    logging.info(f'File saved to {file_path}')
    api_client = FootballAPIClient()
    analyzer = BettingSlipAnalyzer(api_client)
    try:
        ANALYSIS_TIMEOUT = 30
        def extract_and_analyze():
            logging.info('Starting OCR extraction...')
            text = analyzer.extract_text_from_image(file_path)
            logging.info('OCR extraction done.')
            logging.info('Parsing betting slip...')
            selections = analyzer.parse_betting_slip(text)
            logging.info(f'Parsed selections: {selections}')
            if not selections:
                logging.warning('No valid selections found after parsing.')
                return None, 'No valid selections found in the image. Please upload a clearer slip.'
            logging.info('Analyzing selections...')
            analysis_results = analyzer.analyze_selections(selections, lookback)
            logging.info('Analysis complete.')
            return analysis_results, None
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(extract_and_analyze)
            try:
                analysis_results, error = future.result(timeout=ANALYSIS_TIMEOUT)
                if error:
                    logging.error(f'Analysis error: {error}')
                    return render_template('index.html', error=error)
            except concurrent.futures.TimeoutError:
                logging.error('Analysis timed out.')
                return render_template('index.html', error='Analysis timed out. Please try a different image or try again later.')
        reports = analyzer.generate_report(analysis_results)
        viz_image = os.path.basename(reports['viz'])
        for analysis in analysis_results['analyses']:
            analysis['bet_would_win'] = lambda match, bet_type=analysis['bet_type']: analyzer._bet_would_win(match, bet_type)
        logging.info('Rendering results page.')
        return render_template(
            'results.html',
            summary=analysis_results['summary'],
            analyses=analysis_results['analyses'],
            uploaded_file=unique_filename,
            viz_image=viz_image
        )
    except Exception as e:
        error_message = f'Error during analysis: {str(e)}'
        logging.exception(error_message)
        return render_template('index.html', error='An error occurred during analysis. Please check your image and try again. If the problem persists, contact support.')

@app.route('/batch_analyze', methods=['POST'])
def batch_analyze():
    logging.info('--- Received batch analyze request ---')
    if 'slip_images' not in request.files:
        logging.warning('No files part in request')
        return render_template('index.html', error='No files selected for batch analysis')
    files = request.files.getlist('slip_images')
    if not files or all(f.filename == '' for f in files):
        logging.warning('No files selected')
        return render_template('index.html', error='No files selected for batch analysis')
    lookback = int(request.form.get('lookback', 10)) if 'lookback' in request.form else 10
    api_client = FootballAPIClient()
    analyzer = BettingSlipAnalyzer(api_client)
    results = []
    filenames = []
    def process_one(file):
        unique_filename = str(uuid.uuid4()) + '_' + secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(file_path)
        logging.info(f'Batch: File saved to {file_path}')
        try:
            text = analyzer.extract_text_from_image(file_path)
            selections = analyzer.parse_betting_slip(text)
            if not selections:
                return {'filename': unique_filename, 'error': 'No valid selections found.'}
            analysis_results = analyzer.analyze_selections(selections, lookback)
            report_paths = analyzer.generate_report(analysis_results)
            return {
                'filename': unique_filename,
                'summary': analysis_results['summary'],
                'analyses': analysis_results['analyses'],
                'viz_image': os.path.basename(report_paths['viz'])
            }
        except Exception as e:
            logging.exception(f'Batch: Error processing {file.filename}')
            return {'filename': unique_filename, 'error': str(e)}
    with concurrent.futures.ThreadPoolExecutor() as executor:
        batch_results = list(executor.map(process_one, files))
    return render_template('batch_results.html', batch_results=batch_results)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)