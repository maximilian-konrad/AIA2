from flask import Flask, request, render_template, send_from_directory
import os
import pandas as pd
from datetime import datetime
from src.AIA.pipeline.aia_pipeline import AIA

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'outputs'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Initialize AIA
aia = AIA()

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        files = request.files.getlist('file')
        file_paths = []
        for file in files:
            file_path = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(file_path)
            file_paths.append(file_path)
        
        # Process only the uploaded files
        df_results, _ = aia.process_batch(file_paths)
        
        # Save results with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_filename = f'results_{timestamp}.xlsx'
        output_path = os.path.join(OUTPUT_FOLDER, output_filename)
        df_results.to_excel(output_path, index=False)
        
        return render_template('index.html', download_link=output_filename)
    
    return render_template('index.html', download_link=None)

@app.route('/download/<filename>')
def download_file(filename):
    return send_from_directory(OUTPUT_FOLDER, filename, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True) 