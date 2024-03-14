from flask import Flask, request, render_template, redirect, url_for
import os
import subprocess
import logging
from logging.handlers import RotatingFileHandler

app = Flask(__name__)
UPLOAD_FOLDER = "/Users/deniskim/Library/CloudStorage/SynologyDrive-M1/문서/연구/DIAL/code/home/tako/minseok/dataset/"
ALLOWED_EXTENSIONS = {'pdf'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('DatasetProcessor')
handler = RotatingFileHandler('log.txt', maxBytes=10000, backupCount=1)
logger.addHandler(handler)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET'])
def index():
    pdf_files = [f for f in os.listdir(UPLOAD_FOLDER) if allowed_file(f)]
    logger.info('Index page accessed, listed PDF files.')
    return render_template('index.html', pdf_files=pdf_files)

@app.route('/process', methods=['POST'])
def process_dataset():
    dataset = request.form['dataset']
    if dataset == 'pdf':
        # Check if a new file was uploaded
        file = request.files['file']
        if file and file.filename != '' and allowed_file(file.filename):
            filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filename)
            logger.info(f'PDF file uploaded and saved: {filename}')
            subprocess.run(['python', 'preprocess.py', '--dataset', 'pdf', '--filepath', filename])
        else:
            # Handle the case where an existing PDF is selected
            existing_pdf = request.form.get('existing_pdf')
            if existing_pdf and allowed_file(existing_pdf):
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], existing_pdf)
                if os.path.exists(filepath):
                    logger.info(f'Existing PDF selected for processing: {filepath}')
                    subprocess.run(['python', 'preprocess.py', '--dataset', 'pdf', '--filepath', filepath])
                else:
                    logger.warning(f'Attempted to process non-existent PDF: {filepath}')
    else:
        logger.info(f'Processing dataset: {dataset}')
        subprocess.run(['python', 'preprocess.py', '--dataset', dataset])
    return redirect(url_for('index'))


if __name__ == '__main__':
    app.run(debug=True)
