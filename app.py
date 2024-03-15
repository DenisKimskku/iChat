import sys
print(sys.version)
from flask import Flask, request, render_template, redirect, url_for, session
import os
import subprocess
import logging
from logging.handlers import RotatingFileHandler
import threading
import interface_alpha

app = Flask(__name__)
UPLOAD_FOLDER = "/Users/deniskim/Library/CloudStorage/SynologyDrive-M1/문서/연구/DIAL/code/home/tako/minseok/dataset/"

ALLOWED_EXTENSIONS = {'pdf'}


app.secret_key = os.urandom(24)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('DatasetProcessor')
handler = RotatingFileHandler('log.txt', maxBytes=10000, backupCount=1)
logger.addHandler(handler)

gradio_interface_launched = False

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET'])
def index():
    pdf_files = [f for f in os.listdir(UPLOAD_FOLDER) if allowed_file(f)]
    logger.info('Index page accessed, listed PDF files.')
    return render_template('index.html', pdf_files=pdf_files)

@app.route('/process', methods=['POST'])
def process_dataset():
    dataset = request.form['dataset']
    if dataset == 'pdf':
        file = request.files['file']
        if file and file.filename != '' and allowed_file(file.filename):
            filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filename)
            logger.info(f'PDF file uploaded and saved: {filename}')
            subprocess.run(['python', 'preprocess.py', '--dataset', 'pdf', '--filepath', filename])
            session['pdf_file'] = file.filename.split(".")[0]
        else:
            existing_pdf = request.form.get('existing_pdf')
            if existing_pdf and allowed_file(existing_pdf):
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], existing_pdf)
                if os.path.exists(filepath):
                    logger.info(f'Existing PDF selected for processing: {filepath}')
                    subprocess.run(['python', 'preprocess.py', '--dataset', 'pdf', '--filepath', filepath])
                    session['pdf_file'] = existing_pdf.split(".")[0]
                else:
                    logger.warning(f'Attempted to process non-existent PDF: {filepath}')
        #save the name of file in the session, and send it to the chat_interface
        #split the last .pdf from the filename. No os.splittext, that doesn't exist
    else:
        logger.info(f'Processing dataset: {dataset}')
        session['pdf_file'] = dataset
        
        subprocess.run(['python', 'preprocess.py', '--dataset', dataset])
    return redirect(url_for('chat_interface'))

@app.route('/skip_preprocessing')
def skip_preprocessing():
    return redirect(url_for('chat_interface'))
import asyncio
@app.route('/chat_interface')
def chat_interface():
    global gradio_interface_launched
    if not gradio_interface_launched:
        file_name = session.get('pdf_file', 'wikipedia')
        thread = threading.Thread(target=launch_gradio_interface, args=(file_name,), daemon=True)
        thread.start()
        gradio_interface_launched = True
    return render_template('redirect.html')


def launch_gradio_interface(name):
    async def async_launch():
        iface = interface_alpha.setup_gradio_interface(name)
        await iface.launch(server_name="0.0.0.0", server_port=7860)
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(async_launch())


if __name__ == '__main__':
    app.run(debug=True)