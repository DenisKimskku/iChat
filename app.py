import sqlite3
from flask import Flask, request, render_template, redirect, url_for, session
import os
import subprocess
import logging
from logging.handlers import RotatingFileHandler
from datetime import datetime
import threading
import interface_alpha
import asyncio
import sys
import multiprocessing
DATABASE_PATH = 'application.db'
UPLOAD_FOLDER = "/root/dataset/"
ALLOWED_EXTENSIONS = {'pdf'}

gradio_interface_instance = None

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = os.urandom(24)

gradio_interface_launched = False

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger('DatasetProcessor')
handler = RotatingFileHandler('log.txt', maxBytes=10000, backupCount=1)
logger.addHandler(handler)

# Database helper functions
def get_db_connection():
    conn = sqlite3.connect('/root/application.db')
    conn.row_factory = sqlite3.Row
    logger.info("Connected!")
    return conn

def get_user_session(username):
    conn = get_db_connection()
    session = conn.execute('SELECT * FROM user_sessions WHERE username = ?', (username,)).fetchone()
    conn.close()
    return session

def update_user_session(username, pdf_file=None):
    conn = get_db_connection()
    user_session = get_user_session(username)
    if user_session:
        conn.execute('UPDATE user_sessions SET pdf_file = ?, last_accessed = ? WHERE username = ?', (pdf_file, datetime.now(), username))
        logger.info("In user session!")
    else:
        conn.execute('INSERT INTO user_sessions (username, pdf_file, last_accessed) VALUES (?, ?, ?)', (username, pdf_file, datetime.now()))
    conn.commit()
    conn.close()
    logger.info(f'{username}, {pdf_file}')  # Corrected logging statement


def clear_user_session(username):
    update_user_session(username, pdf_file=None)

# Flask routes
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        # Implement proper authentication check here
        session['username'] = username
        update_user_session(username)  # Initialize or update session in DB
        return redirect(url_for('index'))
    return render_template('login.html')

@app.route('/logout', methods=['POST'])
def logout():
    username = session.pop('username', None)
    if username:
        clear_user_session(username)  # Clear session data in DB
    return redirect(url_for('login'))

@app.route('/', methods=['GET'])
def index():
    if 'username' not in session:
        return redirect(url_for('login'))
    user_session = get_user_session(session['username'])
    pdf_files = [f for f in os.listdir(UPLOAD_FOLDER) if allowed_file(f)]
    return render_template('index.html', pdf_files=pdf_files, current_file=user_session['pdf_file'] if user_session else None)

@app.route('/reset-session', methods=['POST'])
def reset_session():
    if 'username' in session:
        session.pop('pdf_file', None)  # Reset only the relevant part of the session
    return '', 204  # Return a no content response

@app.route('/process', methods=['POST'])
def process_dataset():
    if 'username' not in session:
        return redirect(url_for('login'))
    username = session['username']
    dataset = request.form['dataset']
    selected_pdf_file = None
    if dataset == 'pdf':
        file = request.files['file']
        if file and file.filename != '' and allowed_file(file.filename):
            filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filename)
            logger.info(f'PDF file uploaded and saved: {filename}')
            selected_pdf_file = file.filename.split(".")[0]
            update_user_session(username, selected_pdf_file)
            subprocess.run([sys.executable, 'preprocess.py', '--dataset', 'pdf', '--filepath', filename])
            
        else:
            existing_pdf = request.form.get('existing_pdf')
            if existing_pdf and allowed_file(existing_pdf):
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], existing_pdf)
                if os.path.exists(filepath):
                    logger.info(f'Existing PDF selected for processing: {filepath}')
                    selected_pdf_file = existing_pdf.split(".")[0]
                    update_user_session(username, selected_pdf_file)
                    subprocess.run([sys.executable, 'preprocess.py', '--dataset', 'pdf', '--filepath', filepath])
    
                else:
                    logger.warning(f'Attempted to process non-existent PDF: {filepath}')
        #save the name of file in the session, and send it to the chat_interface
        #split the last .pdf from the filename. No os.splittext, that doesn't exist
    else:
        logger.info(f'Processing dataset: {dataset}')
        selected_pdf_file = dataset
        update_user_session(username, selected_pdf_file)
        
        subprocess.run([sys.executable, 'preprocess.py', '--dataset', dataset])
    #update_user_session(username, selected_pdf_file)
    return redirect(url_for('chat_interface'))

@app.route('/skip_preprocessing')
def skip_preprocessing():
    return redirect(url_for('chat_interface'))
global gradio_thread
@app.route('/chat_interface')
def chat_interface():
    global gradio_interface_launched
    if 'username' not in session:
        return redirect(url_for('login'))

    username = session['username']
    user_session = get_user_session(username)
    
    # This is now directly fetched from the database reflecting the latest user selection
    file_name_to_launch = user_session['pdf_file'] if user_session else None
    logger.info("In chat interface!")
    logger.info(file_name_to_launch)

    #if not gradio_interface_launched or (gradio_interface_instance and file_name_to_launch):
        # Assuming a mechanism exists to check if the Gradio interface needs to be updated,
        # such as a change in the selected dataset:
    '''
    if gradio_interface_launched:
        if gradio_thread.is_alive():
            logger.info("I'm closing down22!")
            # If needed, implement logic to properly stop or reset the Gradio interface
            gradio_thread.terminate()
    '''
    # Now we ensure the thread starts with the current file name to launch
    #gradio_thread = threading.Thread(target=launch_gradio_interface, args=(file_name_to_launch,), daemon=True)
    for p in multiprocessing.active_children():
        if p.name == str(username):
            p.terminate()
    logger.info("Inside if loop")
    gradio_interface_launched = True
    #gradio_thread = multiprocessing.Process(target=launch_gradio_interface, args=(file_name_to_launch,), name='gradio_t')
    gradio_thread = multiprocessing.Process(target=launch_gradio_interface, args=(file_name_to_launch,), name=str(username))
    gradio_thread.start()
        #gradio_interface_launched = True
    
    return render_template('redirect.html')
async def launch_gradio_interface_async(name):
    global gradio_interface_instance
    # Close the existing Gradio interface if it's running
    if gradio_interface_instance:
        logger.info("I'm closing down22!")
        await gradio_interface_instance.close()
    iface = interface_alpha.setup_gradio_interface(name)
    await iface.launch(server_name="0.0.0.0", server_port=7861, share=False, ssl_verify=False)
    return iface

def launch_gradio_interface(name):
    global gradio_interface_instance

    # Close the existing Gradio interface if it's running
    if gradio_interface_instance:
        logger.info("I'm closing down!")
        asyncio.run(gradio_interface_instance.close())
        
        gradio_interface_instance = None
    

    # Launch the new Gradio interface
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    gradio_interface_instance = loop.run_until_complete(launch_gradio_interface_async(name))



if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
