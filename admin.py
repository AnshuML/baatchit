from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from utils.pdf_processor import process_pdfs
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from utils.llm_handler import embed_fn as embed_text
import google.generativeai as genai
import os
# Initialize Google Embedding Model
embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=os.getenv("GOOGLE_API_KEY"))

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'pdf'}
MAX_FILE_SIZE = 200 * 1024 * 1024  # 200MB

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def home():
    files = [f for f in os.listdir(UPLOAD_FOLDER) if allowed_file(f)]
    return render_template('admin.html', files=files, max_size=MAX_FILE_SIZE)

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        
        # NEW: Process the uploaded file
        process_and_index(filename)
        
        return jsonify({'message': 'File uploaded and processed successfully'}), 200

    return jsonify({'error': 'Invalid file type'}), 400

def process_and_index(filename):
    pdf_path = os.path.join(UPLOAD_FOLDER, filename)
    texts = process_pdfs([pdf_path])
    
    # Generate embeddings and index with ChromaDB
    embeddings = embedding_model.embed_documents(texts)
    Chroma.from_texts(
        texts=texts,
        embedding=embedding_model,
        persist_directory='db'
    )

@app.route('/delete/<filename>', methods=['DELETE'])
def delete_file(filename):
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if os.path.exists(file_path):
        os.remove(file_path)
        return jsonify({'message': 'File deleted'}), 200
    return jsonify({'error': 'File not found'}), 404

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)
