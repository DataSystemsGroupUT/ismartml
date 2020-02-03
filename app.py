from flask import Flask
from key import key

UPLOAD_FOLDER = 'uploads'

app = Flask(__name__)
app.secret_key = key
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

