import os
#import magic
import urllib.request
from app import app
from flask import Flask, flash, request, redirect, render_template
from werkzeug.utils import secure_filename

from classify import classification_task


#ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])
ALLOWED_EXTENSIONS = set(["npy"])

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
    
@app.route('/')
def upload_form():
    return render_template('upload.html')

@app.route('/', methods=['POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No file selected for uploading')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            #flash('File successfully uploaded')
            outp=classification_task(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            flash(outp)
            return redirect('/')
        else:
            flash('Allowed file types are: {}'.format(str(ALLOWED_EXTENSIONS )))
            return redirect(request.url)



if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080,debug=True)
