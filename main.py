import os
#import magic
import urllib.request
import pandas as pd
from app import app
from flask import Flask, flash, request, redirect, render_template, url_for, session
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
        time = request.form['time']

        if file.filename == '':
            flash('No file selected for uploading')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            session['filename']=filename
            session['time']=time
            #flash('File successfully uploaded')
            #outp=classification_task(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            #flash(outp)
            #return redirect(url_for(".running",filename=filename))
            return redirect('/running')
        else:
            flash('Allowed file types are: {}'.format(str(ALLOWED_EXTENSIONS )))
            return redirect(request.url)


@app.route('/running')
def running():
    return render_template('running.html')

@app.route('/run_optimize')
def run_optimize():
    filename=session.get('filename', 'not set')
    time=int(session.get('time', 'not set'))
    results=classification_task(os.path.join(app.config['UPLOAD_FOLDER'], filename),time)
    #flash(results)
    #session['results']=results
    df=pd.DataFrame(data=results[1]).sort_values(by="rank_test_scores")

    return render_template("results.html",column_names=df.columns.values, row_data=list(df.values.tolist()),zip=zip)


app.run(host='0.0.0.0', port=8080,debug=True)



