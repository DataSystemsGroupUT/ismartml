import os
#import magic
import urllib.request
import pandas as pd
from app import app
from flask import Flask, flash, request, redirect, render_template, url_for, session
from werkzeug.utils import secure_filename

from multi import run_task


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
        period = request.form['period']
        task = request.form['task']

        if file.filename == '':
            flash('No file selected for uploading')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            session['filename']=filename
            session['time']=time
            session['period']=period
            session['task']=task
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
    time=session.get('time', 'not set')
    task=session.get('task', 'not set')
    return render_template('running.html',task=task,time=time)

@app.route('/run_optimize')
def run_optimize():
    session["turn"]=0
    filename=session.get('filename', 'not set')
    time=int(session.get('time', 'not set'))
    period=int(session.get('period', 'not set'))
    task=session.get('task', 'not set')
    iters=time//period
    extra=time%period
    estimator=run_task(os.path.join(app.config['UPLOAD_FOLDER'], filename),task)
    results=estimator(0,time)
    
    session["iters"]=iters
    session["extra"]=extra
    
    
    #flash(results)
    #session['results']=results
    df=pd.DataFrame(data=results).sort_values(by="rank_test_scores")
    col_names=["score","params"]
    res_list = [[a,b]for a, b in zip(df["mean_test_score"].values.tolist(),df["params"].values.tolist())]
    #df=pd.DataFrame(data=results)
    #return render_template("results.html",column_names=df.columns.values, row_data=list(df.values.tolist()),zip=zip)
    
    return render_template("progress.html",task=task,time=time,column_names=col_names, row_data=res_list,zip=zip)
    #return render_template('progress.html',task=task,time=time)
    #return render_template("results.html",column_names=col_names, row_data=res_list,zip=zip)


@app.route('/progress')
def progress():
    turn=int(session.get('turn', 'not set'))
    time=int(session.get('time', 'not set'))
    task=session.get('task', 'not set')
    iters=int(session.get('iters', 'not set'))
    extra=int(session.get('extra', 'not set'))
    filename=session.get('filename', 'not set')
    session["turn"]=turn+1
    print("turn: ", turn)
    print("iter: ", iters)
    print("tunr > iter: ",turn>iters)
    estimator=run_task(os.path.join(app.config['UPLOAD_FOLDER'], filename),task)
    results=estimator(turn,time)
    df=pd.DataFrame(data=results).sort_values(by="rank_test_scores")
    col_names=["score","params"]
    res_list = [[a,b]for a, b in zip(df["mean_test_score"].values.tolist(),df["params"].values.tolist())]
    if(turn>iters):
        print("tick")
        return render_template("results.html",column_names=col_names, row_data=res_list,zip=zip)
    else:
        return render_template("progress.html",task=task,time=time,column_names=col_names, row_data=res_list,zip=zip)


app.run(host='0.0.0.0', port=8080,debug=True)



