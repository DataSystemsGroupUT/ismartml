import os
#import magic
import urllib.request
import pandas as pd
from app import app
from flask import Flask, flash, request, redirect, render_template, url_for, session
from werkzeug.utils import secure_filename
import shutil
from multi import run_task
from extras import format_time
tmp_folder = 'tmp/autosk_tmp'
output_folder = 'tmp/autosk_out'



#ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])
ALLOWED_EXTENSIONS = set(["npy","csv"])

CLASSIFIERS=["adaboost","bernoulli_nb","decision_tree", "extra_trees","gaussian_nb", "gradient_boosting","k_nearest_neighbors", "lda","liblinear_svc","libsvm_svc","multinomial_nb","passive_aggressive","qda","random_forest","sgd","xgradient_boosting"]

REGRESSORS=["adaboost","ard_regression","decision_tree", "extra_trees","gaussian_process", "gradient_boosting","k_nearest_neighbors","liblinear_svr","libsvm_svr","random_forest","sgd","xgradient_boosting"]

PREPROCESSORS_CL=["extra_trees_preproc_for_classification","fast_ica","feature_agglomeration","kernel_pca","kitchen_sinks","liblinear_svc_preprocessor","no_preprocessing","nystroem_sampler","pca","polynomial","random_trees_embedding","select_percentile_classification","select_percentile_regression","select_rates","truncatedSVD"]

PREPROCESSORS_RG=["extra_trees_preproc_for_regression","fast_ica","feature_agglomeration","kernel_pca","kitchen_sinks","liblinear_svc_preprocessor","no_preprocessing","nystroem_sampler","pca","polynomial","random_trees_embedding","select_percentile_classification","select_percentile_regression","select_rates","truncatedSVD"]

#PREPROCESSORS_RG=["densifier","extra_trees_preproc_for_classification","extra_trees_preproc_for_regression","fast_ica","feature_agglomeration","kernel_pca","kitchen_sinks","liblinear_svc_preprocessor","no_preprocessing","nystroem_sampler","pca","polynomial","random_trees_embedding","select_percentile_classification","select_percentile_regression","select_rates","truncatedSVD"]

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
    
@app.route('/')
def upload_form():
    return render_template('upload.html', CLASSIFIERS=CLASSIFIERS,REGRESSORS=REGRESSORS,PREPROCESSORS_CL=PREPROCESSORS_CL,PREPROCESSORS_RG=PREPROCESSORS_RG)

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
        data_type = request.form['data_type']
        if(task=="classification"):
            search_space= request.form.getlist("classifier_ls")
            prep_space= request.form.getlist("prep_cl")
        else:
            search_space= request.form.getlist("regressor_ls")
            prep_space= request.form.getlist("prep_rg")

        if(int(time)<30):
            return "Time budget must be at least 30 seconds"
        if(int(period)<30):
            return "Update period must be at least 30 seconds"
        if(int(period)>int(time)):
            return "Update period can't be larger than total time budget"
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
            session['data_type']=data_type
            session["search_space"]=search_space
            session["prep_space"]=prep_space
            #flash('File successfully uploaded')
            #outp=classification_task(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            #flash(outp)
            #return redirect(url_for(".running",filename=filename))

            for dir_ in [tmp_folder, output_folder]:
                try:
                    shutil.rmtree(dir_)
                except OSError:
                    pass

            return redirect('/running')
        else:
            flash('Allowed file types are: {}'.format(str(ALLOWED_EXTENSIONS )))
            return redirect(request.url)


@app.route('/running')
def running():
    time=int(session.get('time', 'not set'))
    period=int(session.get('period', 'not set'))
    iters=time//period
    extra=time%period
    time=session.get('time', 'not set')
    task=session.get('task', 'not set')
    format_period=format_time(period)
    return render_template('running.html',task=task,time=time,iters=iters,PERIOD=format_period)

@app.route('/run_optimize')
def run_optimize():
    turn=0
    session["turn"]=turn
    filename=session.get('filename', 'not set')
    time=int(session.get('time', 'not set'))
    period=int(session.get('period', 'not set'))
    task=session.get('task', 'not set')
    data_type=session.get('data_type', 'not set')
    search_space=session.get('search_space', 'not set')
    prep_space=session.get('prep_space', 'not set')
    iters=time//period
    extra=time%period
    format_period=format_time(period)
    estimator=run_task(os.path.join(app.config['UPLOAD_FOLDER'], filename),task,data_type)
    results=estimator(0,time,search_space,prep_space)
    
    session["iters"]=iters
    session["extra"]=extra
    
    
    #flash(results)
    #session['results']=results
    df=pd.DataFrame(data=results).sort_values(by="rank_test_scores")
    col_names=["Score","Estimator","Preprocessing"]
    res_list = [[a,b]for a, b in zip(df["mean_test_score"].values.tolist(),df["params"].values.tolist())]
    res_list=[[row[0],row[1]["classifier:__choice__"],row[1]["preprocessor:__choice__"]] for row in res_list]
    #res_list=list(map(list, zip(*res_list)))
    #res_list=[res_list[0],res_list[1]["classifier:__choice__"],res_list[1]["preprocessor:__choice__"]]
    
    #df=pd.DataFrame(data=results)
    #return render_template("results.html",column_names=df.columns.values, row_data=list(df.values.tolist()),zip=zip)
    if iters<=1:
        return render_template("results.html",column_names=col_names, row_data=res_list,zip=zip)
    else:
        return render_template("progress.html",turn=turn,iters=iters,PERIOD=format_period,task=task,time=time,column_names=col_names, row_data=res_list,zip=zip)
    #return render_template('progress.html',task=task,time=time)
    #return render_template("results.html",column_names=col_names, row_data=res_list,zip=zip)


@app.route('/progress')
def progress():
    turn=int(session.get('turn', 'not set'))
    time=int(session.get('time', 'not set'))
    task=session.get('task', 'not set')
    iters=int(session.get('iters', 'not set'))
    period=int(session.get('period', 'not set'))
    extra=int(session.get('extra', 'not set'))
    filename=session.get('filename', 'not set')
    data_type=session.get('data_type', 'not set')
    search_space=session.get('search_space', 'not set')
    prep_space=session.get('prep_space', 'not set')
    format_period=format_time(period)
    turn+=1
    session["turn"]=turn
    estimator=run_task(os.path.join(app.config['UPLOAD_FOLDER'], filename),task,data_type)
    results=estimator(turn,time,search_space,prep_space)
    df=pd.DataFrame(data=results).sort_values(by="rank_test_scores")
    col_names=["score","params"]
    res_list = [[a,b]for a, b in zip(df["mean_test_score"].values.tolist(),df["params"].values.tolist())]
    if(turn>=iters-1):
        return render_template("results.html",column_names=col_names, row_data=res_list,zip=zip)
    else:
        return render_template("progress.html",turn=turn,iters=iters,PERIOD=format_period,task=task,time=time,column_names=col_names, row_data=res_list,zip=zip)


@app.route('/test')
def test():
    return render_template('test.html', start_total="Jul 20, 2019 15:30:25")



app.run(host='0.0.0.0', port=8080,debug=True)



