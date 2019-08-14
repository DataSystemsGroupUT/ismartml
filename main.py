import os
#import magic
import urllib.request
import pandas as pd
from app import app
from flask import Flask, flash, request, redirect, render_template, url_for, session
from werkzeug.utils import secure_filename
import shutil
import pickle
from multi import run_task
from extras import format_time
from extract import get_meta
from predict_meta import predict_meta
tmp_folder = 'tmp/autosk_tmp'
output_folder = 'tmp/autosk_out'



#ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])
ALLOWED_EXTENSIONS = set(["npy","csv"])

CLASSIFIERS=["adaboost","bernoulli_nb","decision_tree", "extra_trees","gaussian_nb", "gradient_boosting","k_nearest_neighbors", "lda","liblinear_svc","libsvm_svc","multinomial_nb","passive_aggressive","qda","random_forest","sgd","xgradient_boosting"]

REGRESSORS=["adaboost","ard_regression","decision_tree", "extra_trees","gaussian_process", "gradient_boosting","k_nearest_neighbors","liblinear_svr","libsvm_svr","random_forest","sgd","xgradient_boosting"]

PREPROCESSORS_CL=["no_preprocessing","extra_trees_preproc_for_classification","fast_ica","feature_agglomeration","kernel_pca","kitchen_sinks","liblinear_svc_preprocessor","nystroem_sampler","pca","polynomial","random_trees_embedding","select_percentile_classification","select_percentile_regression","select_rates","truncatedSVD"]

PREPROCESSORS_RG=["no_preprocessing","extra_trees_preproc_for_regression","fast_ica","feature_agglomeration","kernel_pca","kitchen_sinks","liblinear_svc_preprocessor","nystroem_sampler","pca","polynomial","random_trees_embedding","select_percentile_classification","select_percentile_regression","select_rates","truncatedSVD"]


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS



@app.route('/')
def start():
    return render_template("index.html")

@app.route('/', methods=['POST'])
def start_p():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        #task = request.form['task']
        data_type = request.form['data_type']
        task = request.form['task']
        

        if file.filename == '':
            flash('No file selected for uploading')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

            for dir_ in [tmp_folder, output_folder]:
                try:
                    shutil.rmtree(dir_)
                except OSError:
                    pass
            rec=None
            if task=="classification":
                meta=get_meta(os.path.join(app.config['UPLOAD_FOLDER'], filename),data_type)
                rec=predict_meta(meta[1:])[0][0]
            session["filename"]=filename
            session["data_type"]=data_type
            session["rec"]=rec
            session["task"]=task
            return redirect('/params')
            #os.path.join(app.config['UPLOAD_FOLDER'], values["filename"])
            
            #return str(predict_meta(meta[1:]))

     


        else:
            flash('Allowed file types are: {}'.format(str(ALLOWED_EXTENSIONS )))
            return redirect(request.url)





@app.route('/params')
def params():
    rec=session.get("rec","not set")
    task=session.get("task","not set")
    if task=="classification":
        ESTIMATORS=CLASSIFIERS
        PREPROCESSORS=PREPROCESSORS_CL
    else:
        ESTIMATORS=REGRESSORS
        PREPROCESSORS=PREPROCESSORS_RG
    return render_template('upload.html', ESTIMATORS=ESTIMATORS,PREPROCESSORS=PREPROCESSORS, REC=rec)

@app.route('/params', methods=['POST'])
def params_p():
    if request.method == 'POST':
        # check if the post request has the file part
        values={}
        time = request.form['time']
        period = request.form['period']
        data_type=session.get('data_type', 'not set')
        filename=session.get("filename","not set")
        task=session.get("task","not set")
        search_space= request.form.getlist("estim_ls")
        prep_space= request.form.getlist("prep_ls")

        if(int(time)<30):
            return "Time budget must be at least 30 seconds"
        if(int(period)<30):
            return "Update period must be at least 30 seconds"
        if(int(period)>int(time)):
            return "Update period can't be larger than total time budget"

        values['filename']=filename
        values['time']=int(time)
        values['period']=int(period)
        values['task']=task
        values['data_type']=data_type
        values["search_space"]=search_space
        values["prep_space"]=prep_space
        session["values"]=values

        print("do this")
        return redirect('/running')


@app.route('/running')
def running():
    values=session.get('values', 'not set')
    iters=values["time"]//values["period"]
    extra=values["time"]%values["period"]
    format_period=format_time(values["period"])
    return render_template('running.html',turn=0,task=values["task"],time=values["time"],iters=iters,PERIOD=format_period,RAW_PERIOD=values["period"])

@app.route('/progress')
def progress():
    turn = request.args.get('iter', default = 0, type = int)
    print("turn",turn)
    values=session.get('values', 'not set')
    iters=values["time"]//values["period"]
    extra=values["time"]%values["period"]
    format_period=format_time(values["period"])
    
    estimator=run_task(os.path.join(app.config['UPLOAD_FOLDER'], values["filename"]),values["task"],values["data_type"])
    results=estimator(turn,values["period"],values["search_space"],values["prep_space"])
    df=pd.DataFrame(data=results).sort_values(by="rank_test_scores")
    col_names=["Score","Estimator","Preprocessing","Details"]
    res_list = [[a,b]for a, b in zip(df["mean_test_score"].values.tolist(),df["params"].values.tolist())]
    #session["results"]=res_list
    filehandler = open("tmp/results.p", 'wb') 
    pickle.dump(res_list, filehandler)
    
    turn+=+1
    if(values["task"]=="classification"):
        res_list=[[row[0],row[1]["classifier:__choice__"],row[1]["preprocessor:__choice__"],"view"] for row in res_list]
    else:
        res_list=[[row[0],row[1]["regressor:__choice__"],row[1]["preprocessor:__choice__"],"view"] for row in res_list]
    if(turn>=iters):
        return render_template("results.html",column_names=col_names, row_data=res_list,zip=zip)
    else:
        return render_template("progress.html",turn=turn,iters=iters,PERIOD=format_period,RAW_PERIOD=values["period"], task=values["task"],time=values["time"],column_names=col_names, row_data=res_list,zip=zip)


@app.route('/stop')
def stop():
    values=session.get('values', 'not set')
    filehandler = open("tmp/results.p", 'rb') 
    res_list=pickle.load(filehandler)
    col_names=["Score","Estimator","Preprocessing","Details"]
    if(values["task"]=="classification"):
        res_list=[[row[0],row[1]["classifier:__choice__"],row[1]["preprocessor:__choice__"],"view"] for row in res_list]
    else:
        res_list=[[row[0],row[1]["regressor:__choice__"],row[1]["preprocessor:__choice__"],"view"] for row in res_list]
    return render_template("results.html",column_names=col_names, row_data=res_list,zip=zip)


@app.route('/model')
def view_model():
    #res_list=session.get('results', 'not set')
    filehandler = open("tmp/results.p", 'rb') 
    res_list=pickle.load(filehandler)
    
    index = request.args.get('model', default = 0, type = int)
    model=res_list[index]
    return render_template("model.html",model=model)

app.run(host='0.0.0.0', port=8080,debug=True)



