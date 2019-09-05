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
from extras import *
from extract import get_meta
from predict_meta import predict_meta
from utils import *
import matplotlib.pyplot as plt

tmp_folder = 'tmp/autosk_tmp'
output_folder = 'tmp/autosk_out'



#ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])
ALLOWED_EXTENSIONS = set(["npy","csv"])

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def start():
    if not os.path.exists("data/hash_list.txt"):
        os.mknod("data/hash_list.txt")

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
            
            if(data_type=="numpy" and filename[-3:]!="npy"):
                return "Wrong file extension (expected .npy)"
            if(data_type=="csv" and (filename[-3:]!="csv" and filename[-3:]!="CSV")):
                return "Wrong file extension (expected .csv)"

            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            checksum=hash_file(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            
            with open("data/hash_list.txt","r") as f:
                lines=f.readlines()

            if(checksum+"\n" not in lines):
                with open("data/hash_list.txt","a") as f:
                    f.write(checksum+"\n")


            for dir_ in [tmp_folder, output_folder]:
                try:
                    shutil.rmtree(dir_)
                except OSError:
                    pass
            rec=[]
            if task=="classification":
                meta=get_meta(os.path.join(app.config['UPLOAD_FOLDER'], filename),data_type)
                rec=predict_meta(meta[1:])
            session["filename"]=filename
            session["data_type"]=data_type
            session["rec"]=rec
            session["task"]=task
            return redirect('/features')
            #os.path.join(app.config['UPLOAD_FOLDER'], values["filename"])
            
            #return str(predict_meta(meta[1:]))

     


        else:
            flash('Allowed file types are: {}'.format(str(ALLOWED_EXTENSIONS )))
            return redirect(request.url)


@app.route('/features')
def featur_pg():
    values=session.get('values', 'not set')
    path=os.path.join(app.config['UPLOAD_FOLDER'], session.get("filename","not set"))
    #features = request.form.getlist("features_ls")
    features=return_cols(path)
    new_data=select_cols(path,features)
    for i in range(len(features)):
        plt.clf()
        new_data[features[i]].hist()
        plt.savefig("static/images/figs/"+str(i))
    return render_template("features.html", FEATURES=features)

@app.route('/features', methods=['POST'])
def feature_pgr():
    if request.method == 'POST':
        # check if the post request has the file part
        target_ft = request.form['target_ft']
        session["target_ft"]=target_ft
        features = request.form.getlist("features_ls")
        path=os.path.join(app.config['UPLOAD_FOLDER'], session.get("filename","not set"))
        new_data=select_cols(path,features)
        new_data.to_csv(path)
        return redirect('/params')
    




@app.route('/params')
def params():
    rec=session.get("rec","not set")
    task=session.get("task","not set")
    column_names=["Classifier","Score"]
    bolds=[]
    
    #load dataset and get features

    #session["features"]=list(features)
    
    #plt.hist(new_data.iloc[:,-1])
    #plt.savefig("static/images/fig.png")



    ##Configure for Task
    if task=="classification":
        rec=[x for x in rec if x[1]!=0] #remove predicions with 0 score from results
        #get bold indexes for recomended classifiers
        rec_t=list(map(list, zip(*rec)))
        for cl in CLASSIFIERS_DISP:
            if cl in rec_t[0]:
                bolds.append(CLASSIFIERS_DISP.index(cl))
            elif cl[-3:] == "SVC":
                if "SVC" in rec_t[0]:
                    bolds.append(CLASSIFIERS_DISP.index(cl))
        #Get corect lists for this task
        ESTIMATORS=[CLASSIFIERS, CLASSIFIERS_DISP]
        PREPROCESSORS=[PREPROCESSORS_CL, PREPROCESSORS_CL_DISP] 
        METRICS=METRICS_CL_DISP
    else:
        ESTIMATORS=[REGRESSORS, REGRESSORS_DISP]
        PREPROCESSORS=[PREPROCESSORS_RG, PREPROCESSORS_RG_DISP]
        METRICS=METRICS_RG_DISP
    return render_template('upload.html', METRICS=METRICS,ESTIMATORS=ESTIMATORS,PREPROCESSORS=PREPROCESSORS, column_names=column_names,row_data=rec, zip=zip, TASK=task, BOLD_CL=bolds)

@app.route('/params', methods=['POST'])
def params_p():
    if request.method == 'POST':
        # check if the post request has the file part
        
        
        #discard feuatres

        #plt.savefig("tmp/fig.png")


        #
        
        values={}
        data_type=session.get('data_type', 'not set')
        filename=session.get("filename","not set")
        task=session.get("task","not set")
        search_space= request.form.getlist("estim_ls")
        prep_space= request.form.getlist("prep_ls")
        
        metric = request.form['metric']

        if(not search_space):
            return "You must select at least 1 estimator"
        if(not prep_space):
            return "You must select at least 1 preprocessor"


        values['filename']=filename
        values['task']=task
        values['data_type']=data_type
        values["search_space"]=search_space
        values["prep_space"]=prep_space
        values["metric"]=metric
        session["values"]=values

        return redirect('/budget')


@app.route('/budget')
def budget():
    task=session.get("task","not set")
    
    ##Configure for Task
    if task=="classification":
        METRICS=METRICS_CL_DISP
    else:
        METRICS=METRICS_RG_DISP
    return render_template('budget.html', METRICS=METRICS, zip=zip, TASK=task)

@app.route('/budget', methods=['POST'])
def budget_p():
    if request.method == 'POST':
        # check if the post request has the file part
        
        
        
        values=session.get('values', 'not set')
        time = request.form['time']
        period = request.form['period']
        data_type=session.get('data_type', 'not set')
        filename=session.get("filename","not set")
        task=session.get("task","not set")
        
        metric = request.form['metric']

        if(int(time)<30):
            return "Time budget must be at least 30 seconds"
        if(int(period)<30):
            return "Update period must be at least 30 seconds"
        if(int(period)>int(time)):
            return "Update period can't be larger than total time budget"


        values['time']=int(time)
        values['period']=int(period)
        values["metric"]=metric
        session["values"]=values

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
    target_ft=session.get('target_ft', 'not set')
    iters=values["time"]//values["period"]
    extra=values["time"]%values["period"]
    format_period=format_time(values["period"])
    metric=gen_metric(values["task"],values["metric"])
    
    features=return_cols(os.path.join(app.config['UPLOAD_FOLDER'], values["filename"]))
    estimator=run_task(os.path.join(app.config['UPLOAD_FOLDER'], values["filename"]),values["task"],values["data_type"],target_ft)
    results=estimator(turn,values["period"],values["search_space"],values["prep_space"], metric)
    df=pd.DataFrame(data=results).sort_values(by="rank_test_scores")
    col_names=["Score","Estimator","Preprocessing","Details"]
    res_list = [[a,b]for a, b in zip(df["mean_test_score"].values.tolist(),df["params"].values.tolist())]
    #session["results"]=res_list
    filehandler = open("tmp/results.p", 'wb') 
    pickle.dump(res_list, filehandler)
    
    turn+=+1
    if(values["task"]=="classification"):
        res_list=[[row[0], format_ls("cl",row[1]["classifier:__choice__"]),format_ls("cp",row[1]["preprocessor:__choice__"]),"view"] for row in res_list]
    else:
        res_list=[[row[0], format_ls("rg",row[1]["regressor:__choice__"]),format_ls("rp",row[1]["preprocessor:__choice__"]),"view"] for row in res_list]
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
        res_list=[[row[0], format_ls("cl",row[1]["classifier:__choice__"]),format_ls("cp",row[1]["preprocessor:__choice__"]),"view"] for row in res_list]
    else:
        res_list=[[row[0], format_ls("rg",row[1]["regressor:__choice__"]),format_ls("rp",row[1]["preprocessor:__choice__"]),"view"] for row in res_list]
    return render_template("results.html",column_names=col_names, row_data=res_list,zip=zip)


@app.route('/model')
def view_model():
    #res_list=session.get('results', 'not set')
    filehandler = open("tmp/results.p", 'rb') 
    res_list=pickle.load(filehandler)
    
    index = request.args.get('model', default = 0, type = int)
    model=res_list[index]
    return render_template("model.html",model=model)


@app.route("/test")
def test():
    return render_template("test.html")


@app.after_request
def add_header(r):
    r.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    r.headers["Pragma"] = "no-cache"
    r.headers["Expires"] = "0"
    r.headers['Cache-Control'] = 'public, max-age=0'
    return r


app.run(host='0.0.0.0', port=8080,debug=True)



