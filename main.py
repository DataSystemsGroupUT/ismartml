import os
from imblearn.over_sampling import SMOTE
#import magic
import urllib.request
import pandas as pd
from app import app
from flask import Flask, flash, request, redirect, render_template, url_for, session, send_from_directory
from werkzeug.utils import secure_filename
import shutil
import pickle
from multi import run_task, process_data
from extras import *
from extract import get_meta
from predict_meta import predict_meta
from utils_local import *
import matplotlib.pyplot as plt
import pipeline_gen
from sklearn.pipeline import Pipeline
from joblib import dump, load
from nyoka import skl_to_pmml
import numpy as np

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
        values={}
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        #data_type = request.form['data_type']
        data_type="csv"
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
            


            rec=[]
            if task=="classification":
                meta=get_meta(os.path.join(app.config['UPLOAD_FOLDER'], filename),data_type)
                rec=predict_meta(meta[1:])
            session["filename"]=filename
            session["data_type"]=data_type
            session["rec"]=rec
            session["task"]=task
            return redirect('/features')
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
        features.remove(target_ft)
        session["features"]=features
        path=os.path.join(app.config['UPLOAD_FOLDER'], session.get("filename","not set"))
        new_data=select_cols(path,list(features)+[target_ft])
        new_data.to_csv(path,index=False)
        return redirect('/target_class')
    

@app.route('/target_class')
def target_class():
    task=session.get("task","not set")
    ##Configure for Task
    if task=="classification":
        METRICS=METRICS_CL_DISP
    else:
        METRICS=METRICS_RG_DISP
    values=session.get('values', 'not set')
    target_ft=session.get('target_ft', 'not set')
    path=os.path.join(app.config['UPLOAD_FOLDER'], session.get("filename","not set"))
    data=pd.read_csv(path)
    #features = request.form.getlist("features_ls")
    plt.clf()
    data[target_ft].hist()
    plt.savefig("static/images/figs/target")
    ratio=[True if (min(data[target_ft].value_counts())/max(data[target_ft].value_counts()))<0.6 else False][0]
    pre_metric=["F1" if ratio else "Accuracy" ][0]
    return render_template("target.html",ratio=ratio,METRICS=METRICS,pre_metric=pre_metric)

@app.route('/target_class', methods=['POST'])
def target_class_r():
    if request.method == 'POST':
        # check if the post request has the file part
        values=session.get('values', 'not set')
        metric = request.form['metric']
        smote = request.form['smote']
        values["metric"]=metric
        session["values"]=values
        target_ft=session.get('target_ft', 'not set')
        features=session.get('features', 'not set')
        print(features)
        #features.remove(target_ft)
        #feature dropping can be brought here for better perforamnce
        if smote == "yes":
            path=os.path.join(app.config['UPLOAD_FOLDER'], session.get("filename","not set"))
            X,y=process_data(path,"csv",target_ft)
            print("Original:", X.shape, y.shape)
            sm = SMOTE(random_state=42)
            X_res, y_res = sm.fit_resample(X, y)
            print("New:", X_res.shape, y_res.shape)
            new_data=pd.DataFrame(np.column_stack((X_res,y_res)),columns=list(features)+[target_ft])
            new_data.to_csv(path,index=False)
        return redirect('/params')
    

@app.route('/params')
def params():
    rec=session.get("rec","not set")
    task=session.get("task","not set")
    column_names=["Classifier","Score"]
    bolds=[]
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
    else:
        ESTIMATORS=[REGRESSORS, REGRESSORS_DISP]
        PREPROCESSORS=[PREPROCESSORS_RG, PREPROCESSORS_RG_DISP]
    return render_template('upload.html', ESTIMATORS=ESTIMATORS,PREPROCESSORS=PREPROCESSORS, column_names=column_names,row_data=rec, zip=zip, TASK=task, BOLD_CL=bolds)

@app.route('/params', methods=['POST'])
def params_p():
    if request.method == 'POST':
        values=session.get('values', 'not set')
        data_type=session.get('data_type', 'not set')
        filename=session.get("filename","not set")
        task=session.get("task","not set")
        search_space= request.form.getlist("estim_ls")
        prep_space= request.form.getlist("prep_ls")

        if(not search_space):
            return "You must select at least 1 estimator"
        if(not prep_space):
            return "You must select at least 1 preprocessor"

        values['task']=task
        values['data_type']=data_type
        values["search_space"]=search_space
        values["prep_space"]=prep_space
        session["values"]=values

        return redirect('/budget')

@app.route('/budget')
def budget():
    task=session.get("task","not set")
    ##Configure for Task
    return render_template('budget.html',  zip=zip, TASK=task)

@app.route('/budget', methods=['POST'])
def budget_p():
    if request.method == 'POST':
        values=session.get('values', 'not set')
        time = request.form['time']
        period = request.form['period']
        data_type=session.get('data_type', 'not set')
        filename=session.get("filename","not set")
        task=session.get("task","not set")
        reuse = request.form['reuse']

        if(int(time)<30):
            return "Time budget must be at least 30 seconds"
        if(int(period)<30):
            return "Update period must be at least 30 seconds"
        if(int(period)>int(time)):
            return "Update period can't be larger than total time budget"

        values['time']=int(time)
        values['period']=int(period)
        session["values"]=values
        session["reuse"]=reuse

        return redirect('/running')

@app.route('/running')
def running():
    values=session.get('values', 'not set')
    target_ft=session.get('target_ft', 'not set')
    iters=values["time"]//values["period"]
    extra=values["time"]%values["period"]
    format_period=format_time(values["period"])
    reuse=session.get('reuse', 'not set')


    #check dataset checksum and lookup
    path=os.path.join(app.config['UPLOAD_FOLDER'], session.get("filename","not set"))
    checksum=hash_file(path)+"_"+target_ft+"_"+values["task"]+"_"+values["metric"]
    session["checksum"]=checksum
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
    #copy tmp to run on it
    if reuse=="yes":
        if os.path.exists("tmp_runs/{}".format(checksum)):
            shutil.copytree("tmp_runs/{}".format(checksum),"tmp/autosk_tmp")
            #modify space.pcs
            olds=[]
            old_pres=[]
            #path="tmp/autosk_tmp/smac3-output/run_0.OLD/configspace.pcs"
            path="tmp/autosk_tmp/space.pcs"
            with open(path,"r") as f:
                lines=f.readlines()
                "classifier:__choice__ {decision_tree, gradient_boosting, random_forest} [random_forest]"
                pre="classifier:__choice__ {"
                for line in lines:
                    if "classifier:__choice__ {" in line:
                        olds=[ar.strip() for ar in line[len(pre):].split("}")[0].split(",")]
                    elif "preprocessor:__choice__ {" in line:
                        old_pres=[ar.strip() for ar in line[len("preprocessor:__choice__ {"):].split("}")[0].split(",")]

                        

            
            for param in values["search_space"]:
                if param not in olds:
                    olds.append(param)

            for param in values["prep_space"]:
                if param not in old_pres:
                    old_pres.append(param)

            values["search_space"]=olds
            values["prep_space"]=old_pres
            session["values"]=values




    return render_template('running.html',turn=0,task=values["task"],time=values["time"],iters=iters,PERIOD=format_period,RAW_PERIOD=values["period"])

@app.route('/progress')
def progress():
    turn = request.args.get('iter', default = 0, type = int)
    print("turn",turn)
    values=session.get('values', 'not set')
    target_ft=session.get('target_ft', 'not set')
    checksum=session.get('checksum', 'not set')
    iters=values["time"]//values["period"]
    extra=values["time"]%values["period"]
    format_period=format_time(values["period"])
    metric=gen_metric(values["task"],values["metric"])
    path=os.path.join(app.config['UPLOAD_FOLDER'], session.get("filename","not set"))
    features=return_cols(path)
    estimator=run_task(path,values["task"],values["data_type"],target_ft)
    results=estimator(turn,values["period"],values["search_space"],values["prep_space"], metric)
    df=pd.DataFrame(data=results).sort_values(by="rank_test_scores")
    col_names=["Score","Estimator","Preprocessing","Details","Download"]
    res_list = [[a,b]for a, b in zip(df["mean_test_score"].values.tolist(),df["params"].values.tolist())]
    filehandler = open("tmp/results.p", 'wb') 
    pickle.dump(res_list, filehandler)
    turn+=+1
    #copy tmp files to save for later
    #filehandler = open("tmp/autosk_tmp/spaces.p", 'wb') 
    #pickle.dump([values["search_space"],values["prep_space"]],filehandler)
    if os.path.exists("tmp_runs/{}".format(checksum)):
        shutil.rmtree("tmp_runs/{}".format(checksum))
    shutil.copytree("tmp/autosk_tmp","tmp_runs/{}".format(checksum))
    #
    if(values["task"]=="classification"):
        res_list=[[row[0], format_ls("cl",row[1]["classifier:__choice__"]),format_ls("cp",row[1]["preprocessor:__choice__"]),"view","generate"] for row in res_list]
    else:
        res_list=[[row[0], format_ls("rg",row[1]["regressor:__choice__"]),format_ls("rp",row[1]["preprocessor:__choice__"]),"view","generate"] for row in res_list]
    if(turn>=iters):
        return render_template("results.html",column_names=col_names, row_data=res_list,zip=zip)
    else:
        return render_template("progress.html",turn=turn,iters=iters,PERIOD=format_period,RAW_PERIOD=values["period"], task=values["task"],time=values["time"],column_names=col_names, row_data=res_list,zip=zip)


@app.route('/stop')
def stop():
    values=session.get('values', 'not set')
    filehandler = open("tmp/results.p", 'rb') 
    res_list=pickle.load(filehandler)
    col_names=["Score","Estimator","Preprocessing","Details","Download"]
    if(values["task"]=="classification"):
        res_list=[[row[0], format_ls("cl",row[1]["classifier:__choice__"]),format_ls("cp",row[1]["preprocessor:__choice__"]),"view","generate"] for row in res_list]
    else:
        res_list=[[row[0], format_ls("rg",row[1]["regressor:__choice__"]),format_ls("rp",row[1]["preprocessor:__choice__"]),"view","generate"] for row in res_list]
    return render_template("results.html",column_names=col_names, row_data=res_list,zip=zip)


@app.route('/model')
def view_model():
    filehandler = open("tmp/results.p", 'rb') 
    res_list=pickle.load(filehandler)
    index = request.args.get('model', default = 0, type = int)
    model=res_list[index]
    print(model)
    return render_template("model.html",model=model,model_index=index)
 
@app.route("/generate_model")
def generate_model():
    values=session.get('values', 'not set')
    target_ft=session.get('target_ft', 'not set')
    features=session.get('features', 'not set')
    print(features)
    print(target_ft)
    #features.remove(target_ft)
    index = request.args.get('model', default = 0, type = int)
    filehandler = open("tmp/results.p", 'rb') 
    res_list=pickle.load(filehandler)
    arg_dict=res_list[index][1]
    param_dict=pipeline_gen.process_dict(arg_dict)
    pipe=Pipeline(([("preprocessor",pipeline_gen.build_preprocessor_cl(param_dict)),("classifeir",pipeline_gen.build_classifier(param_dict))]))
    path=os.path.join(app.config['UPLOAD_FOLDER'], session.get("filename","not set"))
    X,y=process_data(path,"csv",target_ft)
    pipe.fit(X,y)
    dump(pipe, 'tmp_files/model_{}.joblib'.format(str(index))) 
    filehandler = open("tmp_files/model_{}.pickle".format(str(index)), 'wb')
    pickle.dump(pipe, filehandler)
    #try:
    #skl_to_pmml(pipe,features,target_ft,"tmp_files/model_{}.pmml".format(index))
    #except:
    #    open("tmp_files/model_{}.pmml".format(index), 'a').close()
    return render_template("download.html",index=index)


@app.route('/download_joblib')
def download_joblib():
    index = request.args.get('model', default = 0, type = int)
    return send_from_directory("tmp_files",'model_{}.joblib'.format(str(index)), as_attachment=True)

@app.route('/download_pickle')
def download_pickle():
    index = request.args.get('model', default = 0, type = int)
    return send_from_directory("tmp_files",'model_{}.pickle'.format(str(index)), as_attachment=True)



@app.route('/download_pmml')
def download_pmml():
    index = request.args.get('model', default = 0, type = int)
    if os.path.getsize("tmp_files/model_{}.pmml".format(index))>0:
        return send_from_directory("tmp_files",'model_{}.joblib'.format(str(index)), as_attachment=True)
    return "This pipeline is not supported for pmml. Try joblib/pickle"



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



