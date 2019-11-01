import os
from imblearn.over_sampling import SMOTE
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
from predict_meta import predict_meta, predict_time
from utils_local import *
import matplotlib.pyplot as plt
import pipeline_gen
#from sklearn.pipeline import Pipeline #original pipline
from imblearn.pipeline import Pipeline #smote pipeline
from joblib import dump, load
#from nyoka import skl_to_pmml
import numpy as np
from sklearn.inspection import plot_partial_dependence
from pdpbox import pdp, get_dataset, info_plots

tmp_folder = 'tmp/autosk_tmp'
output_folder = 'tmp/autosk_out'

#ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])
ALLOWED_EXTENSIONS = set(["npy","csv"])

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/iautosklearn')
def to_main():
    return redirect('/iautosklearn/')

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
                rec=predict_meta(meta)
            values['task']=task
            session["filename"]=filename
            session["values"]=values
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
    features=return_cols(path)
    new_data=select_cols(path,features)
    for i in range(len(features)):
        plt.clf()
        new_data[features[i]].hist()
        plt.savefig("static/images/figs/"+str(i),bbox_inches="tight",transparent=True)
    return render_template("features.html", FEATURES=features)

@app.route('/features', methods=['POST'])
def feature_pgr():
    if request.method == 'POST':
        # check if the post request has the file part
        target_ft = request.form['target_ft']
        session["target_ft"]=target_ft
        features = request.form.getlist("features_ls")
        if(target_ft not in features):
            return "You can't discard the target class"      
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
    unique, counts = np.unique(data[target_ft], return_counts=True)
    classes=dict(zip(unique, counts))
    mx_key=max(classes,key=classes.get)
    plt.clf()
    data[target_ft].hist()
    plt.savefig("static/images/figs/target",bbox_inches="tight",transparent=True)
    ratio=[True if (min(data[target_ft].value_counts())/max(data[target_ft].value_counts()))<0.6 else False][0]
    pre_metric=["F1" if ratio else "Accuracy" ][0]
    return render_template("target.html",TASK=values["task"],ratio=ratio,METRICS=METRICS,pre_metric=pre_metric,classes=classes,mx_key=mx_key)

@app.route('/target_class', methods=['POST'])
def target_class_r():
    if request.method == 'POST':
        SMOTE_N=5
        # check if the post request has the file part
        values=session.get('values', 'not set')
        metric = request.form['metric']
        if values["task"]=="classification":
            smote = request.form['smote']
        else:
            smote = "no"
        values["metric"]=metric
        session["values"]=values
        target_ft=session.get('target_ft', 'not set')
        features=session.get('features', 'not set')
        #feature dropping can be brought here for better perforamnce
        session["smote"]=smote
        if smote == "yes":
            smote_dic={}
            path=os.path.join(app.config['UPLOAD_FOLDER'], session.get("filename","not set"))
            X,y,_=process_data(path,"csv",target_ft)
            unique, counts = np.unique(y, return_counts=True)
            if min(counts)<=SMOTE_N:
                SMOTE_N=min(counts)-1
            smote_ratios =[int(float(x)*max(counts)) for x in request.form.getlist("smote_ratio[]")]
            print(smote_ratios)
            for i in range(len(smote_ratios)):
                smote_dic[unique[i]]=smote_ratios[i]
            print(smote_dic)
            sm = SMOTE(random_state=42,sampling_strategy=smote_dic, k_neighbors=SMOTE_N)
            X_res, y_res = sm.fit_resample(X, y)
            new_data=pd.DataFrame(np.column_stack((X_res,y_res)),columns=list(features)+[target_ft])
            new_data.to_csv(path,index=False)
        print(smote)
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
        values['data_type']=data_type
        values["search_space"]=search_space
        values["prep_space"]=prep_space
        session["values"]=values
        return redirect('/budget')

@app.route('/budget')
def budget():
    task=session.get("task","not set")
    values=session.get('values', 'not set')
    total_pred_time=0
    filename=session.get("filename","not set")
    meta=get_meta(os.path.join(app.config['UPLOAD_FOLDER'], filename),'csv')
    time_pred=predict_time(meta)
    print(time_pred)
    for each in values["search_space"]:
        if each in ESTIMATOR_TIMES.keys():
            tm=ESTIMATOR_TIMES[each]
            total_pred_time+=0.2*(time_pred)
    #total_pred_time=time_pred
    print(total_pred_time)
    ##Configure for Task
    return render_template('budget.html',  zip=zip, TASK=task, PRED_TIME=int(total_pred_time))

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
    col_names=["Classifier","{} Max Score".format(values["metric"]),"Models Trained","Show Models"]
    if values["task"]!="classification":
        col_names[1]="Regressor"
    #Sort list by scores
    res_list = [[a,b]for a, b in zip(df["mean_test_score"].values.tolist(),df["params"].values.tolist())]
    #divide list in dictionaries and dump to drive
    

    grouped_results={}
    if values["task"]=='classification':
        ESTIMATORS=CLASSIFIERS
        ESTIMATORS_DISP=CLASSIFIERS_DISP
        for each in CLASSIFIERS:
            grouped_results[each]=[]
        for each in res_list:
            grouped_results[each[1]['classifier:__choice__' ]].append(each)
    else:
        ESTIMATORS=REGRESSORS
        ESTIMATORS_DISP=REGRESSORS_DISP
        for each in REGRESSORS:
            grouped_results[each]=[]
        for each in res_list:
            grouped_results[each[1]['regressor:__choice__' ]].append(each)
 

    with open("tmp/results.p", 'wb') as filehandler: 
        pickle.dump(grouped_results, filehandler)
    res_list=[]
    for  each in grouped_results.keys():
        if grouped_results[each]:
            res_list.append((ESTIMATORS_DISP[ESTIMATORS.index(each)],round(grouped_results[each][0][0],3),len(grouped_results[each]),"View"))
    res_list.sort(key=lambda x:x[0],reverse=True)
    turn+=+1
    #copy tmp files to save for later
    if os.path.exists("tmp_runs/{}".format(checksum)):
        shutil.rmtree("tmp_runs/{}".format(checksum))
    shutil.copytree("tmp/autosk_tmp","tmp_runs/{}".format(checksum))
    with open("tmp/results.p", 'rb') as filehandler:
        or_list=pickle.load(filehandler)
    estim_dict={"col_names":[],"disp_index":[],"index":[],"fig_names":[],"res_list":[]}
    for each in res_list:
        index=ESTIMATORS[ESTIMATORS_DISP.index(each[0])]
        fres_list=or_list[index]

        if values["task"]=='classification':
            slc=len("classifier:{}:".format(index))
            col_names_e=[x for x in list(fres_list[0][1].keys()) if x[:10]=="classifier" and x[-21:]!="min_impurity_decrease"][1:]
        else:
            slc=len("regressor:{}:".format(index))
            col_names_e=[x for x in list(fres_list[0][1].keys()) if x[:10]=="regressor" and x[-21:]!="min_impurity_decrease"][1:]
        fres_list=[[round(x[0],3),x[1]["preprocessor:__choice__"].replace("_"," ").title()]+ [x[1][k]  if type(x[1][k])!= float  and type(x[1][k])!=str else round(x[1][k],3) if type(x[1][k])==float else x[1][k].replace("_"," ").title() for k in  col_names_e ]+["Interpret"] for x in fres_list]
        col_names_e= [("{} Score".format(values["metric"])),"Preprocessor"]+[x[slc:].replace("_"," ").title() for x in col_names_e]+["Details"]
        disp_index=index.replace("_"," ").title()
        ##plotting
        fig_names=[]
        for i in range(1,len(fres_list[0])):
            if type(fres_list[0][i])==float or type(fres_list[0][i])==int:
                plt.clf()
                plt.xlabel(col_names_e[i])
                plt.ylabel("{} Score".format(values["metric"]))
                plt.scatter([x[i] for x in fres_list],[x[0] for x in fres_list])
                plt.savefig("static/images/figs/"+index+str(i),bbox_inches="tight",transparent=True)
                fig_names.append(index+str(i))
        estim_dict["col_names"].append(col_names_e)	
        estim_dict["disp_index"].append(disp_index)	
        estim_dict["index"].append(index)	
        estim_dict["fig_names"].append(fig_names)	
        estim_dict["res_list"].append(fres_list)	
    res_list.sort(key=lambda x:x[1],reverse=True)
    if(turn>=iters):
        return render_template("results.html",column_names=col_names, row_data=res_list,zip=zip,len=len, CLASSIFIERS=ESTIMATORS,CLASSIFIERS_DISP=ESTIMATORS_DISP, estim_dict=estim_dict)
    else:
        return render_template("progress.html",turn=turn,iters=iters,PERIOD=format_period,RAW_PERIOD=values["period"], task=values["task"],time=values["time"],column_names=col_names, row_data=res_list,zip=zip,CLASSIFIERS=ESTIMATORS, CLASSIFIERS_DISP=ESTIMATORS_DISP,estim_dict=estim_dict)

@app.route('/stop')
def stop():
    values=session.get('values', 'not set')
    with open("tmp/results.p", 'rb') as filehandler:
        grouped_results=pickle.load(filehandler)
    col_names=["{} Max Score".format(values["metric"]),"Classifier","Show Models"]
    res_list=[]
    for  each in grouped_results.keys():
        if grouped_results[each]:
            res_list.append((CLASSIFIERS_DISP[CLASSIFIERS.index(each)],round(grouped_results[each][0][0],3),len(grouped_results[each]),"View"))
    res_list.sort(key=lambda x:x[0],reverse=True)
    estim_dict={"col_names":[],"disp_index":[],"index":[],"fig_names":[],"res_list":[]}
    for each in res_list:
        index=CLASSIFIERS[CLASSIFIERS_DISP.index(each[0])]
        fres_list=grouped_results[index]
        slc=len("classifier:{}:".format(index))
        col_names_e=[x for x in list(fres_list[0][1].keys()) if x[:10]=="classifier" and x[-21:]!="min_impurity_decrease"][1:]
        fres_list=[[round(x[0],3),x[1]["preprocessor:__choice__"].replace("_"," ").title()]+ [x[1][k]  if type(x[1][k])!= float  and type(x[1][k])!=str else round(x[1][k],3) if type(x[1][k])==float else x[1][k].replace("_"," ").title() for k in  col_names_e ]+["Interpret"] for x in fres_list]
        col_names_e= [("{} Score".format(values["metric"])),"Preprocessor"]+[x[slc:].replace("_"," ").title() for x in col_names_e]+["Details"]
        disp_index=index.replace("_"," ").title()
        ##plotting
        fig_names=[]
        for i in range(1,len(fres_list[0])):
            if type(fres_list[0][i])==float or type(fres_list[0][i])==int:
                fig_names.append(index+str(i))
        estim_dict["col_names"].append(col_names_e)	
        estim_dict["disp_index"].append(disp_index)	
        estim_dict["index"].append(index)	
        estim_dict["fig_names"].append(fig_names)	
        estim_dict["res_list"].append(fres_list)	
    return render_template("results.html",column_names=col_names, estim_dict=estim_dict,row_data=res_list,zip=zip)

@app.route('/estimator')
def view_estimator():
    values=session.get('values', 'not set')
    with open("tmp/results.p", 'rb') as filehandler:
        or_list=pickle.load(filehandler)
    index = request.args.get('model', default = None, type = str)
    res_list=or_list[index]
    slc=len("classifier:{}:".format(index))
    col_names=[x for x in list(res_list[0][1].keys()) if x[:10]=="classifier" and x[-21:]!="min_impurity_decrease"][1:]
    res_list=[[round(x[0],3),x[1]["preprocessor:__choice__"].replace("_"," ").title()]+ [x[1][k]  if type(x[1][k])!= float  and type(x[1][k])!=str else round(x[1][k],3) if type(x[1][k])==float else x[1][k].replace("_"," ").title() for k in  col_names ]+["Interpret"] for x in res_list]
    col_names= [("{} Score".format(values["metric"])),"Preprocessor"]+[x[slc:].replace("_"," ").title() for x in col_names]+["Details"]
    disp_index=index.replace("_"," ").title()
    ##plotting
    fig_names=[]
    for i in range(1,len(res_list[0])):
        if type(res_list[0][i])==float or type(res_list[0][i])==int:
            plt.clf()
            plt.xlabel(col_names[i])
            plt.ylabel("{} Score".format(values["metric"]))
            plt.scatter([x[i] for x in res_list],[x[0] for x in res_list])
            plt.savefig("static/images/figs/"+index+str(i),bbox_inches="tight",transparent=True)
            fig_names.append(index+str(i))
    return render_template("estimator_results.html",column_names=col_names,disp_index=disp_index, estimator=index,fig_names=fig_names,row_data=res_list,zip=zip)
 
@app.route('/model')
def view_model():
    with open("tmp/results.p", 'rb') as filehandler:
        res_list=pickle.load(filehandler)
    index = request.args.get('model', default = 0, type = int)
    estim = request.args.get('estimator', default = None, type = str)
    model=res_list[estim][index]
    return render_template("model.html",model=model,estimator=estim,model_index=index)
 
@app.route("/generate_model")
def generate_model():
    #generates model from the parameters and trains the model on the train set
    #Load parameters
    values=session.get('values', 'not set')
    smote=session.get('smote', 'not set')
    smote="no" #dont include smote in the pipeline
    target_ft=session.get('target_ft', 'not set')
    features=session.get('features', 'not set')
    index = request.args.get('model', default = 0, type = int)
    estim = request.args.get('estimator', default = None, type = str)
    filehandler = open("tmp/results.p", 'rb') 
    res_list=pickle.load(filehandler)
    arg_dict=res_list[estim][index][1]
    #constuct and fit pipeline
    param_dict=pipeline_gen.process_dict(arg_dict)
    pipeline_params=[("preprocessor",pipeline_gen.build_preprocessor_cl(param_dict)),("classifeir",pipeline_gen.build_classifier(param_dict))]
    if smote =="yes":
        pipeline_params.insert(0,("smote",SMOTE(random_state=42)))
    pipe=Pipeline(pipeline_params)
    path=os.path.join(app.config['UPLOAD_FOLDER'], session.get("filename","not set"))
    X,y,data=process_data(path,"csv",target_ft)
    pipe.fit(X,y)
    dump(pipe, 'tmp_files/model_{}_{}.joblib'.format(estim,str(index))) 
    with open("tmp_files/model_{}_{}.pickle".format(estim,str(index)), 'wb') as filehandler:
        pickle.dump(pipe, filehandler)
    cl=param_dict["classifier:__choice__"]
    #feature importances
    importance=(pipeline_gen.get_importance(pipe,cl,smote))
    metric_res=pipeline_gen.get_matrix(pipe,X,y,smote)     
    if len(importance)>0:
        imps=[[features[i],round(importance[i],2)] for i in range(len(features))]
        imps=sorted(imps,key=lambda l:l[1],reverse=True)
        plt_features=[x[0] for x in reversed(imps)]
        plt_imps=[x[1] for x in reversed(imps)]
        plt.clf()
        plt.title("Feature Importance")
        plt.ylabel("Feature Name")
        plt.xlabel("Importance")      
        plt.barh(plt_features,plt_imps,align='center',height=0.2, color='c')
        plt.savefig("static/images/figs/model_imp",bbox_inches="tight",transparent=True)
    else:
        imps=[]
    column_names=["Metric","Score"]
    metric_names=["Accuracy","Recall","Precision","F1"]
    metric_res=[[metric_names[i],round(metric_res[i],3)] for i in range(len(metric_res))]

    #partial dependancy
    partial_fig_names=[]
    """
    for feat in features:
        #TODO: fix for estimator position with smote on
        part_fig=plt.figure(figsize=(5,5))
        partial_path="partial_"+str(feat)
        partial_fig_names.append(partial_path)
        #plot_partial_dependence(pipe.steps[1][1], X, [feat], fig=part_fig,feature_names=features) 
        feat_p = pdp.pdp_isolate(model=pipe.steps[1][1], dataset=data, model_features=features, feature=feat)
        fig, axes = pdp.pdp_plot(pdp_isolate_out=feat_p, feature_name=feat, center=True, x_quantile=True, plot_lines=True, frac_to_plot=100, show_percentile=False)
        fig.savefig("static/images/figs/"+partial_path,bbox_inches="tight",transparent=True)
    plt.figure()
    """
    #column_names=["Feature","Importance"]
    #return render_template("download.html",index=index,column_names=column_names,row_data=imps,CL_Name=cl, metric_res=metric_res,zip=zip)
    
    return render_template("download.html",features=features,targets=np.unique(y),estimator=estim,index=index,column_names=column_names,row_data=metric_res,CL_Name=cl, metric_res=metric_res,zip=zip,partial_fig_names=partial_fig_names)

@app.route('/download_joblib')
def download_joblib():
    index = request.args.get('model', default = 0, type = int)
    estim = request.args.get('estimator', default = None, type = str)
    return send_from_directory("tmp_files",'model_{}_{}.joblib'.format(estim,str(index)), as_attachment=True)

@app.route('/download_pickle')
def download_pickle():
    index = request.args.get('model', default = 0, type = int)
    estim = request.args.get('estimator', default = None, type = str)
    return send_from_directory("tmp_files",'model_{}_{}.pickle'.format(estim,str(index)), as_attachment=True)

@app.route('/download_pmml')
def download_pmml():
    index = request.args.get('model', default = 0, type = int)
    if os.path.getsize("tmp_files/model_{}.pmml".format(index))>0:
        return send_from_directory("tmp_files",'model_{}.joblib'.format(str(index)), as_attachment=True)
    return "This pipeline is not supported for pmml. Try joblib/pickle"

@app.route('/plot_pdp')
def plot_pdp():
    path=os.path.join(app.config['UPLOAD_FOLDER'], session.get("filename","not set"))
    index = request.args.get('model', default = 0, type = int)
    estim = request.args.get('estimator', default = None, type = str)
    target_ft=session.get('target_ft', 'not set')
    features=session.get('features', 'not set')
    f1 = request.args.get('f1', default = None, type = str)
    t1 = request.args.get('t1', default = None, type = str)
    X,y,data=process_data(path,"csv",target_ft)


    chosen_class=list(np.unique(y)).index(int(float(t1)))

    with open("tmp_files/model_{}_{}.pickle".format(estim,str(index)), 'rb') as filehandler:
        pipe=pickle.load(filehandler)
    mod_fig=plt.figure(figsize=(10,10))
    mod_path="modal_"+"pdp_"+str(f1.replace('.','_'))
    feat_p = pdp.pdp_isolate(model=pipe.steps[1][1], dataset=data, model_features=features, feature=f1)
    fig, axes = pdp.pdp_plot(pdp_isolate_out=feat_p, feature_name=f1, center=True, x_quantile=True, plot_lines=True, frac_to_plot=100, show_percentile=False, which_classes=[chosen_class], plot_params={"subtitle":"For Class {}, Label: {}".format(chosen_class,t1)})
    fig.savefig("static/images/figs/"+mod_path,bbox_inches="tight",transparent=True)
    plt.figure()

    #return "PARAMS: {}, {}, {}, {}, {}".format(str(index),str(estim),f1,f2,t1) 
    return render_template("modal_plot.html", plot_name=mod_path)


@app.route('/plot_modal')
def plot_modal():
    
    path=os.path.join(app.config['UPLOAD_FOLDER'], session.get("filename","not set"))
    index = request.args.get('model', default = 0, type = int)
    estim = request.args.get('estimator', default = None, type = str)
    target_ft=session.get('target_ft', 'not set')
    features=session.get('features', 'not set')
    f1 = request.args.get('f1', default = None, type = str)
    f2 = request.args.get('f2', default = None, type = str)
    t1 = request.args.get('t1', default = None, type = str)
    X,y,data=process_data(path,"csv",target_ft)


    chosen_class=list(np.unique(y)).index(int(float(t1)))
    
    with open("tmp_files/model_{}_{}.pickle".format(estim,str(index)), 'rb') as filehandler:
        pipe=pickle.load(filehandler)
    mod_fig=plt.figure(figsize=(10,10))
    mod_path="modal_"+str(f1.replace('.','_'))+"_"+str(f2.replace('.','_'))
    #feat_p = pdp.pdp_isolate(model=pipe.steps[1][1], dataset=data, model_features=features, feature=feat)
    pdp_V1_V2 = pdp.pdp_interact(
    model=pipe.steps[1][1], dataset=data, model_features=features, features=[f1, f2], 
    num_grid_points=None,  percentile_ranges=[None, None]
    )
    fig, axes = pdp.pdp_interact_plot(
    pdp_V1_V2, [f1, f2], plot_type='grid',x_quantile=True, ncols=2, plot_pdp=True, 
    which_classes=[chosen_class],plot_params={"subtitle":"For Class {}, Label: {}".format(chosen_class,t1)}
    )
    fig.savefig("static/images/figs/"+mod_path,bbox_inches="tight",transparent=True)
    plt.figure()
    
    return render_template("modal_plot.html", plot_name=mod_path)





@app.route("/loading")
def loading():
    return render_template("loading.html")



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

app.run(host='0.0.0.0', port=80,debug=True)



