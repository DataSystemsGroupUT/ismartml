import os
import shutil
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pipeline_gen
import re
import time
from app import app
from threading import Thread
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline  # smote pipeline
from sklearn.impute import SimpleImputer
from flask import Flask, flash, request, redirect, render_template, url_for, session, send_from_directory
from werkzeug.utils import secure_filename
from multi import run_task, process_data, run_task_tpot
from extras import *
from extract import get_meta
from predict_meta import predict_meta, predict_time
from utils_local import *
from joblib import dump, load
from pdpbox import pdp
from tpot_search import classifier_config_dict 


tmp_folder = 'tmp/autosk_tmp'
output_folder = 'tmp/autosk_out'

ALLOWED_EXTENSIONS = set(["npy", "csv"])


def url_mod(fnc):
    #pre = "/ismartml"
    pre = ""
    return pre + url_for(fnc)


def allowed_file(filename):
    return '.' in filename and filename.rsplit(
        '.', 1)[1].lower() in ALLOWED_EXTENSIONS


def merge_logs():
    while True:
        res_list=[]
        time.sleep(5)
        dr = os.listdir("tmp_files/tpot")
        start = False
        for each in dr:
            with open("tmp_files/tpot/"+each,'r') as f:
                lns=f.readlines()
                curr=[]
                for line in lns:
                    if line[:13] == "# Average CV " :
                        #combined_res.append(line[21:])
                        #combined_res += line
                        start = True
                    elif line[0] == "#":
                        start = False
                    if start:
                        curr.append(line)
            if len(curr) > 2:
                curr[0] = curr[0].split(":")[1]
                curr = curr[:-1]
                del curr[1]
            else:
                curr[0] = curr[0].split(":")[1]
                curr[1] = curr[1][20:]
            res_list.append(curr)
        res_list.sort(key = lambda x: x[0],reverse=True) 
        combined_res = "<table><tr><th>Accuracy</th><th>Pipeline</th></tr>"
        for curr in res_list:
            combined_res+="<tr><td>"
            combined_res+=curr[0]+"</td><td>"
            for each in curr[1:]:
                combined_res += re.sub(r'\([^)]*\)', '', each.strip())
            combined_res+='\n'
            combined_res+="</td></tr>"
        combined_res+="</table>"
        #res_list=[ pipe.split('(')[:-1] for pipe in combined_res]
        #combined_res=str(res_list)
        with open("static/data/prog.txt", "w") as file: 
            file.write(combined_res.replace(',',' - '))



thread = Thread(target=merge_logs)
thread.daemon = True
thread.start()


@app.route('/')
def start():
    if not os.path.exists("data/hash_list.txt"):
        os.mknod("data/hash_list.txt")
    return render_template("index.html")


@app.route('/', methods=['POST'])
def start_p():
    if request.method == 'POST':
        values = {}
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        #data_type = request.form['data_type']
        data_type = "csv"
        task = request.form['task']
        backend = request.form['backend']
        if file.filename == '':
            flash('No file selected for uploading')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            if data_type == "numpy" and filename[-3:] != "npy":
                return "Wrong file extension (expected .npy)"
            if data_type == "csv" and (filename[-3:] != "csv" and filename[-3:] != "CSV"):
                return "Wrong file extension (expected .csv)"
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            rec = []
            if task == "classification":
                meta = get_meta(os.path.join(
                    app.config['UPLOAD_FOLDER'], filename), data_type)
                rec = predict_meta(meta)
            values['task'] = task
            values["backend"] = backend
            values['data_type'] = data_type
            session["filename"] = filename
            session["values"] = values
            session["data_type"] = data_type
            session["rec"] = rec
            session["task"] = task
            return redirect(url_mod("featur_pg"))
        else:
            flash('Allowed file types are: {}'.format(str(ALLOWED_EXTENSIONS)))
            return redirect(request.url)


@app.route('/features')
def featur_pg():
    values = session.get('values', 'not set')
    path = os.path.join(app.config['UPLOAD_FOLDER'],
                        session.get("filename", "not set"))
    data = load_initial(path)
    features= data.columns
    for i in range(len(features)):
        plt.clf()
        data[features[i]].hist()
        plt.savefig("static/images/figs/" + str(i),
                    bbox_inches="tight", transparent=True)
    return render_template("features.html", FEATURES=features)


@app.route('/features', methods=['POST'])
def feature_pgr():
    if request.method == 'POST':
        # check if the post request has the file part
        target_ft = request.form['target_ft']
        session["target_ft"] = target_ft
        features = request.form.getlist("features_ls")
        if target_ft not in features:
            return "You can't discard the target class"
        features.remove(target_ft)
        session["features"] = features
        path = os.path.join(
            app.config['UPLOAD_FOLDER'], session.get("filename", "not set"))
        new_data = select_cols(path, list(features) + [target_ft])
        new_data.to_csv(path, index=False)
        return redirect(url_mod('target_class'))


@app.route('/target_class')
def target_class():
    task = session.get("task", "not set")
    # Configure for Task
    if task == "classification":
        METRICS = METRICS_CL_DISP
    else:
        METRICS = METRICS_RG_DISP
    values = session.get('values', 'not set')
    target_ft = session.get('target_ft', 'not set')
    path = os.path.join(app.config['UPLOAD_FOLDER'],
                        session.get("filename", "not set"))
    data = pd.read_csv(path)
    unique, counts = np.unique(data[target_ft], return_counts=True)
    classes = dict(zip(unique, counts))
    mx_key = max(classes, key=classes.get)
    plt.clf()
    data[target_ft].hist()
    plt.savefig("static/images/figs/target",
                bbox_inches="tight", transparent=True)
    ratio = [True if (min(data[target_ft].value_counts()) /
                      max(data[target_ft].value_counts())) < 0.6 else False][0]
    pre_metric = ["F1" if ratio else "Accuracy"][0]
    return render_template(
        "target.html",
        TASK=values["task"],
        ratio=ratio,
        METRICS=METRICS,
        pre_metric=pre_metric,
        classes=classes,
        mx_key=mx_key)


@app.route('/target_class', methods=['POST'])
def target_class_r():
    if request.method == 'POST':
        SMOTE_N = 5
        # check if the post request has the file part
        values = session.get('values', 'not set')
        metric = request.form['metric']
        if values["task"] == "classification":
            smote = request.form['smote']
        else:
            smote = "no"
        values["metric"] = metric
        session["values"] = values
        target_ft = session.get('target_ft', 'not set')
        features = session.get('features', 'not set')
        # feature dropping can be brought here for better perforamnce
        session["smote"] = smote
        if smote == "yes":
            smote_dic = {}
            path = os.path.join(
                app.config['UPLOAD_FOLDER'], session.get(
                    "filename", "not set"))
            X, y, _ = process_data(path, "csv", target_ft)
            unique, counts = np.unique(y, return_counts=True)
            if min(counts) <= SMOTE_N:
                SMOTE_N = min(counts) - 1
            smote_ratios = [int(float(x) * max(counts))
                            for x in request.form.getlist("smote_ratio[]")]
            print(smote_ratios)
            for i in range(len(smote_ratios)):
                smote_dic[unique[i]] = smote_ratios[i]
            print(smote_dic)
            sm = SMOTE(random_state=42, sampling_strategy=smote_dic,
                       k_neighbors=SMOTE_N)
            X_res, y_res = sm.fit_resample(X, y)
            new_data = pd.DataFrame(np.column_stack(
                (X_res, y_res)), columns=list(features) + [target_ft])
            new_data.to_csv(path, index=False)
        
        if values['backend']=='autosklearn':
            return redirect(url_mod('params'))
        elif values['backend']=='tpot':
            #return redirect(url_mod('params_tpot'))
            return redirect(url_mod('params_tpot'))

@app.route('/params_tpot')
def params_tpot():
    ################TODO :
    #rec = session.get("rec", "not set")
    task = session.get("task", "not set")
    #column_names = ["Classifier", "Score"]
    #bolds = []
    # Configure for Task
    if task == "classification":
        # Get corect lists for this task
        ESTIMATORS = [TPOT_CLASSIFIERS, TPOT_CLASSIFIERS]
        #PREPROCESSORS = [PREPROCESSORS_CL, PREPROCESSORS_CL_DISP]
    else:
        ESTIMATORS = [REGRESSORS, REGRESSORS_DISP]
        #PREPROCESSORS = [PREPROCESSORS_RG, PREPROCESSORS_RG_DISP]
    return render_template(
        'parameters_tpot.html',
        ESTIMATORS=ESTIMATORS,
        #PREPROCESSORS=PREPROCESSORS,
        #column_names=column_names,
        #row_data=rec,
        zip=zip,
        TASK=task,
        #BOLD_CL=bolds)
        )

@app.route('/params_tpot', methods=['POST'])
def params_tpot_p():
    if request.method == 'POST':
        values = session.get('values', 'not set')
        filename = session.get("filename", "not set")
        task = session.get("task", "not set")
        selected_opers = request.form.getlist("estim_ls")
        #prep_space = request.form.getlist("prep_ls")
        if not selected_opers:
            return "You must select at least 1 estimator"
        #if not prep_space:
        #    return "You must select at least 1 preprocessor"
        values["selected_opers"] = selected_opers
        #values["prep_space"] = prep_space
        session["values"] = values
        return redirect(url_mod('budget_tpot'))



@app.route('/params')
def params():
    rec = session.get("rec", "not set")
    task = session.get("task", "not set")
    column_names = ["Classifier", "Score"]
    bolds = []
    # Configure for Task
    if task == "classification":
        # remove predicions with 0 score from results
        rec = [x for x in rec if x[1] != 0]
        # get bold indexes for recomended classifiers
        rec_t = list(map(list, zip(*rec)))
        for cl in CLASSIFIERS_DISP:
            if cl in rec_t[0]:
                bolds.append(CLASSIFIERS_DISP.index(cl))
            elif cl[-3:] == "SVC":
                if "SVC" in rec_t[0]:
                    bolds.append(CLASSIFIERS_DISP.index(cl))
        # Get corect lists for this task
        ESTIMATORS = [CLASSIFIERS, CLASSIFIERS_DISP]
        PREPROCESSORS = [PREPROCESSORS_CL, PREPROCESSORS_CL_DISP]
    else:
        ESTIMATORS = [REGRESSORS, REGRESSORS_DISP]
        PREPROCESSORS = [PREPROCESSORS_RG, PREPROCESSORS_RG_DISP]
    return render_template(
        'upload.html',
        ESTIMATORS=ESTIMATORS,
        PREPROCESSORS=PREPROCESSORS,
        column_names=column_names,
        row_data=rec,
        zip=zip,
        TASK=task,
        BOLD_CL=bolds)


@app.route('/params', methods=['POST'])
def params_p():
    if request.method == 'POST':
        values = session.get('values', 'not set')
        filename = session.get("filename", "not set")
        task = session.get("task", "not set")
        search_space = request.form.getlist("estim_ls")
        prep_space = request.form.getlist("prep_ls")
        if not search_space:
            return "You must select at least 1 estimator"
        if not prep_space:
            return "You must select at least 1 preprocessor"
        values["search_space"] = search_space
        values["prep_space"] = prep_space
        session["values"] = values
        return redirect(url_mod('budget'))

@app.route('/budget_tpot')
def budget_tpot():
    task = session.get("task", "not set")
    values = session.get("values", "not set")
    total_pred_time = 0
    filename = session.get("filename", "not set")
    meta = get_meta(os.path.join(app.config['UPLOAD_FOLDER'], filename), 'csv')
    time_pred = predict_time(meta)
    #for each in values["search_space"]:
    #    if each in ESTIMATOR_TIMES.keys():
    #        tm = ESTIMATOR_TIMES[each]
    #        total_pred_time += 0.2 * (time_pred)
    if total_pred_time<60:
        total_pred_time=60
    return render_template('budget.html', zip=zip,
                           TASK=task, PRED_TIME=int(total_pred_time))


@app.route('/budget_tpot', methods=['POST'])
def budget_tpot_p():
    if request.method == 'POST':
        values = session.get('values', 'not set')
        time = request.form['time']
        period = request.form['period']
        data_type = session.get('data_type', 'not set')
        filename = session.get("filename", "not set")
        task = session.get("task", "not set")
        reuse = request.form['reuse']
        if int(time) < 30:
            return "Time budget must be at least 30 seconds"
        if int(period) < 30:
            return "Update period must be at least 30 seconds"
        if int(period) > int(time):
            return "Update period can't be larger than total time budget"
        values['time'] = int(time)
        values['period'] = int(period)
        session["values"] = values
        session["reuse"] = reuse
        return redirect(url_mod('running_tpot'))



@app.route('/budget')
def budget():
    task = session.get("task", "not set")
    values = session.get("values", "not set")
    total_pred_time = 0
    filename = session.get("filename", "not set")
    meta = get_meta(os.path.join(app.config['UPLOAD_FOLDER'], filename), 'csv')
    time_pred = predict_time(meta)
    for each in values["search_space"]:
        if each in ESTIMATOR_TIMES.keys():
            tm = ESTIMATOR_TIMES[each]
            total_pred_time += 0.2 * (time_pred)
    if total_pred_time<30:
        total_pred_time=30
    return render_template('budget.html', zip=zip,
                           TASK=task, PRED_TIME=int(total_pred_time))


@app.route('/budget', methods=['POST'])
def budget_p():
    if request.method == 'POST':
        values = session.get('values', 'not set')
        time = request.form['time']
        period = request.form['period']
        data_type = session.get('data_type', 'not set')
        filename = session.get("filename", "not set")
        task = session.get("task", "not set")
        reuse = request.form['reuse']
        if int(time) < 30:
            return "Time budget must be at least 30 seconds"
        if int(period) < 30:
            return "Update period must be at least 30 seconds"
        if int(period) > int(time):
            return "Update period can't be larger than total time budget"
        values['time'] = int(time)
        values['period'] = int(period)
        session["values"] = values
        session["reuse"] = reuse
        return redirect(url_mod('running'))

#@app.route('/running_tpot')
#def running_tpot():
#    values = session.get('values', 'not set')
#    target_ft = session.get('target_ft', 'not set')
#    path = os.path.join(app.config['UPLOAD_FOLDER'],
#                        session.get("filename", "not set"))
#    pipeline_optimizer = run_task_tpot(path, values["task"], values["data_type"], values["time"],target_ft)
#    #return str(list(pipeline_optimizer.evaluated_individuals_.keys())[0])
#    res_list=pipeline_optimizer.evaluated_individuals_.keys()
#    res=pipeline_optimizer.evaluated_individuals_
#    #res_list=res[list(pipeline_optimizer.evaluated_individuals_.keys())[0]]
#    res_list=[ pipe.split('(')[:-1] for pipe in res.keys()]
#    #return str(res_list)
#    col_names=["one","two","three","four","five","six"]
#    return render_template(
#            "base_results.html",
#            url_mod=url_mod,
#            column_names=col_names,
#            row_data=res_list,
#            zip=zip,
#            len=len,
#            #task=values['task'])
#            )

@app.route('/running_tpot')
def running_tpot():
    values = session.get('values', 'not set')
    target_ft = session.get('target_ft', 'not set')
    iters = values["time"] // values["period"]
    extra = values["time"] % values["period"]
    format_period = format_time(values["period"])
    reuse = session.get('reuse', 'not set')
    # check dataset checksum and lookup
    path = os.path.join(app.config['UPLOAD_FOLDER'],
                        session.get("filename", "not set"))

    dr = os.listdir("tmp_files/tpot")
    for each in dr:
        os.remove("tmp_files/tpot/"+each)




    return render_template(
        'running_tpot.html',
        url_mod=url_mod,
        turn=0,
        task=values["task"],
        time=values["time"],
        iters=iters,
        PERIOD=format_period,
        RAW_PERIOD=values["period"])



@app.route('/running')
def running():
    values = session.get('values', 'not set')
    target_ft = session.get('target_ft', 'not set')
    iters = values["time"] // values["period"]
    extra = values["time"] % values["period"]
    format_period = format_time(values["period"])
    reuse = session.get('reuse', 'not set')
    # check dataset checksum and lookup
    path = os.path.join(app.config['UPLOAD_FOLDER'],
                        session.get("filename", "not set"))
    checksum = hash_file(path) + "_" + target_ft + "_" + \
        values["task"] + "_" + values["metric"]
    session["checksum"] = checksum
    with open("data/hash_list.txt", "r") as f:
        lines = f.readlines()
    if checksum + "\n" not in lines:
        with open("data/hash_list.txt", "a") as f:
            f.write(checksum + "\n")
    for dir_ in [tmp_folder, output_folder]:
        try:
            shutil.rmtree(dir_)
        except OSError:
            pass
    # copy tmp to run on it
    if reuse == "yes":
        if os.path.exists("tmp_runs/{}".format(checksum)):
            shutil.copytree("tmp_runs/{}".format(checksum), "tmp/autosk_tmp")
            # modify space.pcs
            olds = []
            old_pres = []
            path = "tmp/autosk_tmp/space.pcs"
            with open(path, "r") as f:
                lines = f.readlines()
                "classifier:__choice__ {decision_tree, gradient_boosting, random_forest} [random_forest]"
                pre = "classifier:__choice__ {"
                for line in lines:
                    if "classifier:__choice__ {" in line:
                        olds = [ar.strip()
                                for ar in line[len(pre):].split("}")[0].split(",")]
                    elif "preprocessor:__choice__ {" in line:
                        old_pres = [ar.strip() for ar in line[len(
                            "preprocessor:__choice__ {"):].split("}")[0].split(",")]
            for param in values["search_space"]:
                if param not in olds:
                    olds.append(param)
            for param in values["prep_space"]:
                if param not in old_pres:
                    old_pres.append(param)
            values["search_space"] = olds
            values["prep_space"] = old_pres
            session["values"] = values
    return render_template(
        'running.html',
        url_mod=url_mod,
        turn=0,
        task=values["task"],
        time=values["time"],
        iters=iters,
        PERIOD=format_period,
        RAW_PERIOD=values["period"])


@app.route('/progress_tpot')
def progress_tpot():
    turn = request.args.get('iter', default=0, type=int)
    print("turn", turn)
    values = session.get('values', 'not set')
    target_ft = session.get('target_ft', 'not set')
    checksum = session.get('checksum', 'not set')
    iters = values["time"] // values["period"]
    extra = values["time"] % values["period"]
    format_period = format_time(values["period"])
    metric = gen_metric(values["task"], values["metric"])
    path = os.path.join(app.config['UPLOAD_FOLDER'],
                        session.get("filename", "not set"))
    #TODO: features can be passed from previous calls for optimization
    features = return_cols(path)

    search_space=classifier_config_dict.copy()
    for each in TPOT_CLASSIFIERS:
        if each not in values["selected_opers"]:
            del search_space[each]
    pipeline_optimizer = run_task_tpot(path, values["task"], values["data_type"], values["period"]//60,target_ft,config_dict=search_space)
    res_list=pipeline_optimizer.evaluated_individuals_.keys()
    res=pipeline_optimizer.evaluated_individuals_
    res_list=[ pipe.split('(')[:-1] for pipe in res.keys()]
    col_names=["one","two","three","four","five","six"]
    
    #with open("tmp/tpot_obj/tpot_f", 'wb') as filehandler:
    #    dill.dump(pipeline_optimizer, filehandler)


    turn += 1

    with open("static/data/prog.txt",'r') as f:
        res_data=f.read()

    return render_template("results_tpot.html",res_data=res_data)

      


@app.route('/progress')
def progress():
    turn = request.args.get('iter', default=0, type=int)
    print("turn", turn)
    values = session.get('values', 'not set')
    target_ft = session.get('target_ft', 'not set')
    checksum = session.get('checksum', 'not set')
    iters = values["time"] // values["period"]
    extra = values["time"] % values["period"]
    format_period = format_time(values["period"])
    metric = gen_metric(values["task"], values["metric"])
    path = os.path.join(app.config['UPLOAD_FOLDER'],
                        session.get("filename", "not set"))
    #TODO: features can be passed from previous calls for optimization
    features = return_cols(path)
    estimator = run_task(path, values["task"], values["data_type"], target_ft)
    results = estimator(
        turn,
        values["period"],
        values["search_space"],
        values["prep_space"],
        metric)
    df = pd.DataFrame(data=results).sort_values(by="rank_test_scores")
    col_names = ["Classifier", "{} Max Score".format(
        values["metric"]), "Models Trained", "Show Models"]
    if values["task"] != "classification":
        col_names[0] = "Regressor"
    # Sort list by scores
    res_list = [[a, b]for a, b in zip(
        df["mean_test_score"].values.tolist(), df["params"].values.tolist())]
    # divide list in dictionaries and dump to drive
    grouped_results = {}
    if values["task"] == 'classification':
        ESTIMATORS = CLASSIFIERS
        ESTIMATORS_DISP = CLASSIFIERS_DISP
        for each in CLASSIFIERS:
            grouped_results[each] = []
        for each in res_list:
            grouped_results[each[1]['classifier:__choice__']].append(each)
    else:
        ESTIMATORS = REGRESSORS
        ESTIMATORS_DISP = REGRESSORS_DISP
        for each in REGRESSORS:
            grouped_results[each] = []
        for each in res_list:
            grouped_results[each[1]['regressor:__choice__']].append(each)

    with open("tmp/results.p", 'wb') as filehandler:
        pickle.dump(grouped_results, filehandler)
    res_list = []
    for each in grouped_results.keys():
        if grouped_results[each]:
            res_list.append((ESTIMATORS_DISP[ESTIMATORS.index(each)], round(
                grouped_results[each][0][0], 3), len(grouped_results[each]), "View"))
    res_list.sort(key=lambda x: x[0], reverse=True)
    turn += 1
    # copy tmp files to save for later
    if os.path.exists("tmp_runs/{}".format(checksum)):
        shutil.rmtree("tmp_runs/{}".format(checksum))
    shutil.copytree("tmp/autosk_tmp", "tmp_runs/{}".format(checksum))
    with open("tmp/results.p", 'rb') as filehandler:
        or_list = pickle.load(filehandler)
    estim_dict = {"col_names": [], "disp_index": [],
                  "index": [], "fig_names": [], "res_list": []}
    res_list.sort(key=lambda x: x[1], reverse=True)
    for each in res_list:
        index = ESTIMATORS[ESTIMATORS_DISP.index(each[0])]
        fres_list = or_list[index]
        if values["task"] == 'classification':
            slc = len("classifier:{}:".format(index))
            col_names_e = [x for x in list(fres_list[0][1].keys(
            )) if x[:10] == "classifier" and x[-21:] != "min_impurity_decrease"][1:]
        else:
            slc = len("regressor:{}:".format(index))
            col_names_e = [x for x in list(fres_list[0][1].keys(
            )) if x[:10] == "regressor" and x[-21:] != "min_impurity_decrease"][1:]
        # TODO: 0 if k not in x[1] sets default argumetn to 0, 0 should be
        # replaced with default argument
        fres_list = [
            [
                round(
                    x[0], 3), x[1]["preprocessor:__choice__"].replace(
                    "_", " ").title()] + [
                0 if k not in x[1] else x[1][k] if not isinstance(
                    x[1][k], float) and not isinstance(
                    x[1][k], str) else round(
                    x[1][k], 3) if isinstance(
                    x[1][k], float) else x[1][k].replace(
                    "_", " ").title() for k in col_names_e] + ["Interpret"] for x in fres_list]
        col_names_e = [("{} Score".format(values["metric"])),
                       "Preprocessor"] + [x[slc:].replace("_",
                                                          " ").title() for x in col_names_e] + ["Details"]
        disp_index = index.replace("_", " ").title()
        # plotting
        fig_names = []
        for i in range(1, len(fres_list[0])):
            if isinstance(fres_list[0][i], float) or isinstance(
                    fres_list[0][i], int):
                plt.clf()
                plt.xlabel(col_names_e[i])
                plt.ylabel("{} Score".format(values["metric"]))
                plt.scatter([x[i] for x in fres_list], [x[0]
                                                        for x in fres_list])
                plt.savefig("static/images/figs/" + index + str(i),
                            bbox_inches="tight", transparent=True)
                fig_names.append(index + str(i))
        estim_dict["col_names"].append(col_names_e)
        estim_dict["disp_index"].append(disp_index)
        estim_dict["index"].append(index)
        estim_dict["fig_names"].append(fig_names)
        estim_dict["res_list"].append(fres_list)
    if(turn >= iters):
        return render_template(
            "results.html",
            url_mod=url_mod,
            column_names=col_names,
            row_data=res_list,
            zip=zip,
            len=len,
            CLASSIFIERS=ESTIMATORS,
            CLASSIFIERS_DISP=ESTIMATORS_DISP,
            estim_dict=estim_dict,
            task=values['task'])
    else:
        return render_template(
            "progress.html",
            url_mod=url_mod,
            turn=turn,
            iters=iters,
            PERIOD=format_period,
            RAW_PERIOD=values["period"],
            time=values["time"],
            column_names=col_names,
            row_data=res_list,
            zip=zip,
            CLASSIFIERS=ESTIMATORS,
            CLASSIFIERS_DISP=ESTIMATORS_DISP,
            estim_dict=estim_dict,
            task=values['task'])


@app.route('/stop')
def stop():
    values = session.get('values', 'not set')
    with open("tmp/results.p", 'rb') as filehandler:
        grouped_results = pickle.load(filehandler)
    col_names = ["{} Max Score".format(
        values["metric"]), "Classifier", "Show Models"]
    res_list = []
    for each in grouped_results.keys():
        if grouped_results[each]:
            res_list.append((CLASSIFIERS_DISP[CLASSIFIERS.index(each)], round(
                grouped_results[each][0][0], 3), len(grouped_results[each]), "View"))
    res_list.sort(key=lambda x: x[0], reverse=True)
    estim_dict = {"col_names": [], "disp_index": [],
                  "index": [], "fig_names": [], "res_list": []}
    for each in res_list:
        index = CLASSIFIERS[CLASSIFIERS_DISP.index(each[0])]
        fres_list = grouped_results[index]
        slc = len("classifier:{}:".format(index))
        col_names_e = [x for x in list(fres_list[0][1].keys(
        )) if x[:10] == "classifier" and x[-21:] != "min_impurity_decrease"][1:]
        fres_list = [
            [
                round(
                    x[0], 3), x[1]["preprocessor:__choice__"].replace(
                    "_", " ").title()] + [
                x[1][k] if not isinstance(
                    x[1][k], float) and not isinstance(
                    x[1][k], str) else round(
                    x[1][k], 3) if isinstance(
                    x[1][k], float) else x[1][k].replace(
                    "_", " ").title() for k in col_names_e] + ["Interpret"] for x in fres_list]

        col_names_e = [("{} Score".format(values["metric"])),
                       "Preprocessor"] + [x[slc:].replace("_",
                                                          " ").title() for x in col_names_e] + ["Details"]
        disp_index = index.replace("_", " ").title()
        # plotting
        fig_names = []
        for i in range(1, len(fres_list[0])):
            if isinstance(fres_list[0][i], float) or isinstance(
                    fres_list[0][i], int):
                fig_names.append(index + str(i))
        estim_dict["col_names"].append(col_names_e)
        estim_dict["disp_index"].append(disp_index)
        estim_dict["index"].append(index)
        estim_dict["fig_names"].append(fig_names)
        estim_dict["res_list"].append(fres_list)
    return render_template("results.html", column_names=col_names,
                           estim_dict=estim_dict, row_data=res_list, zip=zip)


@app.route('/estimator')
def view_estimator():
    values = session.get('values', 'not set')
    with open("tmp/results.p", 'rb') as filehandler:
        or_list = pickle.load(filehandler)
    index = request.args.get('model', default=None, type=str)
    res_list = or_list[index]
    slc = len("classifier:{}:".format(index))
    col_names = [x for x in list(res_list[0][1].keys(
    )) if x[:10] == "classifier" and x[-21:] != "min_impurity_decrease"][1:]
    res_list = [
        [
            round(
                x[0], 3), x[1]["preprocessor:__choice__"].replace(
                "_", " ").title()] + [
                    x[1][k] if not isinstance(
                        x[1][k], float) and not isinstance(
                            x[1][k], str) else round(
                                x[1][k], 3) if isinstance(
                                    x[1][k], float) else x[1][k].replace(
                                        "_", " ").title() for k in col_names] + ["Interpret"] for x in res_list]
    col_names = [("{} Score".format(values["metric"])), "Preprocessor"] + \
        [x[slc:].replace("_", " ").title() for x in col_names] + ["Details"]
    disp_index = index.replace("_", " ").title()
    # plotting
    fig_names = []
    for i in range(1, len(res_list[0])):
        if isinstance(res_list[0][i], float) or isinstance(
                res_list[0][i], int):
            plt.clf()
            plt.xlabel(col_names[i])
            plt.ylabel("{} Score".format(values["metric"]))
            plt.scatter([x[i] for x in res_list], [x[0] for x in res_list])
            plt.savefig("static/images/figs/" + index + str(i),
                        bbox_inches="tight", transparent=True)
            fig_names.append(index + str(i))
    return render_template(
        "estimator_results.html",
        column_names=col_names,
        disp_index=disp_index,
        estimator=index,
        fig_names=fig_names,
        row_data=res_list,
        zip=zip)


@app.route('/model')
def view_model():
    with open("tmp/results.p", 'rb') as filehandler:
        res_list = pickle.load(filehandler)
    index = request.args.get('model', default=0, type=int)
    estim = request.args.get('estimator', default=None, type=str)
    model = res_list[estim][index]
    return render_template("model.html", model=model,
                           estimator=estim, model_index=index)


@app.route("/generate_model")
def generate_model():
    # generates model from the parameters and trains the model on the train set
    # Load parameters
    values = session.get('values', 'not set')
    smote = session.get('smote', 'not set')
    smote = "no"  # dont include smote in the pipeline
    target_ft = session.get('target_ft', 'not set')
    features = session.get('features', 'not set')
    index = request.args.get('model', default=0, type=int)
    estim = request.args.get('estimator', default=None, type=str)
    filehandler = open("tmp/results.p", 'rb')
    res_list = pickle.load(filehandler)
    arg_dict = res_list[estim][index][1]
    # constuct and fit pipeline
    param_dict = pipeline_gen.process_dict(arg_dict)
    pipeline_params = [("imputation",SimpleImputer(missing_values=np.nan, strategy='mean')), ("preprocessor", pipeline_gen.build_preprocessor_cl(
        param_dict)), ("classifeir", pipeline_gen.build_classifier(param_dict))]
    if smote == "yes":
        pipeline_params.insert(0, ("smote", SMOTE(random_state=42)))
    pipe = Pipeline(pipeline_params)
    path = os.path.join(app.config['UPLOAD_FOLDER'],
                        session.get("filename", "not set"))
    X, y, data = process_data(path, "csv", target_ft)
    pipe.fit(X, y)
    dump(pipe, 'tmp_files/model_{}_{}.joblib'.format(estim, str(index)))
    with open("tmp_files/model_{}_{}.pickle".format(estim, str(index)), 'wb') as filehandler:
        pickle.dump(pipe, filehandler)
    cl = param_dict["classifier:__choice__"]
    # feature importances
    importance = (pipeline_gen.get_importance(pipe, cl, smote))
    metric_res = pipeline_gen.get_matrix(pipe, X, y, smote)
    if len(importance) > 0:
        imps = [[features[i], round(importance[i], 2)]
                for i in range(len(features))]
        imps = sorted(imps, key=lambda l: l[1], reverse=True)
        plt_features = [x[0] for x in reversed(imps)]
        plt_imps = [x[1] for x in reversed(imps)]
        plt.clf()
        plt.title("Feature Importance")
        plt.ylabel("Feature Name")
        plt.xlabel("Importance")
        plt.barh(plt_features, plt_imps, align='center', height=0.2, color='c')
        plt.savefig("static/images/figs/model_imp",
                    bbox_inches="tight", transparent=True)
    else:
        imps = []
    column_names = ["Metric", "Score"]
    metric_names = ["Accuracy", "Recall", "Precision", "F1"]
    metric_res = [[metric_names[i], round(
        metric_res[i], 3)] for i in range(len(metric_res))]

    # partial dependancy
    partial_fig_names = []
    return render_template(
        "download.html",
        url_mod=url_mod,
        features=features,
        targets=np.unique(y),
        estimator=estim,
        index=index,
        column_names=column_names,
        row_data=metric_res,
        CL_Name=cl,
        metric_res=metric_res,
        zip=zip,
        partial_fig_names=partial_fig_names)


@app.route('/download_joblib')
def download_joblib():
    index = request.args.get('model', default=0, type=int)
    estim = request.args.get('estimator', default=None, type=str)
    return send_from_directory("tmp_files", 'model_{}_{}.joblib'.format(
        estim, str(index)), as_attachment=True)


@app.route('/download_pickle')
def download_pickle():
    index = request.args.get('model', default=0, type=int)
    estim = request.args.get('estimator', default=None, type=str)
    return send_from_directory("tmp_files", 'model_{}_{}.pickle'.format(
        estim, str(index)), as_attachment=True)


@app.route('/plot_pdp')
def plot_pdp():
    path = os.path.join(app.config['UPLOAD_FOLDER'],
                        session.get("filename", "not set"))
    index = request.args.get('model', default=0, type=int)
    estim = request.args.get('estimator', default=None, type=str)
    target_ft = session.get('target_ft', 'not set')
    features = session.get('features', 'not set')
    f1 = request.args.get('f1', default=None, type=str)
    t1 = request.args.get('t1', default=None, type=str)
    X, y, data = process_data(path, "csv", target_ft)
    chosen_class = list(np.unique(y)).index(int(float(t1)))
    with open("tmp_files/model_{}_{}.pickle".format(estim, str(index)), 'rb') as filehandler:
        pipe = pickle.load(filehandler)
    mod_path = "modal_" + "pdp_" + str(f1.replace('.', '_'))
    feat_p = pdp.pdp_isolate(
        model=pipe,
        dataset=data,
        model_features=features,
        feature=f1)
    fig, axes = pdp.pdp_plot(pdp_isolate_out=feat_p, feature_name=f1, center=True, x_quantile=True, plot_lines=True, frac_to_plot=100,
                             show_percentile=False, which_classes=[chosen_class], plot_params={"subtitle": "For Class {}, Label: {}".format(chosen_class, t1)})
    fig.savefig("static/images/figs/" + mod_path,
                bbox_inches="tight", transparent=True)
    plt.figure()
    return render_template("modal_plot.html", plot_name=mod_path)


@app.route('/plot_modal')
def plot_modal():
    path = os.path.join(app.config['UPLOAD_FOLDER'],
                        session.get("filename", "not set"))
    index = request.args.get('model', default=0, type=int)
    estim = request.args.get('estimator', default=None, type=str)
    target_ft = session.get('target_ft', 'not set')
    features = session.get('features', 'not set')
    f1 = request.args.get('f1', default=None, type=str)
    f2 = request.args.get('f2', default=None, type=str)
    t1 = request.args.get('t1', default=None, type=str)
    X, y, data = process_data(path, "csv", target_ft)
    chosen_class = list(np.unique(y)).index(int(float(t1)))
    with open("tmp_files/model_{}_{}.pickle".format(estim, str(index)), 'rb') as filehandler:
        pipe = pickle.load(filehandler)
    mod_path = "modal_" + str(f1.replace('.', '_')) + \
        "_" + str(f2.replace('.', '_'))
    pdp_V1_V2 = pdp.pdp_interact(
        model=pipe, dataset=data, model_features=features, features=[
            f1, f2], num_grid_points=None, percentile_ranges=[
            None, None])
    fig, axes = pdp.pdp_interact_plot(
        pdp_V1_V2, [f1, f2], plot_type='grid', x_quantile=True, ncols=2, plot_pdp=True,
        which_classes=[chosen_class], plot_params={
            "subtitle": "For Class {}, Label: {}".format(chosen_class, t1)}
    )
    fig.savefig("static/images/figs/" + mod_path,
                bbox_inches="tight", transparent=True)
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


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080, debug=True)
