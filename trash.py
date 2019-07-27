
@app.route('/run_optimize')
def run_optimize():
    values=session.get('values', 'not set')
    iters=values["time"]//values["period"]
    extra=values["time"]%values["period"]
    format_period=format_time(values["period"])
    estimator=run_task(os.path.join(app.config['UPLOAD_FOLDER'], values["filename"]),values["task"],values["data_type"])
    results=estimator(0,values["period"],values["search_space"],values["prep_space"])
    df=pd.DataFrame(data=results).sort_values(by="rank_test_scores")
    col_names=["Score","Estimator","Preprocessing","Details"]
    res_list = [[a,b]for a, b in zip(df["mean_test_score"].values.tolist(),df["params"].values.tolist())]
    #session["results"]=res_list
    filehandler = open("tmp/results.p", 'wb') 
    pickle.dump(res_list, filehandler)
    
    if(values["task"]=="classification"):
        res_list=[[row[0],row[1]["classifier:__choice__"],row[1]["preprocessor:__choice__"],"view"] for row in res_list]
    else:
        res_list=[[row[0],row[1]["regressor:__choice__"],row[1]["preprocessor:__choice__"],"view"] for row in res_list]
    if iters<=1:
        return render_template("results.html",column_names=col_names, row_data=res_list,zip=zip)
    else:
        return render_template("progress.html",turn=values["turn"],iters=iters,PERIOD=format_period,task=values["task"],time=values["time"],column_names=col_names, row_data=res_list,zip=zip)


