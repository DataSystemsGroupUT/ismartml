def format_time(secs):
    days=(secs//(60*60*24))
    hours=(secs%(60*60*24))//(60*60)
    minutes=(secs%(60*60))//(60)
    seconds=(secs%(60))
    if(minutes<1):
        return str(seconds)+"s "
    elif(hours<1):
        return str(minutes)+"m "+str(seconds)+"s "
    elif(days<1):
        return str(hours)+"h "+str(minutes)+"m "+str(seconds)+"s "
    else:
        return str(days)+"d "+str(hours)+"h "+str(minutes)+"m "+str(seconds)+"s "

