<html>
    <head>
	<title>Running</title>
	<script src="//ajax.googleapis.com/ajax/libs/jquery/2.1.1/jquery.min.js"></script>
        <script>
            $(function () { 
                //$("#content").load("/run_optimize"); 
		window.location.replace("/progress?iter");
            });
        </script>
	<script> 
	var t=0;
	var x = setInterval(function() { 
	//var now = new Date().getTime(); 
	t+=1; 
	var days = Math.floor(t / ( 60 * 60 * 24)); 
	var hours = Math.floor((t%(60 * 60 * 24))/( 60 * 60)); 
	var minutes = Math.floor((t % (60 * 60)) / ( 60)); 
	var seconds = Math.floor((t % ( 60)) ); 

	if(minutes<1){
	document.getElementById("timer").innerHTML = 
			 seconds + "s "; 
	}

	else if(hours<1){
	document.getElementById("timer").innerHTML =  
			 minutes + "m " + seconds + "s "; 
	}

	else if(days<1){
	document.getElementById("timer").innerHTML =   
			 hours + "h " + minutes + "m " + seconds + "s "; 
	}
	else{
	document.getElementById("timer").innerHTML = days + "d "  
			+ hours + "h " + minutes + "m " + seconds + "s "; 
	}
	}, 1000); 
	</script> 
	 
    </head>
    <body>
	    <div id="content">
		    <h1>Running {{task}} task for {{time}} seconds... 1/{{iters}}</h1>
		    <h1><span>Current Period: </span><span id="timer"></span><span>/{{PERIOD}}</span></h1>
            <button type="button" value=Cancel name="Cancel" onclick="window.location.replace('{{ url_for( 'upload_file' )}}');">
                Cancel
            </button>
	    </div>
    </body>
</html>
