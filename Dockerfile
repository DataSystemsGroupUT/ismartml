#Use ubuntu 18.04
FROM ubuntu:18.04

COPY . /ismartml
WORKDIR /ismartml

#Install Python 3.7
Run apt --yes --force-yes update
Run apt --yes --force-yes install python3.7 python3-pip
Run python3.7 -m pip install pip

#Install Dependencies
Run apt --yes --force-yes install git swig build-essential libssl-dev libffi-dev python3.7-dev 



Run python3.7 -m pip install numpy==1.16.4
Run python3.7 -m pip install -r requirements.txt
Run python3.7 -m pip install scikit-learn==0.21.3

Run echo "key = 'key'" > key.py
Run mkdir data
Run mkdir uploads
Run mkdir static/images/figs
Run mkdir tmp_files

CMD python3.7 main.py 
