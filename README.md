# iSmartML

iSmartML is an interactive and user-guided framework for improving the utility and usability of the AutoML
process. The framework provides the end-user with a userfriendly configuration control panel that allows nontechnical 
users and domain experts (e.g., physicians) to easily define, configure and control the search
space for the AutoML search process according to
their own preferences.

<img width="905" alt="Panel" src="https://user-images.githubusercontent.com/8884249/68866376-fcbd2180-06fc-11ea-9a53-7e5a0fec5d7f.PNG">

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

### Prerequisites


```
sudo apt-get install build-essential swig
```

### Installing

Install iSmartML locally

Clone repository and install dependencies
```
git clone https://github.com/DataSystemsGroupUT/ismartml.git
cd ismartml
pip install -r requirements.txt 
```


## Running

Run locally
```
python main.py
```
The tool should be avilable at
```
http://localhost:8080/
```

Or deploy with a web server



