# iSmartML

**iSmartML** is an interactive and user-guided framework for improving the utility and usability of the *AutoML* process with the following main features:

- The framework provides the end-user with a user-friendly configuration *control panel* that allows nontechnical users and domain experts (e.g., physicians) to easily define, configure and control the search space for the AutoML search process according to their own preferences.

- The framework is equipped with a *recommendation engine*, that uses a meta-learning mechanism, to help the end-users on defining the effective search space for the input dataset, potentially useful pre-processors and accurately estimating the time budget.

- The framework provides the end-user with a *monitoring panel* that allows tracking the progress of the search process during the whole allocated time budget and reports a stream of model configurations by sending alerts whenever a better pipeline is found during any point of time through the search process.

- The framework is equipped with a *logging* mechanism which enables storing the results of the explored configurations over a given dataset on one run so that repeated runs on the same dataset can be more effective by avoiding re-exploring the same candidate configurations on the search space.

- The framework is equipped with an *explanation module* which allows the end-user to understand and diagnose the design of the returned machine learning models using various explanation techniques. In particular, the explanation module allows the end-user to choose the model with the best satisfactory explanation for a higher trust or to use the information of the explanation process to refine and optimize a new iteration of the automated search process.

<p align="center">
<img alt="architecture" src="https://user-images.githubusercontent.com/8884249/68950788-86d0bd00-07c5-11ea-8b91-cab51811cc2b.png">
</p>

## Demo

The tool is avilable at https://bigdata.cs.ut.ee/ismartml/

[![Video Demo for iSmartML](http://img.youtube.com/vi/aug5UXd1dNI/0.jpg)](http://www.youtube.com/watch?v=aug5UXd1dNI "iSmartML")

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

## Using Docker(Recommended)
### From dockerhub
```
docker run -p 8080:80 shotaa/ismartml
```
iSmartML should be avilable at http://localhost:8080/ in your browser


### Alternatively Build Image

```
git clone https://github.com/DataSystemsGroupUT/ismartml.git
docker build -t="ismartml" 
docker run -p 8080:80  -t ismartml 
```


## Without Docker

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


### Running

Run locally
```
python main.py
```
The tool should be avilable at
```
http://localhost:8080/
```

Or deploy with a web server



