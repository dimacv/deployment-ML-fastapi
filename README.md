# Deploying a machine learning model (complex scikit-learn pipeline) using FastAPI.
  
## Docker build:

``` docker build -t fastapi . ```  
  
--------------------------------------------------------------------------------

## Starting a named container in daemon mode on port 8000 :

``` docker run --rm -p 8000:8000 --name fastapi -d fastapi ``` 

