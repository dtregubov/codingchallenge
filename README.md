# codingchallenge
Coding Challenge is a production-ready Python project 
which was built from notebook 'assigment1.ipynb' with BoW model for Text Classification.
The project is ready to run as Docker-based application.

## Deploying (using Docker Compose)

The Makefile is used for convenience with docker-compose. You can do the following by it:

To run the project:
```bash
make run
```

To run the project up as daemon:
```bash
make rund
```

#### Additional commands

To properly sort imports:
```bash
make isort
```


## Project running
To use the project, write next cURL inside terminal:
```bash
curl -X POST -H "Content-Type: application/json" -d '{"sentence": "I love programming"}' http://0.0.0.0:8000/predict
```

## Example of project using scenario:
1) Upload project code from GitHub
2) Open terminal and write 'cd path_to_app' command
3) Run 'make rund' command
4) Write ``` curl -X POST -H "Content-Type: application/json" -d '{"sentence": "I love programming"}' http://0.0.0.0:8000/predict ``` command
