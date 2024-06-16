# codingchallenge
Coding Challenge is a production-ready Python project 
which was built from notebook 'assigment1.ipynb' with BoW model for Text Classification.
The project is ready to run as Docker-based application.

## Deploying (using Docker Compose)

The Makefile is used for convenience with docker-compose. You can do the following by it:

#### Installation

Build application
```bash
make build
```

#### Running
To run the project (you'll need to open new console to run bash):
```bash
make run
```

To run the project up as daemon:
```bash
make rund
```

To get inside the container:
```bash
make bash
```


#### Additional commands

To properly sort imports:
```bash
make isort
```


## Project running
To start the project, write next command inside terminal after 'make bash' command:
```bash
python main.py
```

## Example of project using scenario:
1) Upload project code from GitHub
2) Open terminal and write 'cd path_to_app' command
3) Run 'make rund' command
4) Run 'make bash' command
5) Write 'python main.py' command
6) Wait for a training and write your sentence when "Enter a sentence to predict tag:" question appers
