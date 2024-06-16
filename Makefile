.DEFAULT_GOAL := build

build:
	docker-compose build

run:
	docker-compose up --build

rund:
	docker-compose up --build -d

bash:
	docker-compose exec app /bin/bash

isort:
	docker-compose exec app isort -l120 -m3 --tc $(if $(ISORT_PATH),$(ISORT_PATH), .)
