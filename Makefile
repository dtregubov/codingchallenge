.DEFAULT_GOAL := build

build:
	docker-compose build

run:
	docker-compose up --build

rund:
	docker-compose up --build -d

isort:
	docker-compose exec app isort -l120 -m3 --tc $(if $(ISORT_PATH),$(ISORT_PATH), .)

test:
	docker-compose run --rm app pytest
