.DEFAULT_GOAL := build

run:
	docker-compose up --build

bash:
	docker-compose run --rm app /bin/bash

isort:
	docker-compose run --rm app isort -l120 -m3 --tc $(if $(ISORT_PATH),$(ISORT_PATH), .)
