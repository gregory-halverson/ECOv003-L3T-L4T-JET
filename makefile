DOCKER_IMAGE_NAME = ecov003_l3t_l4t_jet

clean:
	rm -rf *.o *.out *.log
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	rm -rf .pytest_cache
	find . -type d -name "__pycache__" -exec rm -rf {} +

test:
	pytest

build:
	python -m build

twine-upload:
	twine upload dist/*

dist:
	make clean
	make build
	make twine-upload

remove-environment:
	mamba env remove -y -n rasters

install:
	pip install -e .[dev]

uninstall:
	pip uninstall ecov003_l3t_l4t_jet

reinstall:
	make uninstall
	make install

environment:
	mamba create -y -n ECOv003-L3T-L4T-JET -c conda-forge python=3.10

colima-start:
	colima start -m 16 -a x86_64 -d 100 

docker-build:
	docker build -t $(DOCKER_IMAGE_NAME):latest .

docker-build-environment:
	docker build --target environment -t $(DOCKER_IMAGE_NAME):latest .

docker-build-installation:
	docker build --target installation -t $(DOCKER_IMAGE_NAME):latest .

docker-interactive:
	docker run -it $(DOCKER_IMAGE_NAME) fish 

docker-remove:
	docker rmi -f $(DOCKER_IMAGE_NAME)
