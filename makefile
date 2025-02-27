DOCKER_IMAGE_NAME = ecov003_l3t_l4t_jet

environment:
	mamba create -y -n ECOv003-L3T-L4T-JET -c conda-forge python=3.10

install-package:
	pip install -e .[dev]

uninstall-package:
	pip uninstall ecov003_l3t_l4t_jet

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
