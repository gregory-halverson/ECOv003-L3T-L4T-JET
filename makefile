colima-start:
	colima start -m 16 -a x86_64 -d 100 

docker-build:
	docker build -t ecov002_l3t_l4t_jet:latest .

docker-build-environment:
	docker build --target environment -t ecov002_l3t_l4t_jet:latest .

docker-build-installation:
	docker build --target installation -t ecov002_l3t_l4t_jet:latest .

docker-interactive:
	docker run -it ecov002_l3t_l4t_jet bash 
