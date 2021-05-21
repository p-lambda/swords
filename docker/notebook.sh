docker exec -it swords \
	jupyter \
		notebook \
		--ip=0.0.0.0 \
		--port 8888 \
		--no-browser \
		--notebook-dir=/home/swords/notebooks
