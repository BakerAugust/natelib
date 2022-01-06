docker build -t myimage .
docker run -it --expose 8888 --publish=127.0.0.1:8888:8888/tcp myimage bash
run jupyter notebook --ip 0.0.0.0 --no-browser --allow-root