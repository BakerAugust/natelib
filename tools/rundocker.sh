cp ~/.pip/pip.conf .
docker build -t myimage .
rm pip.conf
docker run -v ~/.aws:/root/.aws -v ~/repos/natelib/src/:/src -it myimage -p 8888:8888 bash