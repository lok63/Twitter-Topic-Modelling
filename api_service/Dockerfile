FROM tiangolo/uvicorn-gunicorn-fastapi:python3.8

RUN pip install --upgrade pip

COPY requirements.txt .

# install dependencies
RUN python -m pip install -r requirements.txt

# copy the content of the local src directory to the working directory
COPY ./ .

# command to run on container start
ENV PORT=5001
EXPOSE 5001

#CMD bash run.sh
ENTRYPOINT ["python"]
CMD [ "-u", "./main.py" ]
