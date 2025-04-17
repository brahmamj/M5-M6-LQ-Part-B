FROM python:3.12.9-bullseye

RUN mkdir /housing_api
RUN mkdir /housing_api/app
RUN mkdir /housing_api/app/schemas
RUN mkdir /housing_api/dist
WORKDIR /housing_api
COPY dist/*.whl /housing_api/dist
COPY housing_api/app/*.py /housing_api/app/
COPY housing_api/app/schemas/*.py /housing_api/app/schemas/
COPY housing_api/requirements.txt /housing_api/
RUN pip install -r /housing_api/requirements.txt

CMD ["python","app/main.py"]

