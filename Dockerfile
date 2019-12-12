FROM pytorch/pytorch:1.3-cuda10.1-cudnn7-runtime

WORKDIR /usr/src/app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD [ "python", "./qlearning.py" ]