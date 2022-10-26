FROM python:3

RUN mkdir /api
RUN cd /api

RUN pip install torch torchvision torchaudio
RUN pip install diffusers transformers scipy
RUN pip install fastapi[all] python-dotenv

COPY . .

CMD ["uvicorn", "main:app", "--host", "0.0.0.0"]