import io
import os

import torch
from diffusers import StableDiffusionPipeline
from dotenv import load_dotenv
from fastapi import FastAPI, Response
from fastapi.responses import RedirectResponse
from pydantic import BaseModel


class Prompt(BaseModel):
    prompt: str | None = None


load_dotenv()
TOKEN = os.environ.get("SD_TOKEN", "")

pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", use_auth_token=TOKEN,
                                               torch_dtype=torch.float16, revision="fp16")
device = "cuda" if torch.cuda.is_available() else "cpu"
pipe.to(device)

app = FastAPI()


@app.get("/")
async def home():
    return RedirectResponse("/docs", status_code=302)


@app.post("/sdimage")
async def sd_image_gen(prompt_content: Prompt):
    """
    Receives a prompt in json format, returns an image generated using stable diffusion
    :param prompt_content:
    :return:
    """
    prompt = prompt_content.prompt
    image = pipe(prompt)["sample"][0]
    buf = io.BytesIO()
    image.save(buf, format("PNG"))
    image = buf.getvalue()
    return Response(content=image, media_type="image/png")
