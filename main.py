from diffusers import StableDiffusionPipeline
import torch 
import accelerate
from IPython.display import display

def drawing_picture(prompt):
# функция рисует картину по полученном описанию в виде строки
    model_id = "CompVis/stable-diffusion-v1-4" # выбранная предобученная модель
    pipe = StableDiffusionPipeline.from_pretrained(model_id, revision="fp16", torch_dtype=torch.float16)

    has_cuda = torch.cuda.is_available() # проверка запущена ли CUDA
    pipe = pipe.to('cpu' if not has_cuda else 'cuda')
    generator = torch.Generator('cpu' if not has_cuda else 'cuda').manual_seed(0)
    image = pipe(prompt, guidance_scale=7.5, num_inference_steps=15, generator=generator).images[0]

    return (image)