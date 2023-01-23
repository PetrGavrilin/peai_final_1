from diffusers import StableDiffusionPipeline
import torch 
import accelerate
from IPython.display import display
import streamlit as st

def drawing_picture(prompt):
# функция рисует картину по полученном описанию в виде строки
    model_id = "CompVis/stable-diffusion-v1-4" # выбранная предобученная модель
    pipe = StableDiffusionPipeline.from_pretrained(model_id, revision="fp16", torch_dtype=torch.float16)

    has_cuda = torch.cuda.is_available() # проверка запущена ли CUDA
    st.text(has_cuda)
    #pipe = pipe.to('cpu' if not has_cuda else 'cuda')
    #generator = torch.Generator('cpu' if not has_cuda else 'cuda').manual_seed(0)
    
    pipe = pipe.to('cuda')
    generator = torch.Generator('cuda').manual_seed(0)
    
    image = pipe(prompt, guidance_scale=7.5, num_inference_steps=15, generator=generator).images[0]

    return (image)

def load_text():
    text = st.text_input("Enter your text in English")
    return text   
    
    
st.title('Построение изображений в Streamlit') # вывод шапки
text = load_text() # загрузка текста
result = st.button('Нарисовать изображение') # присвоение статуса по нажатию кнопки

if result:
    st.write('**Ожидайте пока искусственный интеллект рисует картину**') 
    st.write(text)
    picture = drawing_picture(text)
    st.write('**Результаты построения изображения по модели CompVis/stable-diffusion-v1-4:**') 
    st.image(picture)
    st.write('**Надеюсь, вам понравилось**') 
