from datetime import datetime
import os, time, base64
import pandas as pd
import numpy as np
import re
from PIL import Image
from pymongo import MongoClient, IndexModel
from starlette.middleware.cors import CORSMiddleware
from moviepy.editor import  VideoFileClip, ColorClip, CompositeVideoClip, concatenate_videoclips
from moviepy.audio.io.AudioFileClip import AudioFileClip
from langchain_core.messages import HumanMessage
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
import utils as ul
# from dotenv import load_dotenv, find_dotenv
from fastapi import FastAPI
import uvicorn

# _ = load_dotenv(find_dotenv())

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,   # 追記により追加
    allow_methods=["*"],      # 追記により追加
    allow_headers=["*"]       # 追記により追加
)

# Chat AzureOpenAI
AZURE_OPENAI_ENDPOINT=os.environ["AZURE_OPENAI_ENDPOINT"]
AZURE_OPENAI_API_KEY=os.environ["AZURE_OPENAI_API_KEY"]
AZURE_OPENAI_API_VERSION=os.environ["AZURE_OPENAI_API_VERSION"]
AZURE_OPENAI_CHAT_DEPLOYMENT_NAME=os.environ["AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"]

MONGODB_CONNECTION=os.environ["MONGODB_CONNECTION"]
AZURE_STORAGE_CONNECTION_STRING=os.environ["AZURE_STORAGE_CONNECTION_STRING"]
MOVIE_DIRECTORY_PATH = os.environ["MOVIE_DIRECTORY_PATH"]
SPLIT_MOVIE_DIRECTORY_PATH = os.environ["SPLIT_MOVIE_DIRECTORY_PATH"]
OUTPUT_DIRECTORY = os.environ["OUTPUT_DIRECTORY"]

# Azure OpenAI
chat_model = AzureChatOpenAI(
    api_version=AZURE_OPENAI_API_VERSION,
    azure_deployment=AZURE_OPENAI_CHAT_DEPLOYMENT_NAME
)

embeddings_model = AzureOpenAIEmbeddings(
    azure_deployment="text-embedding-small",
    api_version=AZURE_OPENAI_API_VERSION,
)

# Mongodb
mongo_client = MongoClient(MONGODB_CONNECTION)
collection = mongo_client["doc"]["images"]

movie_path = MOVIE_DIRECTORY_PATH
audio_file = './music/chill.mp3'
output_video = 'output_with_audio.mp4'
tmp_output_video = 'output_with_audio'
image_directory = SPLIT_MOVIE_DIRECTORY_PATH + "/tmp_horizontal"
k_num = 10

def save_frames_at_intervals(video_path: str, interval_sec: int, output_dir: str, base_filename: str):
    try:
        video = VideoFileClip(video_path)
        duration_sec = video.duration

        width, height = video.size
        if height > width:
            output_dir = os.path.join(output_dir, "tmp_vertical")
        else:
            output_dir = os.path.join(output_dir, "tmp_horizontal")

        os.makedirs(output_dir, exist_ok=True)

        current_sec = 0
        frame_idx = 0
        digit = len(str(int(duration_sec // interval_sec) + 1))

        while current_sec < duration_sec:
            frame = video.get_frame(current_sec)
            output_filename = '{}_{}_{:.2f}.jpg'.format(base_filename, str(frame_idx).zfill(digit), current_sec)
            output_path = os.path.join(output_dir, output_filename)

            frame_image = Image.fromarray(frame)
            frame_image.save(output_path)
            print(f"Saved frame at {current_sec} sec to {output_path}")

            current_sec += interval_sec
            frame_idx += 1

        video.close()

    except Exception as e:
        print(f"Error processing video {video_path}: {e}")

def split_movies() -> None:
    ul.dir_check(SPLIT_MOVIE_DIRECTORY_PATH + "/tmp_vertical", ".jpg")
    ul.dir_check(SPLIT_MOVIE_DIRECTORY_PATH + "/tmp_horizontal", ".jpg")
    ul.dir_check(SPLIT_MOVIE_DIRECTORY_PATH, ".jpg")
    movie_files = os.listdir(MOVIE_DIRECTORY_PATH)
    print(movie_files)
    try:
        for movie_file in movie_files:
            if movie_file.endswith(".mp4"):
                movie_path = os.path.join(MOVIE_DIRECTORY_PATH, movie_file)
                base_filename = movie_file.replace(".mp4", "")
                save_frames_at_intervals(movie_path, 10, SPLIT_MOVIE_DIRECTORY_PATH, base_filename)
                print(f"split {MOVIE_DIRECTORY_PATH}/{movie_file}")
    except Exception as e:
        print("Error split movies", e)

def encode_image(image_path: str):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

@ul.timer
def get_start_time(image_file: str):
    match = re.search(r"(\d+\.\d+)", image_file)
    if match:
        value = float(match.group(1))
        print(value)
        return(value)
    else:
        print("No start time")
        return None

@ul.timer
def image_to_text(image_file: str) -> str:
    image_path = f"{image_directory}/{image_file}"
    base64_image = encode_image(image_path)
    
    message = HumanMessage(
        content=[
            {"type": "text", "text": "Describe this image:"},
                {"type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}"
                    }
                }
            ]
    )
    response = chat_model.invoke([message])
    print(response.content)
    return response.content

ul.timer
def vector_search_images(message: str) -> np.ndarray:
    query_vector = embeddings_model.embed_query(message)
    search_query = {
        "$search": {
            "cosmosSearch": {
                "vector": query_vector,
                "path": "vector",
                "k": k_num
            },
            "returnStoredSource": True
        }
    }

    project_stage = {
        "$project": {
            "similarityScore": {"$meta": "searchScore"},
            "document": "$$ROOT"
        }
    }

    try:
        results = collection.aggregate([search_query, project_stage])
    except Exception as e:
        print("Error aggregate MongoDB", e)

    result_lists = pd.DataFrame(columns=['name', 'start_time'])
    try:
        for result in results:
            new_row = pd.DataFrame({'name': [result['document']['name']], 'start_time': result['document']['start_time']})
            result_lists = pd.concat([result_lists, new_row], ignore_index=True)
            print(f"タイトル:{result['document']['name']}")
            print(f"類似度:{result['similarityScore']}")
            print(f"開始時間:{result['document']['start_time']}")
    except Exception as e:
        print("Error list result", e)
    print(result_lists)
    return result_lists

def resize_and_pad_clip(clip, target_width: int, target_height: int):
    clip = clip.resize(height=target_height) if clip.size[1] > clip.size[0] else clip.resize(width=target_width)
    bg_clip = ColorClip(size=(target_width, target_height), color=(0, 0, 0))
    clip = clip.set_position(("center", "center"))
    clip = CompositeVideoClip([bg_clip, clip])
    clip.write_videofile('test.mp4', codec='libx264', audio_codec='aac')
    print(type(clip))
    return clip

def download_blob_to_local(directory: str, video_file: np.ndarray) -> None:
    try:
        for file in video_file:
            ul.download_file_startswith(directory=directory, start_name=file)
            print(f"Downloading movie from Azure Blob: {file}")
    except Exception as e:
        print("Error download blob to local", e)

ul.timer
def generate_movie(images: np.ndarray, step_sec: int) -> None:
    df_images = pd.DataFrame(images, columns=['name', 'start_time'])
    df_images = df_images.sort_values(by='name').reset_index(drop=True)

    print(f'sorted: {df_images}')
    df_images['name'] = df_images['name'].apply(lambda x: re.sub(r'_\d+_\d+\.\d+', '', x).replace('.jpg', ''))
    print(f'edited: {df_images}')
    
    download_blob_to_local(movie_path, df_images['name'])
    
    video_clips = []
    for idx, row in df_images.iterrows():
        video_file = movie_path + '/' + row['name'] + '.mp4'
        clip = VideoFileClip(video_file)
        duration = clip.duration
        start_time = row['start_time']
        if duration <= start_time:
            clip = clip.subclip(0, 1)
        elif duration < start_time + step_sec:
            clip = clip.subclip(start_time, duration)
        else:
            clip = clip.subclip(start_time, start_time + step_sec)
        video_clips.append(clip)
    
    if len(video_clips) > 1:
        final_clip = concatenate_videoclips(video_clips)
    elif len(video_clips) == 1:
        final_clip = video_clips[0]
        print(final_clip.duration)
    else:
        print("No valid video clips to concatenate.")
        final_clip = None

    # audio_clip = AudioFileClip(audio_file)
    # audio_clip = audio_clip.subclip(0, final_clip.duration)
    # final_clip = final_clip.set_audio(audio_clip)
    now = datetime.now()
    timestamp = now.strftime("%Y%m%d_%H%M%S")
    output_video = f'output_with_audio_{timestamp}.mp4'
    final_clip.write_videofile(OUTPUT_DIRECTORY + '/' + output_video, codec='libx264', audio_codec='aac')
    final_clip.write_videofile(OUTPUT_DIRECTORY + '/' + output_video, codec='libx264', audio_codec='aac')
    ul.upload_file(OUTPUT_DIRECTORY, output_video)
    for clip in video_clips:
        clip.reader.close()

    print(f"Video with audio saved as {output_video}")

def enrich_message(message: str) -> str:
    prompt = HumanMessage(
        content=[
            {"type": "text", "text": """
                Task: Convert the text into a specific sentence.
                Sample: 
                'In the background, there are tall, rugged mountains with some tree cover, enhancing the scenic beauty of the location. The presence of a church tower with a pointed roof adds a historic and cultural touch to the town. The overall scene is serene, capturing the essence of a peaceful lakeside settlement.'
                'The image shows a young child seated in a high chair. The child is wearing a bib with the brand "BabyBjörn" printed on it. The bib is yellow and appears to be made of a soft, flexible material designed to catch food. The child is dressed in a plaid shirt and a navy blue sweater with white stars. In the background, there are some household items including a curtain and a piece of furniture with various small objects on it, such as a small plant and possibly toys.'
            """},
            {"type": "text", "text": message}
            ]
    )
    response = chat_model.invoke([prompt])
    print(f"Enrich input: {response.content}")
    return response.content

async def hello_world():
    # time.sleep(10)
    print('hello')
    with open('/home/azureuser/output.txt', 'w') as f:
        f.write("このテキストがファイルに書き込まれます。\n")
        f.write("2行目のテキストも追加されます。")

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.get("/start_image_to_vector")
async def start_image_to_vector():
    # image_to_vector()
    hello_world()
    return 'success'

# @app.get("/image_to_vector")
async def image_to_vector():
    start_time = time.time()
    ul.dir_check(MOVIE_DIRECTORY_PATH, ".mp4")
    ul.download_all_file(directory=MOVIE_DIRECTORY_PATH)
    ul.upload_all_file(directory=MOVIE_DIRECTORY_PATH)
    split_movies()

    image_files = [f for f in os.listdir(image_directory) if f.lower().endswith('.jpg')]
    columns = ['name','start_time', 'text', 'vector']
    df = pd.DataFrame(columns=columns)

    df['name'] = image_files
    print(image_files)
    df['start_time'] = df['name'].apply(lambda x: get_start_time(x))
    df['text'] = df['name'].apply(lambda x : image_to_text (x))
    df['vector'] = df['text'].apply(lambda x : embeddings_model.embed_query(x))
    print(df)

    # index_hol_key = {"vector": "cosmosSearch"}
    # index_hol_options = {
    #     'name': 'vectorSearchIndexHol',
    #     'cosmosSearchOptions': {
    #         'kind': 'vector-ivf',
    #         'numLists': 1,
    #         'similarity': 'COS',
    #         'dimensions': 1536
    #     }
    # }

    # index_ver_key = {"vector_ver", "cosmosSearch"}
    # index_ver_options = {
    #     'name': 'vectorSearchIndexVer',
    #     'cosmosSearchOptions': {
    #         'kind': 'vector-ivf',
    #         'numLists': 1,
    #         'similarity': 'COS',
    #         'dimensions': 1536
    #     }
    # }

    # collection.drop_index('vectorSearchIndexHol')
    # collection.create_index([(k, v) for k, v in index_hol_key.items()], **index_hol_options)
    collection.delete_many({})
    for index, row in df.iterrows():
        doc = {
            'name': row['name'],
            'start_time': row['start_time'],
            'text': row['text'],
            'vector': row['vector']
        }
        collection.insert_one(doc)
    
    print(collection.find_one({}))
    
    ul.delete_dir(MOVIE_DIRECTORY_PATH)
    end_time = time.time()
    print(f"Total time: {end_time - start_time}s")


@app.get("/start_vector_search/{message}")
async def start_vector_search(
    message: str
):
    vector_search(message=message)
    return 'success'

async def vector_search(
    message: str
):
    start_time = time.time()
    ul.dir_check(movie_path, ".mp4")
    ul.dir_check(movie_path, ".MOV")
    ul.dir_check(movie_path, ".mov")
    ul.dir_check(OUTPUT_DIRECTORY, ".mp4")
    
    # message='In the background, there are tall, rugged mountains with some tree cover, enhancing the scenic beauty of the location. The presence of a church tower with a pointed roof adds a historic and cultural touch to the town. The overall scene is serene, capturing the essence of a peaceful lakeside settlement.'
    # message='子供が公園の芝生の上で遊んでいる画像'
    message = enrich_message(message)
    images = vector_search_images(message=message)
    generate_movie(images=images, step_sec=6)
    
    ul.delete_dir(movie_path)
    ul.delete_dir(OUTPUT_DIRECTORY)
    end_time = time.time()
    print(f"Total time: {end_time - start_time}s")

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, log_level="debug", reload=True)
