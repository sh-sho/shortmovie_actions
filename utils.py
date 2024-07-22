

from functools import wraps
import os
import shutil
import time
from typing import Any, Callable, TypeVar
from moviepy.editor import  VideoFileClip
from azure.storage.blob import BlobServiceClient
import numpy as np
from dotenv import load_dotenv, find_dotenv

_ = load_dotenv(find_dotenv())

F = TypeVar('F', bound=Callable[..., Any])
AZURE_STORAGE_CONNECTION_STRING=os.environ["AZURE_STORAGE_CONNECTION_STRING"]

connect_str = AZURE_STORAGE_CONNECTION_STRING
container_name = 'movies'
blob_service_client = BlobServiceClient.from_connection_string(connect_str)
container_client = blob_service_client.get_container_client(container_name)
blob_list = container_client.list_blobs()

def timer(func: F) -> None:
    """Any functions wrapper for calculate execution time"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        elapsed_time = time.time() - start
        print(f"{func.__name__} took a {elapsed_time}s")
        return result
    return wrapper

def delete_files(directory: str, file_type: str) -> None:
    """ Delete files """
    files = os.listdir(directory)
    try:
        for file in files:
            if file.endswith(file_type):
                os.remove(os.path.join(directory, file))
    except FileNotFoundError:
        print("no target files")
    except Exception as e:
        print("Error delete files", e)

def dir_check(directory: str, file_type: str) -> None:
    """ Check directory """
    if os.path.exists(directory):
        try:
            delete_files(directory, file_type)
            print(f"success delete files in {directory}")
        except FileNotFoundError:
            print("no target files")
        except Exception as e:
            print("Error delete files", e)
    else:
        try:
            os.makedirs(directory, exist_ok=True)
        except Exception as e:
            print("Error make dirctory ", e)

def delete_dir(directory: str) -> None:
    """ Delete directory """
    if os.path.exists(directory):
        try:
            shutil.rmtree(directory)
            print("success delete directory")
        except Exception as e:
            print("Error delete directory", e)

@timer
def download_all_file(directory: str, file_type: str) -> None:
    try:
        for blob in blob_list:
            if blob.name.endswith(file_type):
                blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob.name)
                download_file_path = os.path.join(directory, os.path.basename(blob.name))

                with open(download_file_path, "wb") as download_file:
                    download_file.write(blob_client.download_blob().readall())
                print(f"Downloading {blob.name} to {download_file_path}")
    except Exception as e:
        print("Error downloading mp4", e)

    try:
        for filename in os.listdir(directory):
            if filename.endswith('.MOV') or filename.endswith('.mov'):
                input_path = os.path.join(directory, filename)
                output_path = os.path.join(directory, os.path.splitext(filename)[0] + '.mp4')
                convert_mov_to_mp4(input_path, output_path)
                print(f"Converted: {input_path} -> {output_path}")
    except Exception as e:
        print("Error converting mov to mp4", e)

def convert_mov_to_mp4(input_path, output_path):
    video_clip = VideoFileClip(input_path)
    video_clip_resized = video_clip.resize(height=720)
    video_clip_resized.write_videofile(output_path, codec='libx264')
    
@timer
def download_file_startswith(directory: str, start_name: str) -> None:
    try:
        blob_list = container_client.list_blobs(name_starts_with=start_name)
        for blob in blob_list:
            blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob.name)
            download_file_path = os.path.join(directory, os.path.basename(blob.name))

            with open(download_file_path, "wb") as download_file:
                download_file.write(blob_client.download_blob().readall())
            print(f"Downloading {blob.name} to {download_file_path}")
    except Exception as e:
        print("Error downloading mp4", e)

    try:
        for filename in os.listdir(directory):
            if filename.endswith('.MOV') or filename.endswith('.mov'):
                input_path = os.path.join(directory, filename)
                output_path = os.path.join(directory, os.path.splitext(filename)[0] + '.mp4')
                convert_mov_to_mp4(input_path, output_path)
                print(f"Converted: {input_path} -> {output_path}")
    except Exception as e:
        print("Error converting mov to mp4", e)

@timer
def upload_file(directory: str, filename: str) -> None:
    with open(directory + '/' + filename, "rb") as file:
        blob_client = blob_service_client.get_blob_client(container=output_container_name, blob=filename)
        blob_client.upload_blob(file, overwrite=True)
        print("\nUploading to Azure Storage as blob:\n\t" + filename)
