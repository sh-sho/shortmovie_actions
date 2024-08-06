

from functools import wraps
import os
import shutil
import time
from typing import Any, Callable, TypeVar
from moviepy.editor import  VideoFileClip
from azure.storage.blob import BlobServiceClient
import multiprocessing as mp
from multiprocessing import current_process
import numpy as np
from dotenv import load_dotenv, find_dotenv

_ = load_dotenv(find_dotenv())

F = TypeVar('F', bound=Callable[..., Any])
NO_OF_PROCESSORS = mp.cpu_count()
AZURE_STORAGE_CONNECTION_STRING=os.environ["AZURE_STORAGE_CONNECTION_STRING"]

connect_str = AZURE_STORAGE_CONNECTION_STRING
container_name = 'movies'
blob_service_client = BlobServiceClient.from_connection_string(connect_str)
container_client = blob_service_client.get_container_client(container_name)
blob_list = container_client.list_blobs()
output_container_name = 'outputs'
output_container_client = blob_service_client.get_container_client(output_container_name)

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
def download_blob(blob, directory: str) -> None:
    try:
        print(f"current process no.{current_process().pid}")
        if blob.name.endswith('.mov'):
            blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob.name)
            download_file_path = os.path.join(directory, os.path.basename(blob.name))
            with open(download_file_path, "wb") as download_file:
                download_file.write(blob_client.download_blob().readall())
            print(f"Downloading {blob.name} to {download_file_path}")

        elif blob.name.endswith('.MOV'):
            blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob.name)
            download_file_path = os.path.join(directory, os.path.basename(blob.name))
            with open(download_file_path, "wb") as download_file:
                download_file.write(blob_client.download_blob().readall())
            print(f"Downloading {blob.name} to {download_file_path}")

        elif blob.name.endswith('.mp4'):
            blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob.name)
            download_file_path = os.path.join(directory, os.path.basename(blob.name))
            with open(download_file_path, "wb") as download_file:
                download_file.write(blob_client.download_blob().readall())
            print(f"Downloading {blob.name} to {download_file_path}")

        elif blob.name.endswith('.MP4'):
            blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob.name)
            download_file_path = os.path.join(directory, os.path.basename(blob.name))
            with open(download_file_path, "wb") as download_file:
                download_file.write(blob_client.download_blob().readall())
            print(f"Downloading {blob.name} to {download_file_path}")
    except Exception as e:
        print("Error download blob", e)

@timer
def download_all_file(directory: str) -> None:
    """
    Download all files and convert to mp4
    """
    try:
        args = [(blob, directory) for blob in blob_list]
        with mp.Pool(processes=NO_OF_PROCESSORS) as pool:
            pool.starmap(download_blob, args)
    except Exception as e:
        print("Error downloading mp4", e)

    try:
        args = [(filename, directory) for filename in os.listdir(directory)]
        with mp.Pool(processes=NO_OF_PROCESSORS) as pool:
            pool.starmap(convert_mov_to_mp4, args)
    except Exception as e:
        print("Error converting mov to mp4", e)

@timer
def convert_mov_to_mp4(filename: str, directory: str):
    """
    Converts mov to mp4 by multiprocessing

    Args:
        filename (str): filename of mov
        directory (str): directory of mov
    """
    try:
        print(f"current process no.{current_process().pid}")
        if filename.endswith('.MOV') or filename.endswith('.mov'):
            input_path = os.path.join(directory, filename)
            output_path = os.path.join(directory, os.path.splitext(filename)[0] + '.mp4')
            print(f"In Process: {current_process().pid}, Converting: {input_path} -> {output_path}")
            video_clip = VideoFileClip(input_path)
            width = video_clip.size[0]
            height = video_clip.size[1]
            rotation = video_clip.rotation
            print(f"{input_path} video_clip.size: width{width}, height{height}, rotation{rotation}")
            # if width < height:
            #     video_clip.write_videofile(output_path, codec='libx264', ffmpeg_params=["-vf", f"scale={height}:{width}"])
            # else:
            #     video_clip.write_videofile(output_path, codec='libx264', ffmpeg_params=["-vf", f"scale={width}:{height}"])
            # if rotation == 90:
            video_clip.write_videofile(output_path, codec='libx264', ffmpeg_params=["-vf", f"scale={height}:{width}"])
            # else:
            #     video_clip.write_videofile(output_path, codec='libx264', ffmpeg_params=["-vf", f"scale={width}:{height}"])
            # video_clip_resized = video_clip.resize(height=720)
            # video_clip_resized.write_videofile(output_path, codec='libx264')
            print(f"Converted: {input_path} -> {output_path}")
        else:
            print(f"Skipping: {filename}")
    except Exception as e:
        print("Error converting mov to mp4", e)

@timer
def get_blob(blob, directory: str):
    """
    get a blob from azure blob

    Args:
        blob (BlobProperties): blob
        directory (str): directory
    """
    print(f"current process no.{current_process().pid}")
    blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob.name)
    download_file_path = os.path.join(directory, os.path.basename(blob.name))

    with open(download_file_path, "wb") as download_file:
        download_file.write(blob_client.download_blob().readall())
    print(f"Downloading {blob.name} to {download_file_path}")

@timer
def download_file_startswith(directory: str, start_name: str) -> None:
    try:
        blob_list = container_client.list_blobs(name_starts_with=start_name)
        args = [(blob, directory) for blob in blob_list]
        with mp.Pool(processes=NO_OF_PROCESSORS) as pool:
            pool.starmap(get_blob, args)
    except Exception as e:
        print("Error downloading mp4", e)

    try:
        args = [(filename, directory) for filename in os.listdir(directory)]
        with mp.Pool(processes=NO_OF_PROCESSORS) as pool:
            pool.starmap(convert_mov_to_mp4, args)
    except Exception as e:
        print("Error converting mov to mp4", e)


@timer
def upload_all_file(directory: str) -> None:
    for filename in os.listdir(directory):
        with open(directory + '/' + filename, "rb") as file:
            blob_client = blob_service_client.get_blob_client(container=container_name, blob=filename)
            blob_client.upload_blob(file, overwrite=True)
            print("\nUploading to Azure Storage as blob:\n\t" + filename)

@timer
def upload_file(directory: str, filename: str) -> None:
    with open(directory + '/' + filename, "rb") as file:
        blob_client = blob_service_client.get_blob_client(container=output_container_name, blob=filename)
        blob_client.upload_blob(file, overwrite=True)
        print("\nUploading to Azure Storage as blob:\n\t" + filename)
