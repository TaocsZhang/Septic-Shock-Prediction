#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import boto3
import pandas as pd
import numpy as np
import pickle
import json
import io
from warnings import warn
from os.path import isfile, join
from pathlib import Path
from tqdm import tqdm
import platform


class FileHandler:

    def __init__(self, bucketname="datarwe-ml-data", root="../../data/", cache=True, verbose=True):
        self._verbose = verbose
        self._s3 = boto3.resource('s3')
        self._client = boto3.client('s3')
        self._bucketname = bucketname
        self._bucket = self._s3.Bucket(name=self._bucketname)
        self.cache = cache
        self.root = root

    def assume_role(self, role_arn, role_session_name):
        sts_client = boto3.client('sts')
        assumed_role_object = sts_client.assume_role(
            RoleArn=role_arn,
            RoleSessionName=role_session_name
        )

        credentials = assumed_role_object['Credentials']

        self._s3 = boto3.resource(
            's3',
            aws_access_key_id=credentials['AccessKeyId'],
            aws_secret_access_key=credentials['SecretAccessKey'],
            aws_session_token=credentials['SessionToken']
        )

        self._client = boto3.client(
            's3',
            aws_access_key_id=credentials['AccessKeyId'],
            aws_secret_access_key=credentials['SecretAccessKey'],
            aws_session_token=credentials['SessionToken']
        )

        if self._verbose:
            print("Successfully assumed role")

    def read_file(self, data, key):
        extension = FileHandler.extract_extension(key)
        if extension == "csv":
            return pd.read_csv(data)
        elif extension == "npy":
            with io.BytesIO(data) as f:
                return np.load(f)
        elif extension == "json":
            data = data.decode('utf-8')
            return json.loads(data)
        elif extension == "pkl":
            data = pickle.load(data)
            return data
        else:
            raise ValueError(f"Unknown filetype {key} with extension {extension}")

    def in_cache(self, key):
        if self.cache:
            return isfile(join(self.root, key))
        else:
            return False

    def load_from_cache(self, key):
        extension = FileHandler.extract_extension(key)
        if extension == "csv":
            return pd.read_csv(join(self.root, key), low_memory=False)
        elif extension == "npy":
            return np.load(join(self.root, key))
        elif extension == "json":
            with open(join(self.root, key), "r") as f:
                data = json.load(f)
            return data
        elif extension == "pkl":
            with open(join(self.root, key), "rb") as f:
                data = pickle.load(f)
            return data
        else:
            raise ValueError(f"Unknown filetype {key} with extension {extension}")

    def get_local_directory(self, key):
        if platform.system() == "Windows":
            directories = join(self.root, "\\".join(key.split("\\")[:-1]))
        else:
            directories = join(self.root, "/".join(key.split("/")[:-1]))
        return directories

    def get_key_directory(self, key):
        if platform.system() == "Windows":
            directories = "/".join(key.split("\\")[:-1])
        else:
            directories = "/".join(key.split("/")[:-1])
        return directories

    def create_subdirectories(self, key):
        directories = self.get_local_directory(key)
        Path(directories).mkdir(parents=True, exist_ok=True)


    def save_to_cache(self, data, key):
        self.create_subdirectories(key)
        extension = FileHandler.extract_extension(key)
        if extension == "csv":
            data.to_csv(join(self.root, key), index=False)
        elif extension == "npy":
            np.save(join(self.root, key), data)
        elif extension == "json":
            with open(join(self.root, key), 'w+') as outfile:
                json.dump(data, outfile)
        elif extension == "pkl":
            with open(join(self.root, key), "wb") as outfile:
                pickle.dump(data, outfile)
        else:
            raise ValueError(f"Unknown filetype {key} with extension {extension}")

    def list_objects(self, prefix=""):
        return [obj.key for obj in self._bucket.objects.filter(Prefix=prefix)][1:]

    def get_cache_location(self, key):
        return join(self.root, key)

    def get_cached_locations(self, keys):
        return [self.get_cache_location(key) for key in keys]

    def get_object(self, key):
        if self.cache and self.in_cache(key):
            data = self.load_from_cache(key)
        else:
            try:
                data = self._s3.Object(self._bucketname, key).get()['Body']
                data = self.read_file(data, key)
                if self.cache:
                    self.save_to_cache(data, key)
            except Exception as e:
                if self._verbose:
                    print(f"Unable to find key {key}", e)
                return None
        return data

    def download_object(self, key):
        self.create_subdirectories(key)
        with open(join(self.root, key), 'wb') as f:
            self._client.download_fileobj(self._bucketname, key, f)

    def get_objects(self, keys):
        assert isinstance(keys, list)
        objs = []
        for key in tqdm(keys):
            objs.append(self.get_object(key))
        return objs


    def create_s3_folder(self, directory):
        self._client.put_object(Bucket=self._bucketname, Key=(directory + '/'))

    def upload_object(self, data, key, overwrite=True, cache=False):
        try:
            if not self.in_cache(key) or overwrite:
                if self._verbose:
                    print(f"Saving key {key} to cache")
                self.save_to_cache(data, key)
            if self._verbose:
                print(f"Uploading {key} from {join(self.root, key)} to s3")
            self.create_s3_folder(self.get_key_directory(key))
            self._client.upload_file(join(self.root, key), self._bucketname, FileHandler.convert_to_unix(key))
            if self._verbose:
                print(f"Success")
            if self.cache is False:
                os.remove(join(self.root, key))
        except Exception as e:
            if self._verbose:
                print(e)

    def update_cache(self, data, key):
        self.save_to_cache(data, key)

    @staticmethod
    def extract_extension(key):
        extension = key.split(".")[-1]
        return extension

    @staticmethod
    def convert_to_unix(filedirectory):
        return "/".join(filedirectory.split("\\"))


# Create instance with default params
file_handler = FileHandler()

