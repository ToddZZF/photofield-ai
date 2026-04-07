from urllib.parse import urlparse
from urllib.error import HTTPError
import urllib.request
from tqdm import tqdm
import os

class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)

def ensure_model(path_or_url, dir):
    path = path_or_url
    if path_or_url.startswith("http"):
        u = urlparse(path_or_url)
        filename = u.path.split("/")[-1]
        path = os.path.join(dir, filename)
        if not os.path.exists(path):
            download(path_or_url, path)
        if filename.endswith(".onnx"):
            data_filename = f"{filename}.data"
            data_path = os.path.join(dir, data_filename)
            if not os.path.exists(data_path):
                try:
                    download(f"{path_or_url}.data", data_path)
                except HTTPError as e:
                    if e.code != 404:
                        raise
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model not found: {path}")
    return path

def download(url, output_path):
    with DownloadProgressBar(
        unit='B',
        unit_scale=True,
        miniters=1,
        desc=url.split('/')[-1]
    ) as t:
        urllib.request.urlretrieve(
            url,
            filename=output_path,
            reporthook=t.update_to
        )
