import requests, zipfile, io

# Download and unzip Vimeo90k
r = requests.get("data.csail.mit.edu/tofu/dataset/vimeo_triplet.zip", stream=True)
z = zipfile.ZipFile(io.BytesIO(r.content))
z.extractall("")