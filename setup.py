from tqdm import tqdm
import requests, zipfile, os, glob, cv2

def download(url, filename):
    response = requests.get(url, stream=True)
    total_size_in_bytes= int(response.headers.get('content-length', 0))
    block_size = 1024 #1 Kibibyte
    progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)
    with open(filename, 'wb') as file:
        for data in response.iter_content(block_size):
            progress_bar.update(len(data))
            file.write(data)
    progress_bar.close()
    if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
        print('ERROR, something went wrong')

def unzip(filename, target_directory):
    with zipfile.ZipFile(file=filename) as zip_file:
        for file in tqdm(iterable=zip_file.namelist(), total=len(zip_file.namelist())):
            zip_file.extract(member=file, path=target_directory)

def download_and_unzip(name, url, zip_filename, target_directory):
    if (os.path.isdir(target_directory)):
        print(f"Found {name}...")
        return
    
    if (not os.path.isfile(zip_filename)) and (not os.path.isdir(target_directory)):
        print(f'Downloading {name} Dataset...')
        download(url, zip_filename)

    if (not os.path.isdir(target_directory)):
        print(f'Unzipping {name}...')
        unzip(zip_filename, target_directory)


def video_to_images(input_video, output_directory):
    os.makedirs(output_directory, exist_ok=True)
    vid = cv2.VideoCapture(input_video)
    success, image = vid.read()
    i = 0
    while success:
        cv2.imwrite(f'{output_directory}/{i}.png', image)
        success, image = vid.read()
        i += 1


# Download and unzip Vimeo90k
download_and_unzip('Vimeo', 'https://data.csail.mit.edu/tofu/dataset/vimeo_triplet.zip', 'vimeo.zip', 'Trainset/vimeo/')

# Download and unzip Davis
download_and_unzip('Davis', 'https://graphics.ethz.ch/Downloads/Data/Davis/DAVIS-data.zip', 'davis.zip', 'Trainset/davis/')

# Prepare Testset
print('Extracting images from test videos...')
test_videos = glob.glob('Testset/*.mp4')
for test_video in tqdm(iterable=test_videos, total=len(test_videos)):
    output_path = os.path.basename(test_video)
    output_path = os.path.splitext(output_path)[0]
    video_to_images(test_video, f'Testset/{output_path}/')