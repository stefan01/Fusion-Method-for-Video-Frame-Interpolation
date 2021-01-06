from tqdm import tqdm
import numpy as np
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


def video_to_images(input_video, output_directory, crop=False, resize=False, newSize=(1280,720)):
    os.makedirs(output_directory, exist_ok=True)
    vid = cv2.VideoCapture(input_video)
    success, image = vid.read()
    print(success)

    # Random Crop Location
    max_x = image.shape[1] - 256
    max_y = image.shape[0] - 256
    crop_x = np.random.randint(0, max_x)
    crop_y = np.random.randint(0, max_y)

    i = 0
    while success:
        # Crop Image
        if crop:
            image = image[crop_y : crop_y + 256, crop_x : crop_x + 256]

        # Resize Image
        image = cv2.resize(image, newSize)

        # Write Image
        cv2.imwrite(f'{output_directory}/{str(i).zfill(3)}.png', image)
        success, image = vid.read()
        i += 1

def videos_to_images(input_files, output_directory, crop=False, resize=False, newSize=(1280,720)):
    print(f'Extracting images from videos...')
    for test_video in tqdm(iterable=input_files, total=len(input_files)):
        print(test_video)
        output_path = os.path.basename(test_video)
        output_path = os.path.splitext(output_path)[0]
        print(output_path)
        if(not os.path.isdir(f'{output_directory}/{output_path}/')):
            video_to_images(test_video, f'{output_directory}/{output_path}/', crop, resize, newSize)

def images_to_video(input_images, output_file, framerate=30):
    print(f'Combining images to video')
    imgs = []
    size = (1920, 1080)
    for image_file in input_images:
        img = cv2.imread(image_file)
        height, width, layers = img.shape
        size = (width, height)
        imgs.append(img)

    out = cv2.VideoWriter(output_file, cv2.VideoWriter.fourcc(*'MPEG'), framerate, size)
    for img in tqdm(iterable=imgs, total=len(imgs)):
        out.write(img)
    out.release()

#def images_to_triplets(input_files, output_directory):
#    if os.path.isdir(output_directory):
#        return
#    root, dirs, files = os.walk(input_files).next()
#    print(dirs)

"""
# Create Trainset dir
try:
    os.mkdir('./Trainset')
except OSError:
    print ("Trainset already exists")

# Download and unzip Vimeo90k
download_and_unzip('Vimeo', 'https://data.csail.mit.edu/tofu/dataset/vimeo_triplet.zip', 'Trainset/vimeo.zip', 'Trainset/vimeo/')

# Download and unzip Davis
download_and_unzip('Davis', 'https://graphics.ethz.ch/Downloads/Data/Davis/DAVIS-data.zip', 'Trainset/davis.zip', 'Trainset/davis/')

# Download and unzip NFS (need for speed)
try:
    os.mkdir('./Testset/nfs')
except OSError:
    print ("NFS Folder already exists")
for (url, name) in [
    ('https://cmu.box.com/shared/static/gl5o6qmq7i8da8w4px55ykvffjufw2m6.zip', 'airboard_1'),
    ('https://cmu.box.com/shared/static/kbx27fz51udgk4k08qh6l0yjmn9h3743.zip', 'airplane_landing'),
    ('https://cmu.box.com/shared/static/tuizb68pqafhooo4753nrk4b2iwwrmj8.zip', 'airtable_3'),
    ('https://cmu.box.com/shared/static/k9rkgfqp4ww7hickuo37z68d5l032fdx.zip', 'basketball_1'),
    ('https://cmu.box.com/shared/static/dsnxt9fqriivqilcqgul2v6b0k2g36xk.zip', 'water_ski_2'),
    ('https://cmu.box.com/shared/static/9m2bvwtrii9bwwqn4lr1cv09jtvfd4xi.zip', 'yoyo')
]:
    download_and_unzip(f'NFS {name}', url, f'Testset/nfs/{name}.zip', f'Testset/nfs/{name}/')
"""
# Prepare Testset
print(glob.glob('Testset/*.mp4'))
videos_to_images(glob.glob('Testset/*.mp4'), 'Testset', resize=True)

#images_to_video(glob.glob('Testset/Clip1/*.png'), 'Testset/c1.avi')