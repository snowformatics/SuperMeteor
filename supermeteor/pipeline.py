
import os
import cv2
from PIL import Image
import imagehash
import glob
import pandas as pd

path_in = "G:/Meine Ablage/sdr/01/"
#path_out = "G:/Meine Ablage/sdr/test_output/"
path_out = "G:/Meine Ablage/sdr/11062023/"
path_negatives = "E:/sdr/wilhelm/negatives/"
#path_positives = "G:/Meine Ablage/sdr/pos/"
path_positives = "G:/Meine Ablage/sdr/11062023/"


def run_segmentation(path_in, path_out, path_negatives, remove_raw_file=False):
    """Reads the raw images from SpectrumLab and start mask-c-rnn segmentation."""
    from pixellib.instance import custom_segmentation

    meteor_class = 3

    # Get all image files in directory
    image_lst = os.listdir(path_in)

    segment_image = custom_segmentation()
    segment_image.inferConfig(num_classes=3, class_names=["BG", "Artifical-Star", "Background", "Meteor"])
    segment_image.load_model("mask_rcnn_model.040-0.432872.h5")

    for image in image_lst:
        #print (image)
        n = 1
        try:
            output, segmask = segment_image.segmentImage(path_in + image, show_bboxes=True,
                                                         output_image_name=None,
                                                         extract_segmented_objects=False,
                                                         save_extracted_objects=False)

            # If objects contain meteor class, we store them
            if meteor_class in output.get('class_ids'):
                for coord in output.get('rois'):
                    # We store the rois, the original image and the mask
                    org_image = cv2.imread(path_in + image)
                    roi = org_image[coord[0] - 25:coord[2] + 25, coord[1] - 25:coord[3] + 25]
                    cv2.imwrite(path_out + '' + image.split('.jpg')[0] + '_' + str(n) + '_roi.jpg', roi)
                    cv2.imwrite(path_out + '' + image.split('.jpg')[0] + '_' + str(n) + '_org.jpg', org_image)
                    cv2.imwrite(path_out + '' + image.split('.jpg')[0] + '_' + str(n) + '_mask.jpg', segmask)
                    n += 1
            else:
                cv2.imwrite(path_negatives + image, segmask)
            if remove_raw_file:
                os.remove(path_in + image)
        except:
            print (image)


def remove_duplicated_rois(path_out):
    """Removes same rois from screenshots with image hashing algorithm."""
    # Extract only images which are meteor rois.
    l = glob.glob(path_out + "*roi.jpg")
    c1 = 0
    c2 = 1
    duplicate_lst = []
    for i in l:
        try:
            if i.endswith('_roi.jpg'):
                image1_path = l[c1]
                image2_path = l[c2]
                t1 = l[c1].split('.')[0]
                t2 = l[c2].split('.')[0]
                # If the files names are identical, we remove one of them
                if t1 == t2:
                    duplicate_lst.append(l[c1].split('\\')[-1])
                else:
                    # If the hash value is very low, most likely duplicated roi
                    hash = imagehash.dhash(Image.open(image1_path))
                    hash2 = imagehash.dhash(Image.open(image2_path))
                    #print (image1_path, image2_path, hash - hash2)

                    if hash - hash2 < 17:
                        duplicate_lst.append(l[c1].split('\\')[-1])
                c1 += 1
                c2 += 1
        except IndexError:
            pass
    return duplicate_lst


def create_results(out_file, path_out, duplicate_lst):
    """Create the final result file for the web app."""
    out_file = open(out_file, 'a')
    # We get all images which contain meteors
    image_lst = os.listdir(path_out)
    l = []
    for image in image_lst:
        if image.endswith('_roi.jpg'):
            l.append((image))
            #l.append(image.split('_roi.jpg')[0])
    # We remove images which are duplicates
    for d in duplicate_lst:
        if d in l:
            l.remove(d)

    # We extract all information from the final image list
    for f in l:
        try:
            img = cv2.imread(path_out + f)
            height = img.shape[0]
            width = img.shape[1]

            # Open org
            #shutil.copyfile(path_pos + f + '_org.jpg', path_capt + f + '_org.jpg')
            timestamp = f.split('_')[1]
            timestamp2 = '20' + timestamp[0:2] + '-' + timestamp[2:4] + '-' + timestamp[4:6] + ' ' + timestamp[6:12]
            out_file.write(
                f + '\t' + '20' + timestamp[0:2] + '\t' + timestamp[2:4] + '\t' + timestamp[4:6] + '\t' + timestamp[6:12]
                + '\t' + timestamp2 + '\t' + str(height) + '\t' + str(width) + '\n')
        except:
            pass


def get_top5_meteors(data):
    data['h'] = pd.to_numeric(data['h'], errors='coerce')
    data['w'] = pd.to_numeric(data['w'], errors='coerce')

    # Group the DataFrame by 'date'
    grouped_df = data.groupby('date')

    # Extract the two largest 'w' and 'h' values per date
    largest_objects_per_day = grouped_df.apply(lambda x: x.nlargest(5, ['w', 'h'])).reset_index(drop=True)
    return largest_objects_per_day


def load_data(DATA_URL):
    f = '%Y-%m-%d'
    f2 = '%H%M%S'
    data = pd.read_csv(DATA_URL, delimiter='\t', dtype={'time':str})
    data['date'] = data["timestemp"].str.slice(stop=10)
    data['date'] = pd.to_datetime(data['date'], format=f)
    data['time'] = pd.to_datetime(data['time'], format=f2)
    data['date'] = data['date'].astype(str)
    data['time'] = data['time'].astype(str)
    data['time'] = data['time'].str.slice(10)
    data['date/time'] = pd.to_datetime(data['date'].astype(str) +data['time'].astype(str))
    return data


def get_top5(path_positives, largest_objects_per_day):
    all_pos = os.listdir(path_positives)
    top5 = []
    id_lst = []
    for pos in all_pos:
        for index, row in largest_objects_per_day.iterrows():
            id = row['image_file'][0:25]
            if id not in id_lst:
                if pos.find(id) != -1:
                    if pos.find('org') != -1:
                        top5.append(pos)
                        id_lst.append(id)
    return top5


def upload_top5(top5, path_positives, image_out):
    import base64
    import time
    from imagekitio import ImageKit
    from imagekitio.models.UploadFileRequestOptions import UploadFileRequestOptions

    imagekit = ImageKit(
        private_key='private_p7g5dfbmyWdd8OxQj2PNqyrAKAk=',
        public_key='public_e1pl0k0cLHamnjuL8EhBYCrYaos=',
        url_endpoint='https://ik.imagekit.io/nb4gbrqqe'
    )
    for img_f in top5:
        print (img_f)
        with open(path_positives + img_f, 'rb') as f:
            # Read the contents of the file
            image_data = f.read()
            # Encode the image data to base64
            encoded_image = base64.b64encode(image_data)
            encoded_image

        upload = imagekit.upload(
            encoded_image,
            file_name=img_f,
            options=UploadFileRequestOptions(
                tags=["meteor", "org"]
            )
        )
    image_out = open(image_out, 'a')
    time.sleep(10)
    list_files = imagekit.list_files()
    d = list_files.response_metadata.raw
    for i in d:
        for the_key, the_value in i.items():
            if the_key == 'url':
                image_out.write(the_value + '\n')



run_segmentation(path_in, path_out, path_negatives, remove_raw_file=True)
duplicate_lst = remove_duplicated_rois(path_out)
out_file = 'out_all.csv'
image_out = 'image_out.csv'
create_results(out_file, path_out, duplicate_lst)

data = load_data(out_file)
largest_objects_per_day = get_top5_meteors(data)
top5 = get_top5(path_positives, largest_objects_per_day)
upload_top5(top5, path_positives, image_out)