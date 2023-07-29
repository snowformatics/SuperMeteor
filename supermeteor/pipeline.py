from pixellib.instance import custom_segmentation
import os
import cv2
import pandas as pd
from datetime import datetime


path_in = "C:/Users/lueck/Nextcloud/meteor/"

today = datetime.now()

try:
    dir_name_pos = os.makedirs("C:/Users/lueck/meteor/pos/" + today.strftime('%Y%m%d'))
    dir_name_neg = os.makedirs("C:/Users/lueck/meteor/neg/"  + today.strftime('%Y%m%d'))

except FileExistsError:
    dir_name_pos = "C:/Users/lueck/meteor/pos/" + today.strftime('%Y%m%d')
    dir_name_neg = "C:/Users/lueck/meteor/neg/" + today.strftime('%Y%m%d')


path_out = dir_name_pos
path_negatives = dir_name_neg
path_positives = dir_name_pos

print (path_positives, path_negatives)


def run_segmentation(path_in, path_out, path_negatives, remove_raw_file=False):
    """Reads the raw images from SpectrumLab and start mask-c-rnn segmentation."""


    meteor_class = 3

    # Get all image files in directory
    image_lst = os.listdir(path_in)

    segment_image = custom_segmentation()
    segment_image.inferConfig(num_classes=3, class_names=["BG", "Artifical-Star", "Background", "Meteor"])
    segment_image.load_model("mask_rcnn_model.040-0.432872.h5")

    for image in image_lst:
        #print (path_in + image)
        n = 1
        try:
            output, segmask = segment_image.segmentImage(path_in + image, show_bboxes=True,
                                                         output_image_name=None,
                                                         extract_segmented_objects=False,
                                                         save_extracted_objects=False)

            #print (image, output.get('class_ids'))
            # If objects contain meteor class, we store them
            if meteor_class in output.get('class_ids'):
                #print ('pos', image)
                for coord in output.get('rois'):

                    # We store the rois, the original image and the mask
                    org_image = cv2.imread(path_in + image)

                    roi = org_image[coord[0] - 25:coord[2] + 25, coord[1] - 25:coord[3] + 25]


                    cv2.imwrite(path_out + '/' + '' + image.split('.jpg')[0] + '_' + str(n) + '_roi.jpg', roi)
                    cv2.imwrite(path_out + '/' + '' + image.split('.jpg')[0] + '_' + str(n) + '_org.jpg', org_image)
                    cv2.imwrite(path_out + '/' + '' + image.split('.jpg')[0] + '_' + str(n) + '_mask.jpg', segmask)
                    n += 1
                    #print (path_out + '' + image.split('.jpg')[0] + '_' + str(n) + '_roi.jpg')
            else:
                #print ('neg', image)
                cv2.imwrite(path_negatives + '/' + image, segmask)
            if remove_raw_file:
               os.remove(path_in + image)
        except:
            print (image)
            print(coord)
            cv2.imshow('', org_image)
            cv2.waitKey(0)
            os.remove(path_in + image)


def run_image_upload(directory_path):
    def get_image_size(image_path):
        image = cv2.imread(image_path)
        if image is not None:
            height, width = image.shape[:2]
            return width * height
        else:
            return None

    l = os.listdir(directory_path)
    extension = "_roi.jpg"
    filtered_files = [file for file in l if file.lower().endswith(extension.lower())]

    data = []
    for file_name in filtered_files:
        image_path = os.path.join(directory_path, file_name)
        image_size = get_image_size(image_path)
        if image_size is not None:
            data.append({"image_name": file_name, "image_size": image_size})

    # Create DataFrame from the list of dictionaries
    df = pd.DataFrame(data)

    # Sort the DataFrame based on "image_size" column in descending order
    sorted_df = df.sort_values(by="image_size", ascending=False)

    # Function to calculate timestamp from image name
    def extract_timestamp_from_image(image_name):
        return datetime.strptime(image_name.split("_")[1], '%y%m%d%H%M%S')

    # Function to check if the time difference is smaller than 100 seconds
    def is_time_difference_smaller_than_100_seconds(image1, image2):
        timestamp1 = extract_timestamp_from_image(image1)
        timestamp2 = extract_timestamp_from_image(image2)
        time_difference = abs((timestamp2 - timestamp1).total_seconds())
        return time_difference < 100

    top_5_images = []

    # Iterate through the sorted data to find the top 5 images with the largest size
    for index, row in sorted_df.iterrows():
        image_name = row["image_name"]
        if not top_5_images:
            top_5_images.append(image_name)
        else:
            can_add_image = True
            for top_image in top_5_images:
                if is_time_difference_smaller_than_100_seconds(top_image, image_name):
                    can_add_image = False
                    break
            if can_add_image:
                top_5_images.append(image_name)
            if len(top_5_images) == 5:
                break


    # Display the final top 5 images
    return top_5_images

def create_results(out_file, path_out):
    """Create the final result file for the web app."""
    out_file = open(out_file, 'a')
    # We get all images which contain meteors
    image_lst = os.listdir(path_out)
    l = []
    for image in image_lst:
        if image.endswith('_roi.jpg'):
            l.append((image))

    # We extract all information from the final image list
    for f in l:
        try:
            img = cv2.imread(path_out + f)
            height = img.shape[0]
            width = img.shape[1]

            timestamp = f.split('_')[1]
            timestamp2 = '20' + timestamp[0:2] + '-' + timestamp[2:4] + '-' + timestamp[4:6] + ' ' + timestamp[6:12]
            out_file.write(
                f + '\t' + '20' + timestamp[0:2] + '\t' + timestamp[2:4] + '\t' + timestamp[4:6] + '\t' + timestamp[6:12]
                + '\t' + timestamp2 + '\t' + str(height) + '\t' + str(width) + '\n')
        except:
            pass

def upload_top5(top5, path_positives, image_out):
    import base64
    import time
    from imagekitio import ImageKit
    from imagekitio.models.UploadFileRequestOptions import UploadFileRequestOptions

    image_out = open(image_out, 'a')

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
            #encoded_image

        upload = imagekit.upload(
            encoded_image,
            file_name=img_f,
            options=UploadFileRequestOptions(
                tags=["meteor", "org"]
            )
        )
        #print(upload.name)
        image_out.write('https://ik.imagekit.io/nb4gbrqqe/' + upload.name + '\n')

#run_segmentation(path_in, path_out, path_negatives, remove_raw_file=True)
directory_path = 'C:/Users/lueck/meteor/pos/20230728/'# dir_name_pos
create_results('out_all2.csv', directory_path)
top_5_images = run_image_upload(directory_path)
upload_top5(top_5_images, directory_path, 'image_out.csv')