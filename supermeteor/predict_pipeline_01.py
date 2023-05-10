import argparse
from pixellib.instance import custom_segmentation
import os
import cv2
import statistics
from PIL import Image

def run(args):

    path_in = args.path_in
    path_out = args.path_out
    meteor_class = 3
    out_stats = open('out_stats.txt', 'w')

    # Get all files in directory
    image_lst = os.listdir(path_in)

    # Predict all images
    all_rois = []
    for image in image_lst:
        #print (image)
        timestamp = image.split('_')[1]
        timestamp = '20' + timestamp[0:2] + '-' + timestamp[2:4] + '-' + timestamp[4:6] + ' ' + timestamp[6:12]
        picture = Image.open(path_in + image)
        picture.save(path_in + '_compr_' + image , optimize=True, quality=50)
        #org_image = cv2.imread(path_in + image)

        segment_image = custom_segmentation()
        segment_image.inferConfig(num_classes=3, class_names=["BG", "Artifical-Star", "Background", "Meteor"])
        segment_image.load_model("ok-mask_rcnn_model.068-0.405656.h5")

        output, segmask = segment_image.segmentImage(path_in + '_compr_' + image, show_bboxes=True,
                                                     #output_image_name=path_out + image + '_output.jpg',
                                                     output_image_name=None,
                                                     extract_segmented_objects=False,
                                                     save_extracted_objects=False)

        # If objects contain meteor class, we store them

        if meteor_class in output.get('class_ids'):
            cv2.imwrite(path_out + '3/' + image + '_output.jpg', segmask)
        else:
            cv2.imwrite(path_out + 'other/' + image + '_output.jpg', segmask)
            #print (output.get('rois'), output.get('class_ids'), output.get('scores'))
            #for coord in output.get('rois'):

                #roi = org_image[coord[0]-25:coord[2]+25, coord[1]-25:coord[3]+25]
                #out_stats.write(image + '\t' + timestamp + '\t' + str(coord) + '\n')
                #cv2.imwrite(path_out + 'rois/' + image + 'roi.jpg', roi)
                #all_rois.append(path_out + 'rois/' + image + 'roi.jpg')

                #if len(all_rois) > 1:
                    #statistics.compare_image_similarity(all_rois)
                #cv2.imshow('', roi)
                #cv2.waitKey(0)


            #cv2.imwrite(path_out + image + '_output.jpg', segmask)
            #cv2.imshow('', segmask)
            #cv2.waitKey(0)

        os.remove(path_in + '_compr_' + image)
        #print(segmask, output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Greeting Application')

    parser.add_argument('--path_in', type=str, help='Path which contains the meteor echo images.')
    parser.add_argument('--path_out', type=str, help='Path for the output.')

    run(parser.parse_args())