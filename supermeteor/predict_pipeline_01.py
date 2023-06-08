from pixellib.instance import custom_segmentation
import os
import cv2




def run():
    #path_in = "E:/sdr/wilhelm/14/"
    path_in = "G:/Meine Ablage/sdr/01/"
    #path_out = "C:/Users/stefanie/PycharmProjects/SuperMeteor/supermeteor/output/"
    path_out = "G:/Meine Ablage/sdr/01_output/"
    #path_out = "G:/Meine Ablage/sdr/14_output/"
    print (path_in, path_out)
    meteor_class = 3
    out_stats = open('out_stats.txt', 'w')

    # Get all files in directory
    image_lst = os.listdir(path_in)
    print (image_lst)


    segment_image = custom_segmentation()
    segment_image.inferConfig(num_classes=3, class_names=["BG", "Artifical-Star", "Background", "Meteor"])
    #segment_image.load_model("ok-mask_rcnn_model.068-0.405656.h5")
    segment_image.load_model("mask_rcnn_model.040-0.432872.h5")

    last_polygon = None
    for image in image_lst:
        print (image)
        n = 1
        #timestamp = image.split('_')[1]
        #timestamp = '20' + timestamp[0:2] + '-' + timestamp[2:4] + '-' + timestamp[4:6] + ' ' + timestamp[6:12]
        try:
            output, segmask = segment_image.segmentImage(path_in + image, show_bboxes=True,
                                                         #output_image_name=path_out + image + '_output.jpg',
                                                         output_image_name=None,
                                                         extract_segmented_objects=False,
                                                         save_extracted_objects=False)

            # If objects contain meteor class, we store them
            if meteor_class in output.get('class_ids'):
                #cv2.imwrite(path_out + 'pos2/' + image + '_output.jpg', segmask)
                for coord in output.get('rois'):
                    #n = len(output.get('rois'))
                    org_image = cv2.imread(path_in + image)
                    roi = org_image[coord[0] - 25:coord[2] + 25, coord[1] - 25:coord[3] + 25]
                    cv2.imwrite(path_out + '' + image + '_' + str(n) + '_roi.jpg', roi)
                    cv2.imwrite(path_out + '' + image + '_' + str(n) + '_org.jpg', org_image)
                    cv2.imwrite(path_out + '' + image + '_' + str(n) + '_mask.jpg', segmask)
                    n += 1
                    #if last_polygon:
                        #print (last_polygon)
                        #time_diff = int(last_polygon[1][10:])-int(timestamp[11:18])
                        #if abs(time_diff) < 300:
                        #org_image = cv2.imread(path_in + image)
                        #roi = org_image[coord[0] - 25:coord[2] + 25, coord[1] - 25:coord[3] + 25]
                       # cv2.imwrite(path_out + 'rois/' + image + '_roi.jpg', roi)
                        #     previos_image = cv2.imread(path_in + last_polygon[0])
                        #
                        #     roi2 = previos_image[last_polygon[3][0] - 25:last_polygon[3][2] + 25, last_polygon[3][1] - 25:last_polygon[3][3] + 25]
                        #    # print(image, roi.shape, roi2.shape, last_polygon)
                        #     if abs(roi.shape[0] - roi2.shape[0]) < 10 and abs(roi.shape[1] - roi2.shape[1]) < 10 and n == 1 and last_polygon[4] == 1:
                        #         cv2.imwrite(path_out + 'same6/' + image + '_output.jpg', last_polygon[2])
                        #
                        #     else:
                        #         cv2.imwrite(path_out + 'pos6/' + image + '_output.jpg', segmask)
                        #         #cv2.imwrite(path_out + 'pos6/' + image + '_org.jpg', org_image)
                        # else:
                        #     cv2.imwrite(path_out + 'pos6/' + image + '_output.jpg', segmask)
                        #     #cv2.imwrite(path_out + 'pos6/' + image + '_org.jpg', org_image)


                        #last_polygon = [image, timestamp, segmask, coord, n]


                    #else:

                        #last_polygon = [image, timestamp, segmask, coord, n]



                #out_stats.write(image + '\t' + timestamp + '\t' + str(coord) + '\n')
                #cv2.imwrite(path_out + 'rois/' + image + 'roi.jpg', roi)
            else:
                cv2.imwrite('E:/sdr/wilhelm/negatives/' + image + '_output.jpg', segmask)
            os.remove(path_in + image)
        except:
            print (image)


        #os.remove(path_in + '_compr_' + image)
        #print(segmask, output)

run()