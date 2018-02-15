import cv2
import os
import numpy as np
from matplotlib import pyplot as plt
from aerotools.gps_tools import get_meters_per_pixel, lat_lon_to_pixel, pixel_to_lat_lon, metres_between_gps
from aerotools.geotiff_tools import clip_raster_with_client_orchard, tiff_read, tiff_write, tiff_info,orchard_polygon
import gdal
from numpy import median
from math import sqrt
from ImageClass import ImageObject
from os.path import splitext, dirname, join, basename
from osgeo import ogr




def image_read(image_path):
    """
    Function reads image from specified path

    :param image_path: path to image to be read
    :return image_array: np.array containing image pixels
    :return image_dataset: gdal.dataset for image
    """

    # Read in images using gdal
    image_dataset = gdal.Open(image_path, gdal.GA_ReadOnly)
    image_array = np.array(image_dataset.GetRasterBand(1).ReadAsArray())

    if image_dataset.RasterCount == 1:
        image_array = map(lambda x: (x + 1) * 128, image_array)

    image_array = np.uint8(image_array)

    print image_array.shape

    return image_array, image_dataset



def resize_img(resize_factor, image_path):
    """
    Resizes image in-place according to resize factor
    :param resize_factor: (float) resize factor between 0 and infinity
    :param image_path: (str) path to image to be resize
    :return: state: (bool) True if operation was successful, False otherwise
    """
    if resize_factor == 1:
        print "Resize not necessary. Skipping."
        return
    print "Applying resize factor of %.1f ..." % resize_factor

    base, ext = splitext(basename(image_path))
#    base = base.split('_')[0]
    base_name = "%s_scale_%.2f" % (base, resize_factor)
    resize_path = join(dirname(dirname(image_path)), base_name, base_name + ext)
    if not os.path.isdir(dirname(resize_path)):
        os.makedirs(dirname(resize_path))
    image = gdal.Open(image_path)
    resize_size = int(round(image.RasterXSize * resize_factor))
    resize_cmd = """gdalwarp -overwrite -s_srs epsg:4326 -t_srs epsg:4326 -co compress=LZW -ts {WIDTH} 0 """ \
                 """-srcnodata -32676 -dstnodata -32676 -r cubic -of GTiff "{IMAGE}" "{OUTPUTIMAGE}" """ \
        .format(WIDTH=resize_size, IMAGE=image_path, OUTPUTIMAGE=resize_path)
    os.system(resize_cmd)
    return resize_path



def determine_matched_features(detector_type, im_correct, im_correct_ds, im_distorted, im_distorted_ds):
    """
    Function detects -> extracts -> matches features from 2 images

    :param detector_type: string that represents the detector to be used
    :param im_correct: correct image np.array
    :param im_correct_ds: gdal.dataset for correct image
    :param im_distorted: distorted image np.array
    :param im_distorted_ds: gdal.dataset for distorted image
    :return match_coord_img_correct: np.array with matching coordinates from correct image corresponding to distorted
    :return match_coord_img_distorted: np.array with matching coordinates from distored image corresponding to correct
    """
    # Selected a feature detector
    if detector_type == 'ORB':
        detector = cv2.ORB_create()
    if detector_type == 'BRISK':
        detector = cv2.BRISK_create()
    if detector_type == 'SURF':
        detector = cv2.xfeatures2d.SURF_create()
        detector.setExtended(True)
        detector.setHessianThreshold(800)
        detector.setNOctaves(2)
    else:
        detector = cv2.ORB_create()

    # Identify keypoints and features using detector
    key_points1, descriptors1 = detector.detectAndCompute(im_correct, None)
    key_points2, descriptors2 = detector.detectAndCompute(im_distorted, None)

    # create BFMatcher - BruteForceMatcher object
    if detector_type == 'SURF':
        print 'SURF detector'
        """
        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)  # or pass empty dictionary

        matcher = cv2.FlannBasedMatcher(index_params, search_params)
        """
        matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
        matches = matcher.match(descriptors1, descriptors2)

    else:
        print detector + ' detector'
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        # Match descriptors.
        matches = matcher.match(descriptors1, descriptors2)

    # Sort them in the order of their distance.
    #matches = sorted(matches, key=lambda x: x.distance)

    # find coordinates of matching features in both images
    # queryIdx - The index or row of the kp1 interest point matrix that matches
    # trainIdx - The index or row of the kp2 interest point matrix that matches
    match_coord_img_correct = [key_points1[mat.queryIdx].pt for mat in matches]
    match_coord_img_distorted = [key_points2[mat.trainIdx].pt for mat in matches]

    match_gps_img_correct = []
    match_gps_img_distorted = []
    for i in range(len(match_coord_img_correct)):
        (p1,q1) = match_coord_img_correct[i]
        (p2,q2) = match_coord_img_distorted[i]
        x1,y1 = pixel_to_lat_lon(im_correct_ds.GetGeoTransform(),p1,q1)
        x2, y2 = pixel_to_lat_lon(im_distorted_ds.GetGeoTransform(), p2, q2)
        match_gps_img_correct.append((x1,y1))
        match_gps_img_distorted.append((x2,y2))

    match_gps_img_correct,match_gps_img_distorted,_ = filter_gps_distance(match_gps_img_correct,match_gps_img_distorted,
                                                                          matches)

    return match_gps_img_correct, match_gps_img_distorted



def filter_gps_distance(correct_gps,distorted_gps,match_features):
    """
    Function filters matched points by distance of separation of GPS coordinates of points
    Selected points are within +- the standard deviation from the mean

    :param correct_gps: np.array with gps coordinates of matched featues from correct image
    :param distorted_gps: np.array with gps coordinates of matched featues from distorted image
    :param match_features: list of matched features
    :return sel_gps_correct: selected matched gps coordinates from correct image
    :return sel_gps_distorted: selected matched gps coordinates from distorted image
    :return sel_match: selected matched features
    """

    dist = []
    for i in range(len(correct_gps)):
        (x1,y1) = correct_gps[i]
        (x2,y2) = distorted_gps[i]
        dist.append(metres_between_gps(y1,x1,y2,x2))

    mean = np.mean(dist)

    print 'mean dist = ', mean
    print 'min dist = ', np.min(dist)
    print 'max dist = ', np.max(dist)

    sel_gps_correct  = []
    sel_gps_distorted = []
    sel_match = []
    for j in range(len(correct_gps)):
        #if dist[j] <mean +std and dist[j] > mean -std:
        if dist[j] <= 5:
            sel_gps_correct.append(correct_gps[j])
            sel_gps_distorted.append(distorted_gps[j])
            sel_match.append(match_features[j])
    print '#selected matches: %d out of %d'% (len(sel_gps_distorted), len(distorted_gps))
    return sel_gps_correct,sel_gps_distorted,sel_match



def draw_matches_image(detector_type, im_correct, im_correct_ds, im_distorted, im_distorted_ds):
    """
    Function draws image showing 2 input images and matching features

    :param detector_type: string that represents the detector to be used
    :param im_correct: correct image np.array
    :param im_correct_ds: gdal.dataset for correct image
    :param im_distorted: distorted image np.array
    :param im_distorted_ds: gdal.dataset for distorted image
    :return:
    """
    # Selected a feature detector
    if detector_type == 'ORB':
        detector = cv2.ORB_create()
    if detector_type == 'BRISK':
        detector = cv2.BRISK_create()
    if detector_type == 'SURF':
        detector = cv2.xfeatures2d.SURF_create()
        detector.setExtended(True)
        detector.setHessianThreshold(800)
        detector.setNOctaves(4)
    else:
        detector = cv2.ORB_create()

    # Identify keypoints and features using detector
    key_points1, descriptors1 = detector.detectAndCompute(im_correct, None)
    key_points2, descriptors2 = detector.detectAndCompute(im_distorted, None)

    # create BFMatcher - BruteForceMatcher object
    if detector_type == 'SURF':
        print 'SURF detector'
        """
        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)  # or pass empty dictionary

        matcher = cv2.FlannBasedMatcher(index_params, search_params)
        """
        matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
        matches = matcher.match(descriptors1, descriptors2)

    else:
        print detector + ' detector'
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        # Match descriptors.
        matches = matcher.match(descriptors1, descriptors2)


    match_coord_img_correct = [key_points1[mat.queryIdx].pt for mat in matches]
    match_coord_img_distorted = [key_points2[mat.trainIdx].pt for mat in matches]

    match_gps_img_correct = []
    match_gps_img_distorted = []
    for i in range(len(match_coord_img_correct)):
        (p1, q1) = match_coord_img_correct[i]
        (p2, q2) = match_coord_img_distorted[i]
        x1, y1 = pixel_to_lat_lon(im_correct_ds.GetGeoTransform(), p1, q1)
        x2, y2 = pixel_to_lat_lon(im_distorted_ds.GetGeoTransform(), p2, q2)
        match_gps_img_correct.append((x1, y1))
        match_gps_img_distorted.append((x2, y2))

    _,__,selected_matches = filter_gps_distance(match_gps_img_correct,match_gps_img_distorted,matches)

    ## write out images with circles indicating the identified features
    i = 0
    j = 0
    k = 0
    for match in selected_matches:
        (x1, y1) = key_points1[match.queryIdx].pt
        radius1 = key_points1[match.queryIdx].size
        cv2.circle(im_correct, (int(x1), int(y1)), int(radius1), (255 - i, 255 - j, 0 + k, 255), 6)
        (x2, y2) = key_points2[match.trainIdx].pt
        radius2 = key_points2[match.trainIdx].size
        cv2.circle(im_distorted, (int(x2), int(y2)), int(radius2), (255 - i, 255 - j, 0 + k, 255), 6)
        i +=51
        j+=15
        k+=85
        if i== 255:
            i = 0
        if j == 255:
            j = 0
        if k ==255:
            k = 0

    print 'writing images showing matched features'
    path_im1 = '/Users/brendontankwa/Desktop/Aerobotics/IRTestCases/res/res3/res1/_corr_img_surf_k1_dl5_n10.tif'
    path_im2 = '/Users/brendontankwa/Desktop/Aerobotics/IRTestCases/res/res3/res1/_dis_img_surf_k1_dl5_n10.tif'
    tiff_write(path_im1, im_correct)
    tiff_write(path_im2, im_distorted)
    im1 = cv2.imread(path_im1)
    im2 = cv2.imread(path_im2)

    fig = plt.figure()
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.set_title('Correct image')
    ax1.imshow(im1)
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.set_title('Distorted image')
    ax2.imshow(im2)
    plt.show()

    os.remove('corr_img.tif')
    os.remove('dis_img.tif')




def register_image(correct_img_gps,distorted_img_gps,correct_path,distorted_path,number_match_used,out_path):
    """
    Function registers the image and creates an output file of registered image from os

    :param correct_img_gps: np.array with gps coordinates of matched featues from correct image
    :param distorted_img_gps: np.array with gps coordinates of matched featues from distorted image
    :param correct_path: path to correct image
    :param distorted_path: path to misaligned image
    :param number_match_used: number of matches to use in alignment
    :param out_path:
    :return: true if image registered. false otherwise
    """
    image_object_correct = ImageObject(correct_path)  # aligned
    image_object_correct.define_child_image(distorted_path)  # needs to align to other image

    correct_gps = correct_img_gps[:number_match_used]
    distorted_gps = distorted_img_gps[:number_match_used]

    child_string = ''
    for dis_gps, corr_gps in zip(distorted_gps, correct_gps):
        child_string += '-gcp '
        parent_lon, parent_lat = corr_gps[0], corr_gps[1]
        child_x, child_y = lat_lon_to_pixel(image_object_correct.child_image.geo_transform, parent_lon, parent_lat)
        child_string += str(child_x) + ' ' + str(child_y) + ' ' + str(parent_lon) + ' ' + str(parent_lat) + ' '

    print 'gcps = ', child_string


    temp_output_path = os.path.splitext(correct_path)[0] + '_aligned_temp' + os.path.splitext(correct_path)[1]
    #output_path = temp_output_path.replace('_temp', '')

    #temp_output_path = '/Users/brendontankwa/Desktop/Aerobotics/IRTestCases/temp.tif'

    os.system("gdal_translate -of GTiff -a_srs EPSG:4236 {0} {1} {2}".format(child_string, distorted_image_path,
                                                                             temp_output_path))
    os.system("gdalwarp -s_srs EPSG:4326 -t_srs EPSG:4326 {0} {1}".format(temp_output_path, out_path))
    os.remove(temp_output_path)

    if os.path.isfile(out_path):
        print 'Image registered'
        return True
    else:
        return False

    #img4 = ImageObject(out_path)
    #print os.path.isfile(out_path)
    #print 'This is the path', out_path
    #plt.imshow(img4), plt.show()



def register_display_image(path_correct_image,path_distorted_image,path_output_image,feature_detector,number_match
                       ,show_mat_image,register_img):
    """
    This function performs image registration of a distorted image on to a correct image

    :param path_correct_image: path to correct image
    :param path_distorted_image: path to misaligned image
    :param path_output_image: path to misaligned image
    :param feature_detector: string that represents the detector to be used - ORB , BRISK , KAZE
    :param show_mat_image: display images showing identified and matched features on both images if == True
    :param number_match: number of gcps to use in registration
    :param show_mat_image: boolean to determine if image showing matched features should be shown
    :param register_img: boolean to determine if image should be registered
    :return:
    """

    img_correct, img_correct_dataset = image_read(path_correct_image)
    img_distorted, img_distorted_dataset = image_read(path_distorted_image)

    hcorr, wcorr = img_correct.shape
    hdist, w2dist = img_distorted.shape

    resizing_factor = float(hdist) / float(hcorr)
    print 'resizing_factor = ', resizing_factor


    resize_im = True
    resized = False
    if resize_im:
        resized_image_path = resize_img(resizing_factor, path_correct_image)
        path_correct_image = resized_image_path
        resized = True

    img_correct, img_distorted_dataset = image_read(path_correct_image)

    detector_used = feature_detector

    grayscale_px_coord, visible_px_coord = determine_matched_features(detector_used, img_correct, img_correct_dataset,
                                                                      img_distorted, img_distorted_dataset)

    number_of_matched_coord = number_match
    if number_of_matched_coord > len(grayscale_px_coord):
        number_of_matched_coord = len(grayscale_px_coord)

    if number_of_matched_coord < 3:
        print 'Number of matches detected < 3. Too small to register image'
        if resized:
            os.remove(resized_image_path)
            os.rmdir(dirname(resized_image_path))
        return False

    if register_img == True:
        register_image(grayscale_px_coord, visible_px_coord, path_correct_image, path_distorted_image,
                   number_of_matched_coord,path_output_image)


    ### optional - draw image showing the correct and distorted image and the detected matched features
    if show_mat_image == True:
        draw_matches_image(detector_used, img_correct, img_correct_dataset,img_distorted, img_distorted_dataset)

    if resized:
        os.remove(resized_image_path)
        os.rmdir(dirname(resized_image_path))



if __name__ == "__main__":
    #correct_image_path = '/Users/brendontankwa/Desktop/Aerobotics/IRTestCases/TestImage.tiff'
    #correct_image_path = '/Users/brendontankwa/Desktop/Aerobotics/grayndvi.tif'
    #correct_image_path = '/Users/brendontankwa/Desktop/Aerobotics/IRTestCases/client_1731_orchard_9368/grayndvi.tif'
    #correct_image_path = '/Users/brendontankwa/Desktop/Aerobotics/orthomosaic_visible.tif'
    correct_image_path = '/Users/brendontankwa/Desktop/Aerobotics/VARI.tif'
    # distorted_image_path = '/Users/brendontankwa/Desktop/Aerobotics/IRTestCases/rot_img.tiff'
    #distorted_image_path = '/Users/brendontankwa/Desktop/Aerobotics/orthomosaic_visible.tif'
    #distorted_image_path = '/Users/brendontankwa/Desktop/Aerobotics/IRTestCases/client_1731_orchard_9368/orthomosaic_visible.tif'
    #distorted_image_path = '/Users/brendontankwa/Desktop/Aerobotics/IRTestCases/client_1731_orchard_9368/VARI.tif'
    #distorted_image_path = '/Users/brendontankwa/Desktop/Aerobotics/VARI.tif'
    distorted_image_path = '/Users/brendontankwa/Desktop/Aerobotics/grayndvi.tif'
    temp_output_path = '/Users/brendontankwa/Desktop/Aerobotics/IRTestCases/temp.tif'
    output_path = '/Users/brendontankwa/Desktop/Aerobotics/IRTestCases/res/res_vari/SSURF_nall_dl5_res.tif'
    #output_path  = '/Users/brendontankwa/Desktop/Aerobotics/IRTestCases/res/res_cl1731_9368/T2_ORB_4_sc1_fsmgps_res.tif'



    detector_used = 'SURF'
    num_match_features_used = 10000
    show_match_image = False
    register_img = True

    register_display_image(correct_image_path, distorted_image_path, output_path, detector_used, num_match_features_used,
                       show_match_image, register_img)