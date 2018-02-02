import gdal
import os
from aerotools.gps_tools import get_meters_per_pixel, lat_lon_to_pixel, pixel_to_lat_lon
import numpy as np
import cv2


class ImageError(Exception):
    def __init__(self, error_message=''):
        """
        Error template for all errors in this module
        Args:
            error_message: Error message to append to generic message.
        """
        Exception.__init__(self, "Error: ImageClass - " + error_message)


class ImageObject:
    def __init__(self, image_path):
        if os.path.isfile(image_path):
            self.dataset = gdal.Open(image_path)
        else:
            raise ImageError(error_message='Image path passed to class not a file')
        self.geo_transform = self.dataset.GetGeoTransform()
        self.image_path = image_path
        self.meters_per_pixel = get_meters_per_pixel(self.dataset)
        self.child_image = None
        self.min_val = None
        self.max_val = None
        self.factor = None

    def extract_box_pixel(self, start_pixel, end_pixel, buffer_factor=0):
        """
        Extract a rectangular box from an image saved to file using the top left pixel address (start pixel) and
        bottom right pixel address (end pixel) coordinates.
        :param start_pixel: Pixel coordinate of top left. np.array([x, y])
        :param end_pixel: Pixel coordinate of bottom right. np.array([x, y])
        :return: Img array
        """
        if buffer_factor != 0:
            start_xy, end_xy = self.handle_buffer(start_pixel.tolist(), end_pixel.tolist(), buffer_factor)
            start_pixel, end_pixel = np.array(start_xy), np.array(end_xy)
        self.test_coordinates(start_pixel)
        self.test_coordinates(end_pixel)
        box_size = end_pixel - start_pixel
        bands = []
        for band in range(self.dataset.RasterCount):
            band += 1
            band = self.dataset.GetRasterBand(band)
            bands.append(band.ReadAsArray(start_pixel[0], start_pixel[1], box_size[0], box_size[1]))
        array = np.dstack(tuple(bands))
        if self.dataset.RasterCount == 1:
            array = array[:, :, 0]
        return array

    def extract_box_gps(self, start_gps, end_gps, buffer_factor=0):
        """
        Extract a rectangular box from an image saved to file using the top left gps address (start gps) and
        bottom right (end gps) coordinates.
        :param start_gps: gps coordinate of top left. np.array([lon, lat])
        :param end_gps: gps coordinate of bottom right. np.array([lon, lat])
        :return: Img array
        """
        end_pixel_x, end_pixel_y = lat_lon_to_pixel(self.geo_transform, end_gps[0], end_gps[1])
        start_pixel_x, start_pixel_y = lat_lon_to_pixel(self.geo_transform, start_gps[0], start_gps[1])
        return self.extract_box_pixel(np.array([start_pixel_x, start_pixel_y]), np.array([end_pixel_x, end_pixel_y]),
                                      buffer_factor=buffer_factor)

    def handle_buffer(self, start_xy, end_xy, buffer_factor):
        if buffer_factor != 0:
            start_list, end_list = self.calc_buffered_points([start_xy[0], start_xy[1]],
                                                             [end_xy[0], end_xy[1]], buffer_factor)
            start_pixel_x, start_pixel_y = start_list[0], start_list[1]
            end_pixel_x, end_pixel_y = end_list[0], end_list[1]
            return [start_pixel_x, start_pixel_y], [end_pixel_x, end_pixel_y]
        else:
            return start_xy, end_xy

    @staticmethod
    def calc_buffered_points(start_xy, end_xy, buffer_factor):
        window_x_size = end_xy[0] - start_xy[0]
        window_y_size = end_xy[1] - start_xy[1]
        start_xy[0] -= int(buffer_factor * window_x_size)
        start_xy[1] -= int(buffer_factor * window_y_size)
        end_xy[0] += int(buffer_factor * window_x_size)
        end_xy[1] += int(buffer_factor * window_y_size)
        return start_xy, end_xy

    def extract_bounding_box_pixel(self, pixels_points, buffer_factor=0):
        """
        Extract a bounding image around a contour of pixel points.
        :param pixels_points: np array of contour pixels. np.array([[x1,y1], [x2,y2]...])
        :return: Img array
        """
        start_pixel_x, start_pixel_y = np.amin(pixels_points[:, 0]), np.amin(pixels_points[:, 1])
        end_pixel_x, end_pixel_y = np.amax(pixels_points[:, 0]), np.amax(pixels_points[:, 1])
        start_pixel = np.array([start_pixel_x, start_pixel_y])
        end_pixel = np.array([end_pixel_x, end_pixel_y])
        return self.extract_box_pixel(start_pixel, end_pixel, buffer_factor=buffer_factor)

    def extract_bounding_box_gps(self, gps_points, buffer_factor=0):
        """
        Extract a bounding image around a contour of gps points.
        :param gps_points: np array of contour gps. np.array([[lon1,lat1], [lon2,lat2]...])
        :return: Img array
        """
        start_gps = np.array([np.amin(gps_points[:, 0]), np.amax(gps_points[:, 1])])
        end_gps = np.array([np.amax(gps_points[:, 0]), np.amin(gps_points[:, 1])])
        return self.extract_box_gps(start_gps, end_gps, buffer_factor=buffer_factor)

    def define_child_image(self, path):
        """
        Associate an image with the parent image. This would be done to be able to extract
        equivalent images across multiple bands (ndvi, dem) or images
        :param path: Path to child image
        :return: None
        """
        self.child_image = ImageObject(path)
        self.child_image.factor = self.dataset.RasterXSize

    def extract_child_image_pixel(self, start_pixel_parent, end_pixel_parent, buffer_factor=0):
        """
        Extract the equivalent image to the one in the parent image defined by the passed pixel limits.
        :param start_pixel_parent: top left pixel coordinate of the desired image in the parent image. np.array([x, y])
        :param end_pixel_parent: bottom right pixel coordinate of the desired image in the parent image. 
        np.array([x, y])
        :return: Img array
        """
        if self.child_image is None:
            raise ImageError(error_message='extract_child_image_pixel. child Image is not defined.')
        start_gps, end_gps = self.convert_pixel_to_gps(start_pixel_parent, end_pixel_parent)
        image = self.child_image.extract_box_gps(start_gps, end_gps, buffer_factor=buffer_factor)
        return image

    def extract_child_image_gps(self, start_gps, end_gps, buffer_factor=0):
        """
        Extract the equivalent image to the one define in the parent image by the passed gps limits.
        :param start_gps: top left gps of the image in the parent image. np.array([lon, lat])
        :param end_gps: bottom right of the image in the parent image. np.array([lon, lat])
        :return: Img array
        """
        if self.child_image is None:
            raise ImageError(error_message='extract_child_image_gps. child Image is not defined.')
        print 'size of the extracted image', self.child_image.extract_box_gps(start_gps, end_gps).shape
        return self.child_image.extract_box_gps(start_gps, end_gps, buffer_factor=buffer_factor)

    def extract_resize_child_gps(self, start_gps, end_gps, buffer_factor=0):
        """
        Extract the equivalent image to the one define in the parent image by the passed gps limits.
        Resize image to be of the same size as the equivalent parent image.
        :param start_gps: start_gps: top left gps of the image in the parent image. np.array([lon, lat])
        :param end_gps: end_gps: bottom right of the image in the parent image. np.array([lon, lat])
        :return: Img Array
        """
        if self.child_image is None:
            raise ImageError(error_message='extract_resize_child_gps. child Image is not defined.')
        image_parent = self.extract_box_gps(start_gps, end_gps, buffer_factor=buffer_factor)
        image_child = self.extract_child_image_gps(start_gps, end_gps, buffer_factor=buffer_factor)
        return cv2.resize(image_child, image_parent.shape)

    def extract_resize_child_pixel(self, start_pixel_parent, end_pixel_parent, buffer_factor=0):
        """
        Extract the equivalent image to the one in the parent image defined by the passed pixel limits.
        Resize image to be of the equivalent size of the parent image that would be extracted.
        :param start_pixel_parent: top left pixel coordinate of the desired image in the parent image. np.array([x, y])
        :param end_pixel_parent: bottom right pixel coordinate of the desired image in the parent image. 
        np.array([x, y])
        :return: Img Array
        """
        if self.child_image is None:
            raise ImageError(error_message='extract_resize_child_pixel. child Image is not defined.')
        image_child = self.extract_child_image_pixel(start_pixel_parent, end_pixel_parent, buffer_factor=buffer_factor)
        image_parent = self.extract_box_pixel(start_pixel_parent, end_pixel_parent, buffer_factor=buffer_factor)
        return cv2.resize(image_child, (image_parent.shape[1], image_parent.shape[0]))

    def convert_pixel_to_gps(self, start_pixel_parent, end_pixel_parent):
        """
        Given two pixels the function converts these pixels into the gps coordinate system of the parent image.
        :param start_pixel_parent: first pixel address to be converted
        :param end_pixel_parent: second address to be converted
        :return: start_gps, end_gps
        """
        start_lon, start_lat = pixel_to_lat_lon(self.geo_transform, start_pixel_parent[0], start_pixel_parent[1])
        end_lon, end_lat = pixel_to_lat_lon(self.geo_transform, end_pixel_parent[0], end_pixel_parent[1])
        return np.array([start_lon, start_lat]), np.array([end_lon, end_lat])

    def test_coordinates(self, coordinates):
        """
        Test that extent of polygon coordinates are with in the image bounds
        Args:
            coordinates: polygon coordinates (numpy array)

        Returns: Coordinates with maximum check and replaced if necessary

        """
        if len(coordinates.shape) < 2:
            coordinates = np.array([coordinates])
        end_x, end_y = self.dataset.RasterXSize, self.dataset.RasterYSize
        if np.amax(coordinates[:, 0]) > self.dataset.RasterXSize:
            raise ImageError(error_message="requested X dimension in excess of image X dimension. Requested:{0}. "
                                           "Image:{1} ".format(str(np.amax(coordinates[:, 0])), str(end_x)))

        if np.amax(coordinates[:, 1]) > self.dataset.RasterYSize:
            raise ImageError(error_message="requested Y dimension in excess of image Y dimension. Requested:{0}. "
                                           "Image:{1} ".format(str(np.amax(coordinates[:, 1])), str(end_y)))

    def map_to_unit_space(self, array):
        """
        Maps any given sub array of an image to an array whose values lie between -1 and 1.
        Assigns attribute of array minimum and array maximum.
        Is useful for scipy and skimage implementations of float images
        :param array: sub array of class image
        :return: sub array with values mappedplacement_image to between -1 and 1
        """

        self.min_val = float(np.amin(array)) - 0.0005*float(np.amax(array))
        self.max_val = float(np.amax(array)) + 0.0005*float(np.amax(array))
        factor_1 = (self.max_val - self.min_val)/2.0
        factor_2 = (self.max_val + self.min_val)/2.0
        return (array.astype(np.float32) - factor_2)/factor_1

    def map_to_real_space(self, unitary_array):
        """
        Generate inverse mapping for an array values from unit space to real space 
        :param unitary_array: Array produced by map_to_unit_space where are values are between -1 and 1 
        :return: real space values of array
        """
        if self.min_val is None or self.max_val is None:
            raise ImageError(error_message="map_to_real_space. Either min or max not defined. Function can only be "
                                           "employed if map_to_unit_space has previously been employed for array "
                                           "passed. Min value: {0}. "
                                           "Max value:{1} ".format(str(self.min_val), str(self.max_val)))
        factor_1 = (self.max_val - self.min_val)/2.0
        factor_2 = (self.max_val + self.min_val)/2.0
        return unitary_array*factor_1 + factor_2

    def write_array_out_to_file(self, array, start_pixel_point, output_path, write_type, dataset=None,
                                geo_transform=None):
        """
        Writes an array to file as a geo tiff with the correct projection. Necessary to specify what the top left
        pixel address is of this sub image in the master image to generate correct top left gps. Take care in the 
        case of writing out resized child images
        :param array: array to write out
        :param start_pixel_point: top left pixel coordinate of sub image in master image coordinates. np.array([x,y])
        :return: None
        """
        def set_no_data_value(dataset, band, k):
            """
            private function set no data values in a band written to file if present
            :param band: band to write to file
            :return: checked and set band
            """
            if not(dataset.GetRasterBand(k+1).GetNoDataValue() is None):
                band.SetNoDataValue(dataset.GetRasterBand(k+1).GetNoDataValue())

        array[array < -100] = np.amin(array[array > -100])
        # Check that valid pixel address is passed
        if start_pixel_point.shape[0] < 2:
            raise ImageError(error_message="write_out_array_to_file. start_pixel_coordinate not of length 2."
                                           " start_pixel_point: {0}. type(start_pixel_point):{1} "
                             .format(str(start_pixel_point), str(type(start_pixel_point))))
        # Check that valid path is passed
        if not(os.path.isdir(os.path.dirname(output_path)) and os.path.splitext(output_path)[1] == '.tif'):
            raise ImageError(error_message="write_out_array_to_file. Output path does not end in tif file"
                                           " or directory does not exist. Path: {0} "
                             .format(str(output_path)))
        # check for overrides of dataset or geotransform.
        if dataset is None:
            dataset = self.dataset
        if geo_transform is None:
            geo_transform = self.geo_transform

        # Extract shape of array to write out
        cols, rows = array.shape[0], array.shape[1]
        num_bands = dataset.RasterCount

        lon, lat = pixel_to_lat_lon(self.geo_transform, start_pixel_point[0], start_pixel_point[1])
        # Extract projections and create drivers
        proj = dataset.GetProjection()
        outdriver = gdal.GetDriverByName("GTiff")
        outdata = outdriver.Create(str(output_path), rows, cols, num_bands, write_type)
        # Find starting lon and lat for geo transform
        geo_transform = list(geo_transform)
        geo_transform[0] = lon
        geo_transform[3] = lat
        geo_transform = tuple(geo_transform)
        # Set transform and write out data
        outdata.SetGeoTransform(geo_transform)
        outdata.SetProjection(proj)
        for k in range(num_bands):
            if num_bands == 1:
                if len(array.shape) > 2:
                    array = array[:, :, 0]
                outdata.GetRasterBand(k+1).WriteArray(array)
                set_no_data_value(dataset, outdata.GetRasterBand(k+1), k)
            else:
                outdata.GetRasterBand(k + 1).WriteArray(array[:, :, k])
                set_no_data_value(dataset, outdata.GetRasterBand(k+1), k)
        outdata.FlushCache()

    def resize_and_write_out_image(self, resize_factor, output_path, write_type):
        """
        Writes a resized version of the parent image of the class to file at the folder address specified by
        output_path. The write type must be specified. (write_type = gdal.GDT_Byte or gdal.GDT_Float32)
        :param resize_factor: factor by which to resize
        :param output_path: path tp write to
        :param write_type: gdal.GDT_Byte or gdal.GDT_Float32
        :return: 
        """

        # Extract projections and create drivers
        num_bands = self.dataset.RasterCount
        rows, cols = (int(float(self.dataset.RasterXSize) * resize_factor),
                      int(float(self.dataset.RasterYSize) * resize_factor))
        proj = self.dataset.GetProjection()
        outdriver = gdal.GetDriverByName("GTiff")
        outdata = outdriver.Create(str(output_path), rows, cols, num_bands, write_type)
        # Find starting lon and lat for geo transform
        geo_transform = list(self.geo_transform)
        geo_transform[1] *= 1.0 / resize_factor
        geo_transform[5] *= 1.0 / resize_factor
        geo_transform = tuple(geo_transform)
        # Set transform and write out data
        outdata.SetGeoTransform(geo_transform)
        outdata.SetProjection(proj)
        for k in range(num_bands):
            if num_bands == 1:
                band = self.dataset.GetRasterBand(k + 1)
                array = band.ReadAsArray(0, 0, self.dataset.RasterXSize, self.dataset.RasterYSize)
                array = cv2.resize(array, (rows, cols))
                outdata.GetRasterBand(k + 1).WriteArray(array)
                del array
            else:
                band = self.dataset.GetRasterBand(k + 1)
                array = band.ReadAsArray(0, 0, self.dataset.RasterXSize, self.dataset.RasterYSize)
                array = cv2.resize(array, (rows, cols))
                outdata.GetRasterBand(k + 1).WriteArray(array)
                del array
        outdata.FlushCache()


if __name__ == "__main__":
    from read_in_contour import get_contours
    import matplotlib.pyplot as plt
    contour_path = '/home/willem/Documents/Trees_to_do/download/train_data/Height_Map_Training/2526/output/' \
                   '7794_2526_returned_contours.shp'
    path = '/home/willem/Documents/Trees_to_do/download/train_data/Height_Map_Training/2526/' \
           '1414-7794-2526-2017-11-17-height_map_full.tif'
    image_object = ImageObject(path)
    # The below code should extract a full image from file.
    contour_list = get_contours(contour_path)
    for contour in [contour_list[0]]:
        contour = np.array(contour)
        img = image_object.extract_bounding_box_gps(contour, buffer_factor = 0.5)
        img2 = image_object.extract_bounding_box_gps(contour)
        plt.figure(1)
        plt.imshow(img, cmap='gray')
        plt.figure(2)
        plt.imshow(img2, cmap='gray')
        plt.show()

