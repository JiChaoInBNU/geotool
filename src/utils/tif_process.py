# Date:2018.11.12
# Author: Kingdrone
# A tool implementation on gdal API
# functions:
# 1. clip raster with shapefile
# 2. clip raster with grid
# 3. get mask raster with shapefile

from PIL import Image, ImageDraw
import os
from osgeo import gdal, gdalnumeric
import numpy as np
import ogr
from src.utils.event_register import log_path
import glob
gdal.UseExceptions()

def write_log(v):
    with open(log_path, 'w') as f:
        f.write(str(v))

class GeoTiff(object):
    def __init__(self, tif_path):
        """
        A tool for Remote Sensing Image
        Args:
            tif_path: tif path
        Examples::
            >> tif = GeoTif('xx.tif')
            # if you want to clip tif with grid reserved geo reference
            >> tif.clip_tif_with_grid(512, 'out_dir')
            # if you want to clip tif with shape file
            >> tif.clip_tif_with_shapefile('shapefile.shp', 'save_path.tif')
            # if you want to mask tif with shape file
            >> tif.mask_tif_with_shapefile('shapefile.shp', 'save_path.tif')
        """
        self.dataset = gdal.Open(tif_path)
        self.bands_count = self.dataset.RasterCount
        # get each band
        self.bands = [self.dataset.GetRasterBand(i + 1) for i in range(self.bands_count)]
        self.col = self.dataset.RasterXSize
        self.row = self.dataset.RasterYSize
        self.geotransform = self.dataset.GetGeoTransform()
        self.src_path = tif_path

    def get_left_top(self):
        return self.geotransform[3], self.geotransform[0]

    def get_pixel_height_width(self):
        return abs(self.geotransform[5]), abs(self.geotransform[1])

    def __getitem__(self, *args):
        """

        Args:
            *args: range, an instance of tuple, ((start, stop, step), (start, stop, step))

        Returns:
            res: image block , array ,[bands......, height, weight]

        """
        if isinstance(args[0], tuple) and len(args[0]) == 2:
            # get params
            start_row, end_row = args[0][0].start, args[0][0].stop
            start_col, end_col = args[0][1].start, args[0][1].stop
            start_row = 0 if start_row is None else start_row
            start_col = 0 if start_col is None else start_col
            num_row = self.row if end_row is None else (end_row - start_row)
            num_col = self.col if end_col is None else (end_col - start_col)
            # dataset read image array
            res = self.dataset.ReadAsArray(start_col, start_row, num_col, num_row)
            return res
        else:
            raise NotImplementedError('the param should be [a: b, c: d] !')

    def clip_tif_with_grid(self, args):
        """
        clip image with grid
        Args:
            clip_size:
            out_dir:

        Returns:

        """
        clip_size, out_dir = args
        if not os.path.exists(out_dir):
            # check the dir
            os.makedirs(out_dir)
            print('create dir', out_dir)

        row_num = int(self.row / clip_size)
        col_num = int(self.col / clip_size)

        gtiffDriver = gdal.GetDriverByName('GTiff')
        if gtiffDriver is None:
            raise ValueError("Can't find GeoTiff Driver")

        count = 1
        for i in range(row_num):
            for j in range(col_num):
                clipped_image = np.array(self[i * clip_size: (i + 1) * clip_size, j * clip_size: (j + 1) * clip_size])
                clipped_image = clipped_image.astype(np.int8)

                try:
                    save_path = os.path.join(out_dir, '%d_%d.tif' % (i+1, j+1))
                    save_image_with_georef(clipped_image, gtiffDriver,
                                           self.dataset, j*clip_size, i*clip_size, save_path)
                    print('clip successfully！(%d/%d)' % (count, row_num * col_num))
                    count += 1

                    # write log
                    write_log(int(count / (row_num * col_num)*100))
                except Exception:
                    raise IOError('clip failed!%d' % count)

    def world2Pixel(self, x, y):
        """
        Uses a gdal geomatrix (gdal.GetGeoTransform()) to calculate
        the pixel location of a geospatial coordinate
        """
        ulY, ulX = self.get_left_top()
        distY, distX = self.get_pixel_height_width()

        pixel_x = abs(int((x - ulX) / distX))
        pixel_y = abs(int((ulY - y) / distY))
        pixel_y = self.row if pixel_y > self.row else pixel_y
        pixel_x = self.col if pixel_x > self.col else pixel_x
        return pixel_x, pixel_y

    def clip_tif_with_shapefile(self, args):
        """

        Args:
            shapefile_path:
            save_dir:

        Returns:

        """
        fieldname, shapefile_path, save_dir = args
        driver = ogr.GetDriverByName('ESRI Shapefile')
        dataSource = driver.Open(shapefile_path, 0)
        if dataSource is None:
            raise IOError('could not open!')
        gtiffDriver = gdal.GetDriverByName('GTiff')
        if gtiffDriver is None:
            raise ValueError("Can't find GeoTiff Driver")

        layer = dataSource.GetLayer(0)
        minX, maxX, minY, maxY = layer.GetExtent()

        poly_num = layer.GetFeatureCount()  # get poly count
        for i in range(poly_num):
            # copy extent for initialization
            p_minx, p_maxx, p_miny, p_maxy = maxX, 0, maxY, 0

            points = []  # store points
            pixels = []  # store pixels

            polygon = layer.GetFeature(i)
            geom = polygon.GetGeometryRef()
            feature_type = geom.GetGeometryName()
            #################################################
            if feature_type == 'POLYGON' or 'MULTIPOLYGON':
                # multi polygon operation
                for j in range(geom.GetGeometryCount()):
                    sub_polygon = geom.GetGeometryRef(j)
                    if feature_type == 'MULTIPOLYGON':
                        sub_polygon = sub_polygon.GetGeometryRef(0)
                    for p_i in range(sub_polygon.GetPointCount()):
                        px = sub_polygon.GetX(p_i)
                        py = sub_polygon.GetY(p_i)
                        points.append((px, py))
                        # find the polygon extent
                        p_minx = px if px < p_minx else p_minx
                        p_miny = py if py < p_miny else p_miny
                        p_maxx = px if px > p_maxx else p_maxx
                        p_maxy = py if py > p_maxy else p_maxy
                    p_ulx, p_uly = self.world2Pixel(p_minx, p_maxy)
                    p_lrx, p_lry = self.world2Pixel(p_maxx, p_miny)
                    for p in points:
                        origin_pixel_x, origin_pixel_y = self.world2Pixel(p[0], p[1])
                        # the pixel in new image
                        new_pixel_x, new_pixel_y = origin_pixel_x - p_ulx, origin_pixel_y - p_uly
                        pixels.append((new_pixel_x, new_pixel_y))
                #################################################
                    # draw mask
                    rasterPoly = Image.new("L", (p_lrx - p_ulx, p_lry - p_uly), 0)
                    rasterize = ImageDraw.Draw(rasterPoly)
                    # pixel x, y
                    rasterize.polygon(pixels, 1)

                    mask = np.array(rasterPoly)
                    clipped_image = self[p_uly: p_lry, p_ulx: p_lrx] * mask  # mask the cilp

                    # save the cilpped image with geo_reference
                    try:
                        save_name = polygon.GetFieldAsString(fieldname)
                        save_path = os.path.join(save_dir, save_name + '.tif')
                        save_image_with_georef(clipped_image, gtiffDriver, self.dataset, p_ulx, p_uly, save_path)
                        polygon.Destroy()  # delete feature
                        print('clip successfully！(%d/%d)' % (i + 1, poly_num))
                        # process log
                        write_log(str(int((i + 1)/poly_num*100)))

                    except Exception:
                        raise IOError('clip failed!%d' % (i + 1))
            else:
                pts = geom.GetGeometryRef(0)

                for j in range(pts.GetPointCount()):
                    px = pts.GetX(j)
                    py = pts.GetY(j)
                    points.append((px, py))
                    # find the polygon extent
                    p_minx = px if px < p_minx else p_minx
                    p_miny = py if py < p_miny else p_miny
                    p_maxx = px if px > p_maxx else p_maxx
                    p_maxy = py if py > p_maxy else p_maxy
                p_ulx, p_uly = self.world2Pixel(p_minx, p_maxy)
                p_lrx, p_lry = self.world2Pixel(p_maxx, p_miny)
                for p in points:
                    origin_pixel_x,  origin_pixel_y = self.world2Pixel(p[0], p[1])
                    # the pixel in new image
                    new_pixel_x, new_pixel_y = origin_pixel_x - p_ulx,  origin_pixel_y - p_uly
                    pixels.append((new_pixel_x, new_pixel_y))

                # draw mask
                rasterPoly = Image.new("L", (p_lrx - p_ulx, p_lry - p_uly), 0)
                rasterize = ImageDraw.Draw(rasterPoly)
                # pixel x, y
                rasterize.polygon(pixels, 1)

                mask = np.array(rasterPoly)
                clipped_image = self[p_uly: p_lry, p_ulx: p_lrx] * mask  # mask the cilp

                # save the cilpped image with geo_reference
                try:
                    save_name = polygon.GetFieldAsString(fieldname)
                    save_path = os.path.join(save_dir, str(i+1) + '.tif')
                    save_image_with_georef(clipped_image, gtiffDriver, self.dataset, p_ulx, p_uly, save_path)
                    polygon.Destroy()  # delete feature
                    print('clip successfully！(%d/%d)' % (i+1, poly_num))
                except Exception:
                    raise IOError('clip failed!%d' % (i+1))

        dataSource.Destroy()  # delete ds

    def mask_tif_with_shapefile(self, args):
        """

        Args:
            shapefile_path:
            save_path:
            label:

        Returns:

        """
        shapefile_path, save_path, label = args
        driver = ogr.GetDriverByName('ESRI Shapefile')
        dataSource = driver.Open(shapefile_path, 0)
        if dataSource is None:
            raise IOError('could not open!')
        gtiffDriver = gdal.GetDriverByName('GTiff')
        if gtiffDriver is None:
            raise ValueError("Can't find GeoTiff Driver")

        layer = dataSource.GetLayer(0)
        # # Convert the layer extent to image pixel coordinates
        minX, maxX, minY, maxY = layer.GetExtent()
        ulX, ulY = self.world2Pixel(minX, maxY)
        lrX, lrY = self.world2Pixel(maxX, minY)

        # initialize mask drawing
        rasterPoly = Image.new("L", (lrX - ulX, lrY - ulY), 0)
        rasterize = ImageDraw.Draw(rasterPoly)

        feature_num = layer.GetFeatureCount()  # get poly count
        for i in range(feature_num):
            points = []  # store points
            pixels = []  # store pixels
            feature = layer.GetFeature(i)
            geom = feature.GetGeometryRef()
            feature_type = geom.GetGeometryName()

            if feature_type == 'POLYGON' or 'MULTIPOLYGON':
                # multi polygon operation
                # 1. use label to mask the max polygon
                # 2. use -label to mask the other polygon
                for j in range(geom.GetGeometryCount()):
                    sub_polygon = geom.GetGeometryRef(j)
                    if feature_type == 'MULTIPOLYGON':
                        sub_polygon = sub_polygon.GetGeometryRef(0)
                    for p_i in range(sub_polygon.GetPointCount()):
                        px = sub_polygon.GetX(p_i)
                        py = sub_polygon.GetY(p_i)
                        points.append((px, py))

                    for p in points:
                        origin_pixel_x, origin_pixel_y = self.world2Pixel(p[0], p[1])
                        # the pixel in new image
                        new_pixel_x, new_pixel_y = origin_pixel_x - ulX, origin_pixel_y - ulY
                        pixels.append((new_pixel_x, new_pixel_y))

                    rasterize.polygon(pixels, label)
                    pixels = []
                    points = []
                    if feature_type != 'MULTIPOLYGON':
                        label = -abs(label)

                # restore the label value
                label = abs(label)
            else:
                for j in range(geom.GetPointCount()):
                    px = geom.GetX(j)
                    py = geom.GetY(j)
                    points.append((px, py))

                for p in points:
                    origin_pixel_x, origin_pixel_y = self.world2Pixel(p[0], p[1])
                    # the pixel in new image
                    new_pixel_x, new_pixel_y = origin_pixel_x - ulX, origin_pixel_y - ulY
                    pixels.append((new_pixel_x, new_pixel_y))

                feature.Destroy()  # delete feature

                if feature_type == 'LINESTRING':
                    rasterize.line(pixels, label)
                if feature_type == 'POINT':
                    # pixel x, y
                    rasterize.point(pixels, label)

            write_log(str(int((i+1) / feature_num*100)))
        mask = np.array(rasterPoly)
        mask = mask[np.newaxis, :]  # extend an axis to three
        # save the cilpped image with geo_reference
        try:
            save_image_with_georef(mask, gtiffDriver, self.dataset, ulX, ulY, save_path)

            print('mask successfully')
        except Exception:
            raise IOError('mask failed!')

        dataSource.Destroy()  # delete ds

def channel_first_to_last(image):
    """

    Args:
        image: 3-D numpy array of shape [channel, width, height]

    Returns:
        new_image: 3-D numpy array of shape [height, width, channel]
    """
    new_image = np.transpose(image, axes=[1, 2, 0])
    return new_image

def channel_last_to_first(image):
    """

    Args:
        image: 3-D numpy array of shape [channel, width, height]

    Returns:
        new_image: 3-D numpy array of shape [height, width, channel]
    """
    new_image = np.transpose(image, axes=[2, 0, 1])
    return new_image

def save_image_with_georef(image, driver, original_ds, offset_x=0, offset_y=0, save_path=None):
    """

    Args:
        save_path: str, image save path
        driver: gdal IO driver
        image: an instance of ndarray
        original_ds: a instance of data set
        offset_x: x location in data set
        offset_y: y location in data set

    Returns:

    """
    # get Geo Reference
    ds = gdalnumeric.OpenArray(image)
    gdalnumeric.CopyDatasetInfo(original_ds, ds, xoff=offset_x, yoff=offset_y)
    driver.CreateCopy(save_path, ds)
    # write by band
    clip = image.astype(np.int8)
    # write the dataset
    if len(image.shape)==3:
        for i in range(image.shape[0]):
            ds.GetRasterBand(i + 1).WriteArray(clip[i])
    else:
        ds.GetRasterBand(1).WriteArray(clip)
    del ds

def define_ref_predict(args):
    """
    define reference for raster referred to a geometric raster.
    Args:
        tif_dir: the dir to save referenced raster
        mask_dir:
        save_dir:

    Returns:

    """
    tif_dir, mask_dir, save_dir = args
    tif_list = glob.glob(os.path.join(tif_dir, '*.tif'))

    mask_list = glob.glob(os.path.join(mask_dir, '*.png'))
    mask_list += (glob.glob(os.path.join(mask_dir, '*.jpg')))
    mask_list += (glob.glob(os.path.join(mask_dir, '*.tif')))

    tif_list.sort()
    mask_list.sort()

    os.makedirs(save_dir, exist_ok=True)
    gtiffDriver = gdal.GetDriverByName('GTiff')
    if gtiffDriver is None:
        raise ValueError("Can't find GeoTiff Driver")
    for i in range(len(tif_list)):
        save_name = tif_list[i].split('\\')[-1]
        save_path = os.path.join(save_dir, save_name)
        tif = GeoTiff(tif_list[i])
        mask = np.array(Image.open(mask_list[i]))
        mask = channel_last_to_first(mask)
        save_image_with_georef(mask, gtiffDriver, tif.dataset, save_path=save_path)
        write_log(str(int((i+1)/len(tif_list) * 100)))


class GeoShaplefile(object):
    def __init__(self, file_path=""):
        self.file_path = file_path
        self.layer = ""
        self.minX, self.maxX, self.minY, self.maxY = (0, 0, 0, 0)
        self.feature_type = ""
        self.feature_num = 0
        self.fieldname_list = []
        self.open_shapefile()


    def open_shapefile(self):
        driver = ogr.GetDriverByName('ESRI Shapefile')
        dataSource = driver.Open(self.file_path, 0)
        if dataSource is None:
            raise IOError('could not open!')
        gtiffDriver = gdal.GetDriverByName('GTiff')
        if gtiffDriver is None:
            raise ValueError("Can't find GeoTiff Driver")

        self.layer = dataSource.GetLayer(0)
        self.minX, self.maxX, self.minY, self.maxY = self.layer.GetExtent()
        # layer.g
        self.feature_num = self.layer.GetFeatureCount()  # get poly count


        ldefn = self.layer.GetLayerDefn()
        for n in range(ldefn.GetFieldCount()):
            fdefn = ldefn.GetFieldDefn(n)
            self.fieldname_list.append(fdefn.name)


        if self.feature_num > 0:
            polygon = self.layer.GetFeature(0)
            geom = polygon.GetGeometryRef()
            # feature type
            self.feature_type = geom.GetGeometryName()

if __name__ == '__main__':
    shp = GeoShaplefile('C:/Users/mi/Desktop/data/wgshy1.shp')
    print(shp.fieldname_list)