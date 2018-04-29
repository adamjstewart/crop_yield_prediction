"""Input/output tools for reading and writing GeoTIFF files."""

from osgeo import gdal

import numpy as np
import os


def read_geotiff(datafile, verbose=0):
    """Read GeoTIFF file of satellite imagery or classification labels.

    Args:
        datafile (str): the input GeoTIFF file
        verbose (int): the verbosity level

    Returns:
        dataset (osgeo.gdal.Dataset): the raster dataset
        array (numpy.ndarray): the raster data in array form.
            For satellite imagery, return shape is (rows * cols, bands).
            For classification labels, return shape is (rows * cols,)
        geo_transform (tuple): affine transform between coordinate systems
        projection (str): the coordinate system to project onto
        ctable (osgeo.gdal.ColorTable): the color table
        rows (int): the number of rows in the original dataset
        cols (int): the number of columns in the original dataset
    """
    if verbose > 0:
        print('Reading', datafile)

    # Read the GeoTIFF file
    dataset = gdal.Open(datafile)

    # Gather metadata for later usage
    geo_transform = dataset.GetGeoTransform()
    projection = dataset.GetProjectionRef()
    bands = dataset.RasterCount
    cols = dataset.RasterXSize
    rows = dataset.RasterYSize

    # Gather color table
    band = dataset.GetRasterBand(1)
    ctable = band.GetColorTable()

    # Gather data in array form
    array = dataset.ReadAsArray()

    # Reshape data array
    # For satellite imagery: (bands, rows, cols) -> (rows * cols, bands)
    # For classification labels: (rows, cols) -> (rows * cols,)
    if bands == 1:
        # Classification labels
        # Flatten first two dimensions:
        # (rows, cols) -> (rows * cols,)
        array = array.flatten()
    else:
        # Satellite imagery
        # Move bands axis to the end:
        # (bands, rows, cols) -> (rows, cols, bands)
        array = np.moveaxis(array, 0, -1)
        # Flatten first two dimensions:
        # (rows, cols, bands) -> (rows * cols, bands)
        array = array.reshape((rows * cols, bands))

    return dataset, array, geo_transform, projection, ctable, rows, cols


def write_geotiff(array, geo_transform, projection, ctable,
                  rows, cols, filename, suffixes=[], verbose=0):
    """Write GeoTIFF file of classification labels.

    Args:
        array (numpy.ndarray): the raster data in array form
        geo_transform (tuple): affine transform between coordinate systems
        projection (str): the coordinate system to project onto
        ctable (osgeo.gdal.ColorTable): the color table
        rows (int): the number of rows in the original dataset
        cols (int): the number of columns in the original dataset
        filename (str): the filename to save to
        suffixes (list): a list of suffixes to append to filename (optional)
        verbose (int): the verbosity level
    """
    # Save results in results directory
    filename = os.path.join('results', os.path.basename(filename))

    # Add suffixes to filename if necessary
    if suffixes:
        root, ext = os.path.splitext(filename)
        root += '_' + '_'.join(suffixes)
        filename = root + ext

    if verbose > 0:
        print('Writing', filename)

    # Delete any existing files
    if os.path.exists(filename):
        os.remove(filename)

    # Reshape data array: (rows * cols,) -> (rows, cols)
    array = array.reshape((rows, cols))

    # Create a new GeoTIFF file
    driver = gdal.GetDriverByName('GTiff')
    dataset = driver.Create(filename, cols, rows, 1, gdal.GDT_Byte)

    # Set the affine coordinate transform
    dataset.SetGeoTransform(geo_transform)

    # Set the projection
    dataset.SetProjection(projection)

    # Write data array to the dataset
    band = dataset.GetRasterBand(1)
    band.WriteArray(array)

    # Set the color table
    band.SetColorTable(ctable)
