
from dataclasses import dataclass

import numpy as np
from osgeo import osr, gdal


@dataclass
class DSWXGeogrid:
    """
    A dataclass representing the geographical grid configuration
    for an RTC (Radar Terrain Correction) run.

    Attributes:
    -----------
    start_x : float
        The starting x-coordinate of the grid.
    start_y : float
        The starting y-coordinate of the grid.
    end_x : float
        The ending x-coordinate of the grid.
    end_y : float
        The ending y-coordinate of the grid.
    spacing_x : float
        The spacing between points in the x-direction.
    spacing_y : float
        The spacing between points in the y-direction.
    length : int
        The number of points in the y-direction.
    width : int
        The number of points in the x-direction.
    epsg : int
        The EPSG code representing the coordinate reference system of the grid.
    """
    start_x: float = np.nan
    start_y: float = np.nan
    end_x: float = np.nan
    end_y: float = np.nan
    spacing_x: float = np.nan
    spacing_y: float = np.nan
    length: int = np.nan
    width: int = np.nan
    epsg: int = np.nan

    def get_geogrid_from_geotiff(self,
                                 geotiff_path):
        """
        Extract geographical grid parameters from a GeoTIFF file
        and update the dataclass attributes.

        Parameters
        ----------
        geotiff_path : str
            The file path to the GeoTIFF file from which the grid
            parameters are to be extracted.
        """
        tif_gdal = gdal.Open(geotiff_path)
        geotransform = tif_gdal.GetGeoTransform()
        self.start_x = geotransform[0]
        self.spacing_x = geotransform[1]

        self.start_y = geotransform[3]
        self.spacing_y = geotransform[5]

        self.length = tif_gdal.RasterYSize
        self.width = tif_gdal.RasterXSize

        self.end_x = self.start_x + self.width * self.spacing_x
        self.end_y = self.start_y + self.length * self.spacing_y

        projection = tif_gdal.GetProjection()
        proj = osr.SpatialReference(wkt=projection)
        output_epsg = proj.GetAttrValue('AUTHORITY', 1)
        self.epsg = int(output_epsg)
        tif_gdal = None
        del tif_gdal

    @classmethod
    def from_geotiff(cls, geotiff_path):
        """
        Extract geographical grid parameters from a GeoTIFF file
        and update the dataclass attributes.
        Parameters
        ----------
        geotiff_path : str
            The file path to the GeoTIFF file from which the grid
            parameters are to be extracted.
        """
        tif_gdal = gdal.Open(geotiff_path)
        geotransform = tif_gdal.GetGeoTransform()
        start_x, spacing_x, _, start_y, _, spacing_y = geotransform
        length = tif_gdal.RasterYSize
        width = tif_gdal.RasterXSize
        end_x = start_x + width * spacing_x
        end_y = start_y + length * spacing_y
        projection = tif_gdal.GetProjection()
        proj = osr.SpatialReference(wkt=projection)
        output_epsg = proj.GetAttrValue('AUTHORITY', 1)
        epsg = int(output_epsg)
        tif_gdal = None
        del tif_gdal
        return cls(start_x, start_y, end_x, end_y, spacing_x, spacing_y,
                   length, width, epsg)

    def update_geogrid(self, geotiff_path):
        """
        Update the existing geographical grid parameters based on a new
        GeoTIFF file, extending the grid to encompass both the existing
        and new grid areas.
        """
        new_geogrid = DSWXGeogrid.from_geotiff(geotiff_path)

        if self.epsg != new_geogrid.epsg and not np.isnan(self.epsg):
            raise ValueError("EPSG codes of the existing and "
                             "new geogrids do not match.")
        self.start_x = min(filter(lambda x: not np.isnan(x),
                                  [self.start_x, new_geogrid.start_x]))
        self.end_x = max(filter(lambda x: not np.isnan(x),
                                [self.end_x, new_geogrid.end_x]))

        if self.spacing_y > 0 or np.isnan(self.spacing_y):
            self.end_y = max(filter(lambda x: not np.isnan(x),
                                    [self.end_y, new_geogrid.end_y]))
            self.start_y = min(filter(lambda x: not np.isnan(x),
                                      [self.start_y, new_geogrid.start_y]))
        else:
            self.start_y = max(filter(lambda x: not np.isnan(x),
                                      [self.start_y, new_geogrid.start_y]))
            self.end_y = min(filter(lambda x: not np.isnan(x),
                                    [self.end_y, new_geogrid.end_y]))

        self.spacing_x = new_geogrid.spacing_x \
            if not np.isnan(new_geogrid.spacing_x) else self.spacing_x
        self.spacing_y = new_geogrid.spacing_y \
            if not np.isnan(new_geogrid.spacing_y) else self.spacing_y

        if not np.isnan(self.start_x) and not np.isnan(self.end_x) and \
           not np.isnan(self.spacing_x):
            self.width = int((self.end_x - self.start_x) / self.spacing_x)

        if not np.isnan(self.start_y) and not np.isnan(self.end_y) and \
           not np.isnan(self.spacing_y):
            self.length = int((self.end_y - self.start_y) / self.spacing_y)

        self.epsg = new_geogrid.epsg \
            if not np.isnan(new_geogrid.epsg) else self.epsg

