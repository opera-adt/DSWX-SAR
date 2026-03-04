
from dataclasses import dataclass
from typing import Optional, Tuple
import math
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

    def update_geogrid(self, geotiff_path: str) -> None:
        """
        Update this geogrid to be the union of itself and the geogrid from geotiff_path.
        Works for both north-up rasters (spacing_y < 0) and south-up (spacing_y > 0).

        Conventions:
        - We union using true bounds (xmin/xmax/ymin/ymax) independent of sign.
        - We keep this object's spacing sign if already set; otherwise adopt new's.
        - start_x/start_y represent the "origin corner" implied by spacing signs:
            * spacing_x > 0  => start_x is xmin
            * spacing_x < 0  => start_x is xmax
            * spacing_y < 0  => start_y is ymax (north-up)
            * spacing_y > 0  => start_y is ymin (south-up)
        """
        new = DSWXGeogrid.from_geotiff(geotiff_path)

        # EPSG check
        if (not np.isnan(self.epsg)) and int(self.epsg) != int(new.epsg):
            raise ValueError(
                f"EPSG codes do not match: existing={self.epsg}, new={new.epsg}"
            )

        # Decide spacings (magnitude + sign)
        # If self spacing is unset, inherit from new; otherwise keep self.
        if np.isnan(self.spacing_x):
            self.spacing_x = new.spacing_x
        if np.isnan(self.spacing_y):
            self.spacing_y = new.spacing_y

        # Helper: bounds independent of sign
        def _bounds(g):
            xmin = min(g.start_x, g.end_x)
            xmax = max(g.start_x, g.end_x)
            ymin = min(g.start_y, g.end_y)
            ymax = max(g.start_y, g.end_y)
            return xmin, ymin, xmax, ymax

        # Union bounds (handle NaNs gracefully)
        xmin0, ymin0, xmax0, ymax0 = _bounds(self) if not np.isnan(self.start_x) else (np.nan, np.nan, np.nan, np.nan)
        xmin1, ymin1, xmax1, ymax1 = _bounds(new)

        xs = [v for v in (xmin0, xmin1) if not np.isnan(v)]
        ys = [v for v in (ymin0, ymin1) if not np.isnan(v)]
        xe = [v for v in (xmax0, xmax1) if not np.isnan(v)]
        ye = [v for v in (ymax0, ymax1) if not np.isnan(v)]

        xmin_u = min(xs)
        ymin_u = min(ys)
        xmax_u = max(xe)
        ymax_u = max(ye)

        # Reconstruct start/end consistent with spacing sign convention
        sx = self.spacing_x
        sy = self.spacing_y

        # X
        if sx >= 0:
            self.start_x = xmin_u
            self.end_x   = xmax_u
        else:
            self.start_x = xmax_u
            self.end_x   = xmin_u

        # Y
        if sy < 0:  # north-up
            self.start_y = ymax_u
            self.end_y   = ymin_u
        else:       # south-up or unusual
            self.start_y = ymin_u
            self.end_y   = ymax_u

        # width/length (ensure positive)
        if not np.isnan(sx) and sx != 0:
            self.width = int(round((self.end_x - self.start_x) / sx))
            self.width = abs(self.width)

        if not np.isnan(sy) and sy != 0:
            self.length = int(round((self.end_y - self.start_y) / sy))
            self.length = abs(self.length)

        # EPSG
        if np.isnan(self.epsg):
            self.epsg = new.epsg


    @staticmethod
    def _snap_bounds_to_spacing(
        xmin: float, ymin: float, xmax: float, ymax: float,
        spacing_x: float, spacing_y: float,
    ) -> Tuple[float, float, float, float]:
        """
        Snap bounds to pixel grid so width/length become integers.
        Uses spacing magnitude; preserves original spacing sign when writing geogrid.
        """
        sx = abs(spacing_x)
        sy = abs(spacing_y)
        if sx == 0 or sy == 0 or math.isnan(sx) or math.isnan(sy):
            raise ValueError("Invalid spacing for snapping bounds.")

        # snap outward (cover the requested bbox)
        xmin_s = math.floor(xmin / sx) * sx
        ymin_s = math.floor(ymin / sy) * sy
        xmax_s = math.ceil(xmax / sx) * sx
        ymax_s = math.ceil(ymax / sy) * sy
        return xmin_s, ymin_s, xmax_s, ymax_s

    def clip_to_bbox(
        self,
        bbox: Tuple[float, float, float, float],
        bbox_epsg: Optional[int] = None,
        snap: bool = True,
        snap_outward: bool = True,  # keep outward snap as default
    ) -> None:
        """
        Intersect current geogrid with bbox (must be in same epsg).
        Updates start/end/width/length to the intersection.
        """
        xmin_b, ymin_b, xmax_b, ymax_b = bbox

        epsg_ok = not (self.epsg is None or (isinstance(self.epsg, float) and np.isnan(self.epsg)))
        if bbox_epsg is not None and epsg_ok and int(self.epsg) != int(bbox_epsg):
            raise ValueError(f"EPSG mismatch: geogrid epsg={self.epsg}, bbox epsg={bbox_epsg}. "
                             "Reproject bbox first or force EPSG earlier.")

        xmin_g, ymin_g, xmax_g, ymax_g = self.bounds()

        ixmin = max(xmin_g, xmin_b)
        iymin = max(ymin_g, ymin_b)
        ixmax = min(xmax_g, xmax_b)
        iymax = min(ymax_g, ymax_b)

        if ixmin >= ixmax or iymin >= iymax:
            if ixmin >= ixmax:
                print('error x', ixmin, ixmax)
            if iymin >= iymax:
                print('error y', iymin, iymax)
            raise ValueError("DB bbox does not intersect the data extent (geogrid).")

        if snap:
            # snap outward so you don't accidentally clip requested area by <1 pixel
            ixmin, iymin, ixmax, iymax = self._snap_bounds_to_spacing(
                ixmin, iymin, ixmax, iymax, self.spacing_x, self.spacing_y
            )

        # Re-apply sign convention used in your geogrid:
        # spacing_x typically +, spacing_y may be negative.
        sx = self.spacing_x
        sy = self.spacing_y

        # start_x should be left edge
        self.start_x = ixmin
        self.end_x = ixmax

        # start_y should match your spacing_y sign convention:
        # If sy < 0, start_y is the TOP (max y). If sy > 0, start_y is BOTTOM (min y).
        if sy < 0:
            self.start_y = iymax
            self.end_y = iymin
        else:
            self.start_y = iymin
            self.end_y = iymax

        self.width  = int(round((self.end_x - self.start_x) / self.spacing_x)) if sx != 0 else self.width
        self.length = int(round((self.end_y - self.start_y) / self.spacing_y)) if sy != 0 else self.length

    def bounds(self) -> Tuple[float, float, float, float]:
        """
        Return geogrid bounds as (xmin, ymin, xmax, ymax) in the geogrid EPSG.
        Works regardless of spacing_y sign (north-up geotiffs usually have spacing_y < 0).
        """
        # Ensure initialized
        required = {
            "start_x": self.start_x,
            "start_y": self.start_y,
            "end_x": self.end_x,
            "end_y": self.end_y,
            "spacing_x": self.spacing_x,
            "spacing_y": self.spacing_y,
        }
        missing = [k for k, v in required.items() if v is None or (isinstance(v, float) and np.isnan(v))]
        if missing:
            raise RuntimeError(
                f"Geogrid not initialized; missing {missing}. "
                "Call update_geogrid()/from_geotiff() before clip_to_bbox()."
            )

        xmin = min(self.start_x, self.end_x)
        xmax = max(self.start_x, self.end_x)
        ymin = min(self.start_y, self.end_y)
        ymax = max(self.start_y, self.end_y)
        return xmin, ymin, xmax, ymax
