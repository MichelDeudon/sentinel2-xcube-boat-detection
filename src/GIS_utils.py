""" Author: Michel Deudon. Credits OSMNX. """

import math
import geopandas as gpd
from shapely.geometry import Point


def project_geometry(geometry, crs='+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs', to_crs=None, to_latlong=False):
    """
    Args:
        geometry: shapely Polygon or MultiPolygon, the geometry to project
        crs: dict or string or pyproj.CRS, the starting coordinate reference system of the passed-in geometry
        to_crs: dict or string or pyproj.CRS
        to_latlong : bool, if True, project from crs to lat-long, if False, project from crs to local UTM zone
    Returns:
        geometry_proj, crs: tuple, the projected shapely geometry and the crs of the projected geometry
    """

    gdf = gpd.GeoDataFrame()
    gdf.crs = crs
    gdf.gdf_name = 'geometry to project'
    gdf['geometry'] = None
    gdf.loc[0, 'geometry'] = geometry
    assert len(gdf) > 0, 'You cannot project an empty GeoDataFrame.'

    if to_crs is not None:
        gdf_proj = gdf.to_crs(to_crs)
    else:
        if to_latlong:
            latlong_crs = '+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs'
            gdf_proj = gdf.to_crs(latlong_crs) # project the gdf to latlong
        else:
            avg_longitude = gdf['geometry'].unary_union.centroid.x # calculate the centroid of the union of all the geometries in the GeoDataFrame
            utm_zone = int(math.floor((avg_longitude + 180) / 6.) + 1) # calculate the UTM zone from this avg longitude and define the UTM CRS to project
            utm_crs = '+proj=utm +zone={} +ellps=WGS84 +datum=WGS84 +units=m +no_defs'.format(utm_zone)
            gdf_proj = gdf.to_crs(utm_crs) # project the GeoDataFrame to the UTM CRS

    gdf_proj.gdf_name = gdf.gdf_name
    geometry_proj = gdf_proj['geometry'].iloc[0]
    return geometry_proj, gdf_proj.crs


def bbox_from_point(lat, lon, r=1000):
    """
    Args:
        lat: float, latitude
        lon: float, longitude
        r: int, how many meters the north, south, east, and west sides of the box should each be from the point
    Returns:
        north, south, east, west: tuple
    """

    # reverse the order of the (lat,lng) point so it is (x,y) for shapely, then project to UTM and buffer in meters
    point_proj, crs_proj = project_geometry(Point((lon, lat)))
    buffer_proj = point_proj.buffer(r)

    # project back to lat-long then get the bounding coordinates
    buffer_latlong, _ = project_geometry(buffer_proj, crs=crs_proj, to_latlong=True)
    x1, y1, x2, y2 = buffer_latlong.bounds # west, south, east, north
    return x1, y1, x2, y2