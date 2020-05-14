# The MIT License (MIT)
# Copyright (c) 2019 by the xcube development team and contributors
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
# of the Software, and to permit persons to whom the Software is furnished to do
# so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import math
import warnings
from typing import Tuple, Union, Optional, Sequence, Dict, Any

import pandas as pd

DEFAULT_CRS = 'http://www.opengis.net/def/crs/EPSG/0/4326'
DEFAULT_TIME_TOLERANCE = '10M'
SH_MAX_IMAGE_SIZE = 2500
DEFAULT_TILE_SIZE = 1000


def _safe_int_div(x: int, y: int) -> int:
    return (x + y - 1) // y


class CubeConfig:
    """
    Sentinel Hub cube configuration.
    :param dataset_name: Dataset name. Mandatory.
    :param band_names: Band names. Mandatory.
    :param band_units: Band units. Optional.
    :param band_sample_types: Band sample types. Optional.
    :param tile_size: Tile size as tuple (width, height). Optional.
    :param geometry: Tuple x1,y1,x2,y2
    :param spatial_res:
    :param crs:
    :param time_range:
    :param time_period:
    :param time_tolerance:
    :param collection_id:
    :param four_d:
    :param exception_type:
    """

    def __init__(self,
                 dataset_name: str = None,
                 band_names: Sequence[str] = None,
                 band_units: Union[str, Sequence[str]] = None,
                 band_sample_types: Union[str, Sequence[str]] = None,
                 tile_size: Union[str, Tuple[int, int]] = None,
                 geometry: Union[str, Tuple[float, float, float, float]] = None,
                 spatial_res: float = 0.0001,                  
                 crs: str = None,
                 time_range: Union[str, pd.Timestamp, Tuple[str, str], Tuple[pd.Timestamp, pd.Timestamp]] = None,
                 time_period: Union[str, pd.Timedelta] = None,
                 time_tolerance: Union[str, pd.Timedelta] = None,
                 collection_id: str = None,
                 four_d: bool = False,
                 exception_type=ValueError):

        crs = crs or DEFAULT_CRS
        time_period = time_period or None
        time_tolerance = time_tolerance or None

        if not dataset_name:
            raise exception_type('dataset name must be given')
        if not band_names:
            raise exception_type('band names must be a given')
        if not geometry:
            raise exception_type('geometry must be given')
        if not time_range:
            raise exception_type('time range must be given')
        if time_period is None and time_tolerance is None:
            time_tolerance = DEFAULT_TIME_TOLERANCE

        if isinstance(geometry, str):
            x1, y1, x2, y2 = tuple(map(float, geometry.split(',', maxsplit=3)))
            geometry = x1, y1, x2, y2
        else:
            x1, y1, x2, y2 = geometry

        if isinstance(tile_size, tuple) or isinstance(tile_size, list):
            tile_width, tile_height = tile_size[0], tile_size[1]
        else:
            raise exception_type('tile_size must be a tuple')

        if tile_width > SH_MAX_IMAGE_SIZE:
            tile_width = SH_MAX_IMAGE_SIZE
        if tile_height > SH_MAX_IMAGE_SIZE:
            tile_height = SH_MAX_IMAGE_SIZE

        width, height = tile_width, tile_height

        if isinstance(time_range, str):
            time_range = tuple(map(lambda s: s.strip(),
                                   time_range.split(',', maxsplit=1) if ',' in time_range else (
                                       time_range, time_range)))
            time_range = tuple(time_range)
        if len(time_range) == 1:
            time_range = time_range + time_range
        if len(time_range) != 2:
            exception_type('Time range must be have two elements')

        start_time, end_time = tuple(time_range)
        if isinstance(start_time, str) or isinstance(end_time, str):
            def convert_time(time_str):
                return pd.to_datetime(time_str, utc=True)

            start_time, end_time = tuple(map(convert_time, time_range))

        time_range = start_time, end_time

        if isinstance(time_period, str):
            time_period = pd.to_timedelta(time_period)

        if isinstance(time_tolerance, str):
            time_tolerance = pd.to_timedelta(time_tolerance)

        self._dataset_name = dataset_name
        self._band_names = tuple(band_names)
        self._band_units = band_units or None
        self._band_sample_types = band_sample_types or None
        self._geometry = geometry
        self._spatial_res = spatial_res
        self._crs = crs
        self._time_range = time_range
        self._time_period = time_period
        self._time_tolerance = time_tolerance
        self._collection_id = collection_id
        self._four_d = four_d
        self._size = width, height
        self._tile_size = tile_width, tile_height
        self._num_tiles = width // tile_width, height // tile_height

    @classmethod
    def from_dict(cls, cube_config_dict: Dict[str, Any], exception_type=ValueError) -> 'CubeConfig':
        code = CubeConfig.__init__.__code__
        valid_keywords = set(code.co_varnames[1: code.co_argcount])
        given_keywords = set(cube_config_dict.keys())
        for keyword in cube_config_dict.keys():
            if keyword in valid_keywords:
                given_keywords.remove(keyword)
        if len(given_keywords) == 1:
            raise exception_type(f'Found invalid parameter {given_keywords.pop()!r} in cube configuration')
        elif len(given_keywords) > 1:
            given_keywords_text = ', '.join(map(lambda s: f'{s!r}', sorted(given_keywords)))
            raise exception_type(f'Found invalid parameters in cube configuration: {given_keywords_text}')
        return CubeConfig(exception_type=exception_type, **cube_config_dict)

    def as_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dictionary that can be passed to ctor as kwargs"""
        time_range = (self.time_range[0].isoformat(), self.time_range[1].isoformat()) \
            if self.time_range else None
        time_period = str(self.time_period) \
            if self.time_period else None
        time_tolerance = str(self.time_tolerance) \
            if self.time_tolerance else None
        return dict(dataset_name=self.dataset_name,
                    band_names=self.band_names,
                    band_units=self.band_units,
                    band_sample_types=self.band_sample_types,
                    tile_size=self.tile_size,
                    geometry=self.geometry,
                    spatial_res=self.spatial_res,
                    crs=self.crs,
                    time_range=time_range,
                    time_period=time_period,
                    time_tolerance=time_tolerance,
                    collection_id=self.collection_id,
                    four_d=self.four_d)

    @property
    def dataset_name(self) -> str:
        return self._dataset_name

    @property
    def band_names(self) -> Tuple[str, ...]:
        return self._band_names

    @property
    def band_units(self) -> Union[None, str, Tuple[str, ...]]:
        return self._band_units

    @property
    def band_sample_types(self) -> Union[None, str, Tuple[str, ...]]:
        return self._band_sample_types

    @property
    def crs(self) -> str:
        return self._crs

    @property
    def geometry(self) -> Tuple[float, float, float, float]:
        return self._geometry

    @property
    def spatial_res(self) -> float:
        return self._spatial_res

    @property
    def time_range(self) -> Tuple[pd.Timestamp, pd.Timestamp]:
        return self._time_range

    @property
    def time_period(self) -> Optional[pd.Timedelta]:
        return self._time_period

    @property
    def time_tolerance(self) -> Optional[pd.Timedelta]:
        return self._time_tolerance

    @property
    def collection_id(self) -> Optional[str]:
        return self._collection_id

    @property
    def four_d(self) -> bool:
        return self._four_d

    @property
    def size(self) -> Tuple[int, int]:
        return self._size

    @property
    def tile_size(self) -> Tuple[int, int]:
        return self._tile_size

    @property
    def num_tiles(self) -> Tuple[int, int]:
        return self._num_tiles

    @property
    def is_wgs84_crs(self) -> bool:
        return self._crs.endswith('/4326') or self._crs.endswith('/WGS84')

    @classmethod
    def _adjust_size(cls, size: int, tile_size: int) -> int:
        if size > tile_size:
            num_tiles = _safe_int_div(size, tile_size)
            size = num_tiles * tile_size
        return size