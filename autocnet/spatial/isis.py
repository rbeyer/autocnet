import pvl
from warnings import warn
from numbers import Number
import numpy as np
from subprocess import CalledProcessError

import kalasiris as isis


def point_info(cube_path, x, y, point_type, allow_outside=False):
    """
    Use Isis's campt to get image/ground point info from an image

    Parameters
    ----------
    cube_path : str
                path to the input cube

    x : float
        point in the x direction. Either a sample or a longitude value
        depending on the point_type flag

    y : float
        point in the y direction. Either a line or a latitude value
        depending on the point_type flag

    point_type : str
                 Options: {"image", "ground"}
                 Pass "image" if  x,y are in image space (sample, line) or
                 "ground" if in ground space (longitude, lattiude)

    Returns
    -------
    : PvlObject
      Pvl object containing campt returns
    """
    point_type = point_type.lower()

    if point_type not in {"image", "ground"}:
        raise Exception(f'{point_type} is not a valid point type, valid types are ["image", "ground"]')


    if isinstance(x, Number) and isinstance(y, Number):
        x, y = [x], [y]

    if point_type == "image":
        # convert to ISIS pixels
        x = np.add(x, .5)
        y = np.add(y, .5)

    if pvl.load(cube_path).get("IsisCube").get("Mapping"):
      pvlres = []
      # We have a projected image
      for x,y in zip(x,y):
        try:
          if point_type.lower() == "ground":
            pvlres.append(
                isis.mappt(
                    cube_path,
                    longitude=x,
                    latitude=y,
                    allowoutside=allow_outside,
                    coordsys="UNIVERSAL",
                    type=point_type
                ).stdout
            )
          elif point_type.lower() == "image":
            pvlres.append(
                isis.mappt(
                    cube_path,
                    sample=x,
                    line=y,
                    allowoutside=allow_outside,
                    type=point_type
                    ).stdout
            )
        except CalledProcessError as e:
          print(f"MAPPT call failed, image: {cube_path}\n{e.stderr}")
          return
      dictres = [dict(pvl.loads(res)["Results"]) for res  in pvlres]
      if len(dictres) == 1:
        pvlres = dictres[0]

    else:
        if point_type == "ground":
            # campt uses lat, lon for ground but sample, line for image.
            # So swap x,y for ground-to-image calls
            x, y = y, x

        # ISIS's campt wants points in a file, so write to a temp file
        with isis.fromlist.temp(
                [f"{xval}, {yval}" for xval, yval in zip(x, y)]
        ) as f:
            try:
                cp = isis.campt(
                    cube_path,
                    coordlist=f,
                    allowoutside=allow_outside,
                    usecoordlist=True,
                    coordtype=point_type
                )
            except CalledProcessError as e:
                warn(f"CAMPT call failed, image: {cube_path}\n{e.stderr}")
                return

        pvlres = pvl.loads(cp.stdout)
        dictres = []
        if len(x) > 1 and len(y) > 1:
            for r in pvlres.getall("GroundPoint"):
                if r['Error'] is not None:
                    raise CalledProcessError(
                        returncode=1,
                        cmd=cp.args,
                        stdout=r,
                        stderr=r['Error'])
                else:
                    # convert all pixels to PLIO pixels from ISIS
                    r["Sample"] -= .5
                    r["Line"] -= .5
                    dictres.append(dict(r))
        else:
            if pvlres['GroundPoint']['Error'] is not None:
                # This probably isn't the right exception to call.
                raise CalledProcessError(
                    returncode=1,
                    cmd=cp.args,
                    # stdout=pvlres,
                    # stderr=pvlres['GroundPoint']['Error']
                )
            else:
                pvlres["GroundPoint"]["Sample"] -= .5
                pvlres["GroundPoint"]["Line"] -= .5
                dictres = dict(pvlres["GroundPoint"])
    return dictres


def image_to_ground(cube_path, sample, line, lattype="PlanetocentricLatitude", lonttype="PositiveEast360Longitude"):
    """
    Use Isis's campt to convert a line sample point on an image to lat lon

    Returns
    -------
    lats : np.array, float
           1-D array of latitudes or single floating point latitude

    lons : np.array, float
           1-D array of longitudes or single floating point longitude

    """
    res = point_info(cube_path, sample, line, "image")

    try:
        if isinstance(res, list):
            lats, lons = np.asarray([[r[lattype].value, r[lonttype].value] for r in res]).T
        else:
            lats, lons = res[lattype].value, res[lonttype].value
    except Exception as e:
        if isinstance(res, list):
            lats, lons = np.asarray([[r[lattype], r[lonttype]] for r in res]).T
        else:
            lats, lons = res[lattype], res[lonttype]
    return lats, lons


def ground_to_image(cube_path, lon, lat):
    """
    Use Isis's campt to convert a lat lon point to line sample in
    an image

    Returns
    -------
    lines : np.array, float
            array of lines or single floating point line

    samples : np.array, float
              array of samples or single dloating point sample

    """
    res = point_info(cube_path, lon, lat, "ground")

    try:
        if isinstance(res, list):
            lines, samples = np.asarray([[r["Line"], r["Sample"]] for r in res]).T
        else:
            lines, samples =  res["Line"], res["Sample"]
    except:
        raise Exception(res)

    return lines, samples


