from __future__ import annotations

import awkward as ak
import numpy as np
import utils

def def_chain(funcs, kwargs_list):
    def func(data):
        tmp = data
        for f, kw in zip(funcs, kwargs_list):
            tmp = f(tmp, **kw)

        return tmp

    return func


def sort_data(obj):
    indices = np.lexsort((obj.time, obj.evtid))
    return obj[indices]


def group_by_evtid(data):
    counts = ak.run_lengths(data.evtid)
    return ak.unflatten(data, counts)


def group_by_time(obj, window=10):
    runs = np.array(np.cumsum(ak.run_lengths(obj.evtid)))
    counts = ak.run_lengths(obj.evtid)

    time_diffs = np.diff(obj.time)
    index_diffs = np.diff(obj.evtid)

    change_points = np.array(np.where((time_diffs > window * 1000) & (index_diffs == 0)))[0]
    total_change = np.sort(np.concatenate(([0], change_points, runs), axis=0))

    counts = ak.Array(np.diff(total_change))
    return ak.unflatten(obj, counts)


def sum_energy(grouped):
    sum_energy = ak.sum(grouped.edep, axis=-1)
    t0 = ak.fill_none(ak.firsts(grouped.time, axis=-1), np.nan)
    index = ak.fill_none(ak.firsts(grouped.evtid, axis=-1), np.nan)

    return ak.zip({"sum_energy": sum_energy, "t0": t0, "evtid": index})


def smear_energy(data, reso=2, energy_name="sum_energy"):
    flat_energy = data[energy_name].to_numpy()
    rng = np.random.default_rng()
    return ak.with_field(
        data, rng.normal(loc=flat_energy, scale=np.ones_like(flat_energy) * reso), "energy_smeared"
    )

def distance_to_surface(data,detector="det001"):

    # get detector origin
    x,y,z = utils.get_detector_origin(detector)
    
    # get the r-z points to produce the detector (G4GenericPolyCone)
    r,z  = utils.get_detector_corners(detector)

    # loop over pairs
    dists=[]
    for (rtmp,ztmp,rnext_tmp,znext_tmp) in zip(r[:-1],z[:-1],r[1:],z[1:]):
        
        s1 = ak.zip({"x":rtmp,"y":ztmp})
        s2 = ak.zip({"x":rnext_tmp,"y":znext_tmp})
        v_3d=ak.zip({"x":data.xloc-x,"y":data.yloc-y,"z":data.zloc-z})
        v = ak.zip({"x":np.sqrt(np.power(v_3d.x,2)+np.power(v_3d.y,2)),"y":v_3d.z})

        dist  = utils.dist(s1,s2,v)
        dists.append(dist)


    raise np.min(dists)
