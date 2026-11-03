"""Map image-derived Homr staff grids back to photographed pixel coordinates."""
import argparse,json,hashlib
from pathlib import Path
import cv2
import numpy as np
from homr.autocrop import autocrop

def map_photo(photo,source):
        original=cv2.imread(str(photo))
        if original is None:raise ValueError('photograph unreadable')
        crop=autocrop(original)
        # Upstream autocrop returns the original or a slice, so its exact crop
        # offset is available without matching content or consulting a score.
        assert np.shares_memory(crop,original)
        offset=crop.ctypes.data-original.ctypes.data
        top=offset//original.strides[0];left=(offset%original.strides[0])//original.strides[1]
        geo=json.loads(source.read_text(encoding='utf8'));width,height=geo['preprocessed_size']
        sx,sy=crop.shape[1]/width,crop.shape[0]/height
        staves=[]
        for si,system in enumerate(geo['systems']):
            for physical in system['staffs']:
                grid=physical['grid'];count=len(grid[0]['y'])//5
                for index in range(count):
                    lines=[]
                    for k in range(5):
                        pts=[[(p['x']+.5)*sx-.5+left,(p['y'][5*index+k]+.5)*sy-.5+top] for p in grid]
                        lines.append({'number':k+1,'start':pts[0],'end':pts[-1],'polyline':pts})
                    space=float(np.median([np.diff(p['y'][5*index:5*index+5]) for p in grid]))*sy
                    staves.append({'index':len(staves),'system':si+1,'space_px':space,'lines':lines})
        result={'image':str(photo),'image_sha256':hashlib.sha256(photo.read_bytes()).hexdigest(),'image_size_px':[original.shape[1],original.shape[0]],
                'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'crop_size_px':[crop.shape[1],crop.shape[0]],'staves':staves,
                'crop_xy':[int(left),int(top)],'source':str(source),
                'reference_used':False,'coordinate_space':'original photographed pixels',
                'geometry_source':'Frozen Homr segmentation of this same photograph'}
        return result

def main():
    p=argparse.ArgumentParser();p.add_argument('photo',type=Path,nargs='?');p.add_argument('geometry',type=Path,nargs='?');p.add_argument('output',type=Path,nargs='?');args=p.parse_args()
    if args.photo:
        if not args.geometry or not args.output:p.error('photo, geometry and output are required together')
        result=map_photo(args.photo,args.geometry);args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(result,indent=2)+'\n',encoding='utf8');return
    root=Path('build/route_trials_20261010');out=Path('build/staff_removal_20261010/geometries');out.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((Path('build/relation_decoder_20261010')/'manifest.json').read_text(encoding='utf8'))
    for row in manifest['rows']:
        source=root/'homr465_full'/row['id']/(row['page']+'.geometry.json')
        result=map_photo(Path(row['photo']),source)
        (out/(row['id']+'_'+row['page']+'.json')).write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')
    print(json.dumps({'pages':len(manifest['rows']),'reference_used':False}),flush=True)

if __name__=='__main__':main()
