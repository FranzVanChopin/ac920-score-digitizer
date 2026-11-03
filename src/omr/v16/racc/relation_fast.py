"""Exact, faster relation stage (Codex route_trials, board build of 10-24), installed at run time.

install(rd) patches the already imported route_trials.relation_decode module `rd`; Codex's files
are never edited.  Two changes, both returning exactly what the originals return:

  testset3_stroke_group_cache.strokes   the grouping loop over the Hough segments in C++
                                        (hr_stroke_groups): same filters, same comparisons, same
                                        first-match order; a group's np.mean of half-integer centres
                                        is exact_sum / n (every partial sum is exact in float64),
                                        min / max are kept incrementally
  link_heads                            per-stem x mean / sorted y computed once per stem (not per
                                        head x stem pair); distance screen on arrays, exact negation
                                        of Codex's test; edge model on the survivors in stem order
  beam_relations                        per-row `mean(row >= t) >= .32` on the whole block at once
                                        (row count / width = the same float64 mean)
  recognize (one block, source patch)   staff-scale inference reordered: stroke tiles first, then
                                        the stems and bars strokes() (each mostly one HoughLinesP,
                                        single-threaded C++ that releases the GIL) run in two threads
                                        while the symbol tiles run; all three are pure functions of
                                        their inputs, so the order does not change any value

HOMR_FAST_CHECK=1 (shared with homr_fast) also runs Codex's original code and raises on any
difference.  Nothing is installed if a patched source differs from the reviewed version
(sha256 below), so a Codex update falls back to its own code.
"""
import concurrent.futures as cf
import ctypes
import hashlib
import inspect
import os
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
CHECK = os.environ.get("HOMR_FAST_CHECK") == "1"
# link_heads screen: a head may lie this many staff spaces beyond a detected stem end (Codex: .35, kept
# as the default so the output stays exact; route A sets 0.5 via run_page.py --link-margin)
LINK_MARGIN = .35
# route A shared network: a callable (raw, staffs) -> (predictions, stroke probabilities, report) in the legacy
# graph frame, used instead of the symbol / stroke tiles; None (default) keeps the exact path
SHARED = None

# sha256[:16] of the reviewed Codex sources (board build, app/route_trials, LF line endings)
SOURCES = {
    "recognize": "e8e3fb665aff3b82",
    "infer": "00740865d6687943",
    "strokes": "460f1ae6e73f1882",
    "link_heads": "1bfda4aee08bb467",
    "beam_relations": "0f861f5887d209a6",
    "assign_graph": "9d76f49a15437a0a",
}

OLD_BLOCK = """    else:
        from .staff_scale_inference import infer
        predictions,prob,inference_profile=infer(symbol,stroke,raw,payload['staves'],staff_scale)
    from .testset3_stroke_group_cache import strokes as grouped_strokes
    stems=grouped_strokes(prob[0],rth[0]);bars=grouped_strokes(prob[2],rth[2],minimum=2.8)
"""
NEW_BLOCK = """    else:
        predictions,prob,inference_profile,stems,bars=_RELATION_FAST_INFER(symbol,stroke,raw,payload['staves'],staff_scale,rth)
    if staff_scale is None:
        from .testset3_stroke_group_cache import strokes as grouped_strokes
        stems=grouped_strokes(prob[0],rth[0]);bars=grouped_strokes(prob[2],rth[2],minimum=2.8)
"""


def _sha(src):
    return hashlib.sha256(src.replace("\r\n", "\n").encode("utf-8")).hexdigest()[:16]


def _lib():
    path = HERE / ("relgeo.dll" if os.name == "nt" else "librelgeo.so")
    if not path.exists():
        return None
    lib = ctypes.CDLL(str(path))
    lib.hr_stroke_groups.restype = ctypes.c_int
    lib.hr_stroke_groups.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_int32), ctypes.c_double, ctypes.c_double,
                                     ctypes.POINTER(ctypes.c_int)]
    return lib


def _differ(name, a, b):
    raise AssertionError(f"relation_fast: {name} differs from Codex's original: {a!r} != {b!r}")


def install(rd):
    import cv2
    from route_trials import relation_net as rn
    from route_trials import staff_scale_inference as ssi
    from route_trials import symbol_cnn as sn
    from route_trials import testset3_stroke_group_cache as sgc

    report = {"installed": [], "skipped": []}
    if getattr(rd, "_relation_fast", False):
        report["skipped"].append("already installed")
        return report
    srcs = {"recognize": inspect.getsource(rd.recognize), "infer": inspect.getsource(ssi.infer),
            "strokes": inspect.getsource(sgc.strokes), "link_heads": inspect.getsource(rd.link_heads),
            "beam_relations": inspect.getsource(rd.beam_relations), "assign_graph": inspect.getsource(rd.assign_graph)}
    for name, src in srcs.items():
        if SOURCES[name] is not None and _sha(src) != SOURCES[name]:
            report["skipped"].append(f"{name}: source differs from the reviewed version ({_sha(src)})")
            return report
    if srcs["recognize"].replace("\r\n", "\n").count(OLD_BLOCK) != 1:
        report["skipped"].append("recognize: block not found")
        return report
    lib = _lib()
    if lib is None:
        report["skipped"].append("C++ library missing")
        return report

    # ------------------------------------------------------------ strokes
    orig_strokes = sgc.strokes

    def strokes(prob, threshold, space=16., minimum=1.6):
        """Codex's strokes(); only the group assignment runs in C++."""
        mask = (prob >= round(threshold*255)).astype('uint8')
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5,1),np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((round(minimum*space),1),np.uint8))
        cv2.setRNGSeed(7123)
        lines = cv2.HoughLinesP(mask, 1, np.pi/720, threshold=round(minimum*space*.65),
                                minLineLength=round(minimum*space), maxLineGap=round(.3*space))
        segs = lines.reshape(-1, 4).tolist() if lines is not None else []
        segs = sorted(segs, key=lambda p: -abs(p[3]-p[1]))
        n = len(segs)
        arr = np.ascontiguousarray(np.array(segs, dtype=np.int32).reshape(-1, 4))
        out = np.empty(max(1, n), dtype=np.int32)
        k = lib.hr_stroke_groups(n, arr.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)), float(space), float(minimum),
                                 out.ctypes.data_as(ctypes.POINTER(ctypes.c_int)))
        if k < 0:
            return orig_strokes(prob, threshold, space, minimum)
        groups = [[] for _ in range(k)]
        for (x1, y1, x2, y2), g in zip(segs, out[:n].tolist()):
            if g < 0:
                continue
            if y1 > y2:
                x1, y1, x2, y2 = x2, y2, x1, y1
            groups[g].append([x1, y1, x2, y2])
        result = []
        for group in groups:
            xy = np.array(group).reshape(-1,2)
            lo,hi = int(xy[:,1].min()), int(xy[:,1].max())
            slope,intercept = np.polyfit(xy[:,1],xy[:,0],1)
            a,b = float(slope*lo+intercept), float(slope*hi+intercept)
            yy = np.arange(lo,hi+1)
            xx = np.clip(np.rint(slope*yy+intercept).astype(int),0,prob.shape[1]-1)
            result.append({'id':len(result), 'xy':[[a,float(lo)],[b,float(hi)]],
                           'confidence':float(prob[yy,xx].mean()/255),
                           'bbox':[int(min(a,b)),lo,max(1,int(abs(b-a))+1),hi-lo+1]})
        return result

    def strokes_checked(prob, threshold, space=16., minimum=1.6):
        r, o = strokes(prob, threshold, space, minimum), orig_strokes(prob, threshold, space, minimum)
        if r != o:
            _differ("strokes", len(r), len(o))
        return r

    sgc.strokes = strokes_checked if CHECK else strokes
    report["installed"].append("strokes")

    # ------------------------------------------------------------ infer + strokes, reordered
    orig_infer = ssi.infer

    def infer_and_strokes(symbol, stroke, raw, staffs, target_space, rth):
        """ssi.infer (copied) with the stroke tiles first; stems/bars strokes() overlap the symbol tiles."""
        if SHARED is not None:                     # route A: maps of the shared network, no tiles
            # (predictions, maps, report, stems, bars), stems = sgc.strokes(maps[0], rth[0]) and
            # bars = sgc.strokes(maps[2], rth[2], minimum=2.8), possibly prefetched during the segment stage
            return SHARED(raw, staffs, rth, sgc.strokes)
        spaces=np.asarray([s['space_px'] for s in staffs],dtype='float64')
        if not len(spaces) or not np.isfinite(spaces).all() or np.any(spaces<=0):
            raise ValueError('Invalid image-derived staff spacing')
        median=float(np.median(spaces));factor=target_space/median
        if not 8<=target_space<=24 or not .25<=factor<=2:
            raise ValueError('Staff normalization exceeds bounded scale range')
        h,w=raw.shape;size=(round(w*factor),round(h*factor))
        if min(size)<32 or size[0]*size[1]>16000000 or max(size)>8192:
            raise ValueError('Normalized inference exceeds image memory guard')
        scaled=cv2.resize(raw,size,interpolation=cv2.INTER_AREA if factor<1 else cv2.INTER_LINEAR)
        prob=rn.predict(stroke,scaled)
        legacy=(round(w*sn.FACTOR),round(h*sn.FACTOR))
        projected=np.empty((6,legacy[1],legacy[0]),dtype=np.uint8)
        for k in range(6):projected[k]=cv2.resize(prob[k],legacy,interpolation=cv2.INTER_NEAREST)
        grouped_strokes = sgc.strokes
        with cf.ThreadPoolExecutor(2) as ex:
            f_stems = ex.submit(grouped_strokes, projected[0], rth[0])
            f_bars = ex.submit(grouped_strokes, projected[2], rth[2], minimum=2.8)
            predictions=sn.predict(symbol,scaled)
            stems, bars = f_stems.result(), f_bars.result()
        # Resize uses half-pixel centres. Separate x/y ratios preserve rounding.
        predictions[:,1]=(predictions[:,1]+.5)*legacy[0]/size[0]-.5
        predictions[:,2]=(predictions[:,2]+.5)*legacy[1]/size[1]-.5
        report={'mode':'staff','target_space_px':target_space,'measured_median_space_px':median,
                'scale':factor,'inference_size':list(size),'graph_size':list(legacy),
                'tile_shape':[sn.W,sn.H],'tile_overlap_px':32,'sequential_tiles':True,
                'coordinate_space':'photograph pixels scaled by 16/19.8',
                'reference_supplied_to_recognizer':False}
        return predictions,projected,report,stems,bars

    def infer_checked(symbol, stroke, raw, staffs, target_space, rth):
        if SHARED is not None:                     # nothing exact to compare the shared network with
            return infer_and_strokes(symbol, stroke, raw, staffs, target_space, rth)
        r = infer_and_strokes(symbol, stroke, raw, staffs, target_space, rth)
        p, q, rep = orig_infer(symbol, stroke, raw, staffs, target_space)
        st, ba = orig_strokes(q[0], rth[0]), orig_strokes(q[2], rth[2], minimum=2.8)
        same = (np.array_equal(r[0], p) and r[0].dtype == p.dtype and np.array_equal(r[1], q) and r[2] == rep
                and r[3] == st and r[4] == ba)
        if not same:
            _differ("infer + strokes", (len(r[3]), len(r[4])), (len(st), len(ba)))
        return r

    # ------------------------------------------------------------ link_heads
    orig_link = rd.link_heads

    def link_heads(nodes, stems, net, threshold, space):
        """Codex's link_heads; the per-stem numbers (x mean, sorted y) are computed once per stem
        instead of once per (head, stem) pair, and the distance screen runs on arrays."""
        links = []
        if stems:
            sx_l, lo_l, hi_l = [], [], []
            for stem in stems:
                xy = np.array(stem['xy']); sx = xy[:, 0].mean(); lo, hi = sorted(xy[:, 1])
                sx_l.append(sx); lo_l.append(lo); hi_l.append(hi)
            sx_a = np.array(sx_l, dtype=np.float64); lo_a = np.array(lo_l, dtype=np.float64)
            hi_a = np.array(hi_l, dtype=np.float64)
            if not (sx_a.dtype == np.float64 and all(type(v) is np.float64 for v in sx_l + lo_l + hi_l)):
                return orig_link(nodes, stems, net, threshold, space)
        for head in nodes:
            if head['class'] >= 2:
                continue
            hx, hy = head['xy']; candidates = []
            if stems:
                # original: `if abs(sx-hx)>1.4*space or not lo-.35*space<hy<hi+.35*space: continue`
                # LINK_MARGIN (module setting, default .35 = Codex's) is the only change allowed here
                keep = ~(np.abs(sx_a - hx) > 1.4*space) & (lo_a - LINK_MARGIN*space < hy) & (hy < hi_a + LINK_MARGIN*space)
                for i in np.flatnonzero(keep).tolist():
                    stem = stems[i]
                    features = rd.rn.edge_features(head['xy'], stem['xy'], space)
                    p = float(rd.probabilities(net, np.asarray(features, dtype='float32')).reshape(-1)[0])
                    if p >= threshold:
                        candidates.append((p, stem['id']))
            if candidates:
                confidence, stem = max(candidates); links.append({'head': head['id'], 'stem': stem, 'confidence': confidence})
        return links

    def link_checked(nodes, stems, net, threshold, space):
        r, o = link_heads(nodes, stems, net, threshold, space), orig_link(nodes, stems, net, threshold, space)
        if r != o and LINK_MARGIN == .35:            # a changed margin is meant to differ
            _differ("link_heads", len(r), len(o))
        return r

    rd.link_heads = link_checked if CHECK else link_heads
    report["installed"].append("link_heads")

    # ------------------------------------------------------------ beam_relations
    orig_beam = rd.beam_relations

    def beam_relations(stem, heads, prob, thresholds, space):
        """Codex's beam_relations; the per-row `mean(row >= t) >= .32` test runs on the whole block
        (row counts of a bool block / row width = the same float64 means)."""
        xy=np.array(stem['xy']);sx=float(xy[:,0].mean());hy=float(np.mean([h['xy'][1] for h in heads]))
        lo,hi=sorted(xy[:,1]);direction='up' if hy-lo>hi-hy else 'down';tip=lo if direction=='up' else hi
        y0=max(0,round(tip-.45*space if direction=='up' else tip-2.1*space));y1=min(prob.shape[1],round(tip+2.1*space if direction=='up' else tip+.45*space)+1)
        left=max(0,round(sx-.6*space));right=min(prob.shape[2],round(sx+.6*space)+1)
        if y1 > y0:
            block = prob[1,y0:y1,left:right]>=round(thresholds[1]*255)
            width = block.shape[1]
            rows = (block.sum(axis=1)/width>=.32).tolist() if width else [False]*(y1-y0)
        else:
            rows = []
        runs=np.diff(np.r_[False,rows,False].astype('int8'))
        starts=np.where(runs==1)[0];ends=np.where(runs==-1)[0]
        beams=min(3,sum(end-start>=.14*space for start,end in zip(starts,ends)))
        flags=0
        if not beams:
            left=max(0,round(sx+.05*space));right=min(prob.shape[2],round(sx+1.1*space))
            for c in range(3,6):
                crop=prob[c,y0:y1,left:right]
                if crop.size and (crop>=round(thresholds[c]*255)).sum()>.08*space*space:flags=c-2
        evidence=[{'y_range':[int(y0+a),int(y0+b)],'stem_x':sx} for a,b in zip(starts,ends) if b-a>=.14*space]
        return direction,int(beams),flags,float(tip),evidence

    def beam_checked(stem, heads, prob, thresholds, space):
        r, o = beam_relations(stem, heads, prob, thresholds, space), orig_beam(stem, heads, prob, thresholds, space)
        if r != o:
            _differ("beam_relations", r[:4], o[:4])
        return r

    rd.beam_relations = beam_checked if CHECK else beam_relations
    report["installed"].append("beam_relations")

    # ------------------------------------------------------------ assign_graph
    orig_assign = rd.assign_graph

    def distance_table(all_nodes, staffs):
        """staff_distance(staff, x, y) for every node and staff: np.interp of an array returns per
        element what the scalar calls return (same interval search result and slope formula, for
        strictly increasing polyline x); max/divide stay python floats as in staff_distance.
        None for a staff whose polylines are not strictly increasing in x (scalar calls then)."""
        xs = np.array([n['xy'][0] for n in all_nodes], dtype=np.float64)
        ys = [n['xy'][1] for n in all_nodes]
        table = []
        for staff in staffs:
            top_xy = np.array(staff['lines'][0]['polyline']).T
            bot_xy = np.array(staff['lines'][-1]['polyline']).T
            if (top_xy.ndim != 2 or bot_xy.ndim != 2 or not (np.diff(top_xy[0]) > 0).all()
                    or not (np.diff(bot_xy[0]) > 0).all() or not len(xs)):
                table.append(None)
                continue
            tops = np.interp(xs, *top_xy).tolist()
            bots = np.interp(xs, *bot_xy).tolist()
            sp = staff['space_px']
            table.append([max(t - y, y - b, 0) / sp for t, b, y in zip(tops, bots, ys)])
        return table

    def assign_graph(pred, staffs, thresholds, stems, net, threshold, gray=None, gate_report=None):
        """Codex's assign_graph (gray=None path); staff distances from distance_table."""
        if gray is not None:
            return orig_assign(pred, staffs, thresholds, stems, net, threshold, gray, gate_report)
        all_nodes=[]
        for c,x,y,confidence in pred:
            c=int(c)
            if confidence>=thresholds[c]:all_nodes.append({'id':len(all_nodes),'class':c,'xy':[float(x),float(y)],'confidence':float(confidence)})
        global_links=rd.link_heads(all_nodes,stems,net,threshold,16.)
        table = distance_table(all_nodes, staffs)

        def dist(i, node):
            row = table[i]
            return rd.staff_distance(staffs[i], *node['xy']) if row is None else row[node['id']]

        by_stem=rd.defaultdict(list)
        for link in global_links:by_stem[link['stem']].append(all_nodes[link['head']])
        ownership={}
        for heads in by_stem.values():
            def cost(index):
                staff=staffs[index];distances=[dist(index, h) for h in heads]
                x=float(np.mean([h['xy'][0] for h in heads]));left=staff['lines'][0]['start'][0];right=staff['lines'][0]['end'][0]
                if not left-staff['space_px']<x<right+staff['space_px']:return float('inf')
                return -sum(d<=.35 for d in distances)+.05*sum(min(d,12) for d in distances)
            index=min(range(len(staffs)),key=cost)
            for head in heads:ownership[head['id']]=index
        result=[[] for _ in staffs];mapping={}
        for node in all_nodes:
            x,y=node['xy'];eligible=[(dist(i, node),i) for i,s in enumerate(staffs) if s['lines'][0]['start'][0]-.5*s['space_px']<x<s['lines'][0]['end'][0]+.5*s['space_px']]
            if not eligible:continue
            distance,index=min(eligible);index=ownership.get(node['id'],index)
            if distance>(5 if 2<=node['class']<=7 else 12):continue
            new=dict(node,id=len(result[index]));mapping[node['id']]=(index,new['id']);result[index].append(new)
        links=[[] for _ in staffs]
        for link in global_links:
            if link['head'] in mapping:
                index,head=mapping[link['head']];links[index].append(dict(link,head=head))
        return result,links

    def assign_checked(pred, staffs, thresholds, stems, net, threshold, gray=None, gate_report=None):
        r = assign_graph(pred, staffs, thresholds, stems, net, threshold, gray, gate_report)
        o = orig_assign(pred, staffs, thresholds, stems, net, threshold, gray, gate_report)
        if r != o:
            _differ("assign_graph", [len(v) for v in r[0]], [len(v) for v in o[0]])
        return r

    rd.assign_graph = assign_checked if CHECK else assign_graph
    report["installed"].append("assign_graph (distance table)")

    # ------------------------------------------------------------ bar_bounds
    orig_bar = rd.bar_bounds

    def crop_box(xx, yy, shape, mx, my):
        """[y0, y1) x [x0, x1) holding every sample's bilinear neighbourhood plus mx / my pixels (the
        dilation radius and a safety pixel), clipped to the image; None if a coordinate is not finite."""
        if not (np.isfinite(xx).all() and np.isfinite(yy).all()):
            return None
        x0 = max(0, int(np.floor(xx.min())) - mx); x1 = min(shape[1], int(np.ceil(xx.max())) + 2 + mx)
        y0 = max(0, int(np.floor(yy.min())) - my); y1 = min(shape[0], int(np.ceil(yy.max())) + 2 + my)
        return (y0, y1, x0, x1) if x1 > x0 and y1 > y0 else None

    def bar_bounds(staffs, bars, prob=None, image=None, music_nodes=None):
        """Codex's bar_bounds; the full-page dilations are computed on a crop around the sampled
        points.  Dilation is local (margin >= kernel radius + 1; real image borders stay borders),
        and bilinear remap is translation-exact for integer crop offsets (float32 x - x0 is exact,
        the subpixel table index is unchanged), so every sampled value is the full-page one."""
        space=float(np.median([s['space_px'] for s in staffs]));left=min(s['lines'][0]['start'][0] for s in staffs);right=max(s['lines'][0]['end'][0] for s in staffs)
        if image is not None:
            xs=np.arange(max(0,round(right-6*space)),min(image.shape[1],round(right+space)),dtype='float32')
            rows = [np.array([np.interp(xs,*np.array(line['polyline']).T) for line in staff['lines']],dtype='float32')
                    for staff in staffs]
            box = crop_box(xs, np.concatenate([r.ravel() for r in rows]) if rows else xs, image.shape, 3, 5) \
                if len(xs) else None
            if box is None:
                return orig_bar(staffs, bars, prob, image, music_nodes)
            y0, y1, x0, x1 = box
            vertical=cv2.dilate((image[y0:y1, x0:x1]<155).astype('uint8'),np.ones((5,1),np.uint8)).astype('float32')
            ends=[]
            for staff, yy in zip(staffs, rows):
                xx=np.broadcast_to(xs,yy.shape).copy();q=cv2.remap(vertical,xx-np.float32(x0),yy-np.float32(y0),cv2.INTER_LINEAR)
                supported=xs[(q>.5).sum(0)>=4]
                if len(supported)>space:ends.append(float(supported[-1]))
            if ends:right=max(ends)
        hits=[]
        for bar in bars:
            a,b=np.array(bar['xy']);x=float((a[0]+b[0])/2);lo,hi=sorted((a[1],b[1]))
            if not left+2.5*space<x<right+.8*space:continue
            covered=[]
            for staff in staffs:
                top=np.interp(x,*np.array(staff['lines'][0]['polyline']).T);bottom=np.interp(x,*np.array(staff['lines'][-1]['polyline']).T)
                covered.append(lo<top+.35*space and hi>bottom-.35*space)
            if any(covered):hits.append(x)
        if prob is not None:
            hits=[]
            radius=max(1,round(.12*space))
            xs=np.arange(max(0,round(left+2.5*space)),min(image.shape[1],round(right+.8*space)),dtype='float32')
            fractions=np.linspace(.05,.95,48,dtype='float32')[:,None]
            maps = []
            for staff in staffs:
                top=np.interp(xs,*np.array(staff['lines'][0]['polyline']).T)
                bottom=np.interp(xs,*np.array(staff['lines'][-1]['polyline']).T)
                yy=(top[None]+fractions*(bottom-top)[None]).astype('float32');xx=np.broadcast_to(xs,yy.shape).copy()
                maps.append((xx, yy))
            box = crop_box(xs, np.concatenate([m[1].ravel() for m in maps]), image.shape, radius + 3, 3) \
                if len(xs) and maps else None
            if box is None:
                return orig_bar(staffs, bars, prob, image, music_nodes)
            y0, y1, x0, x1 = box
            likelihood=cv2.dilate(np.ascontiguousarray(prob[2][y0:y1, x0:x1]),np.ones((1,2*radius+1),np.uint8)).astype('float32')/255
            physical=cv2.dilate((image[y0:y1, x0:x1]<155).astype('uint8'),np.ones((1,2*radius+1),np.uint8)).astype('float32')
            for xx, yy in maps:
                mx_, my_ = xx-np.float32(x0), yy-np.float32(y0)
                q=cv2.remap(likelihood,mx_,my_,cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)
                ink=cv2.remap(physical,mx_,my_,cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)
                accepted=(q.mean(0)>=.55)&((q>.6).mean(0)>=.45)&(ink.mean(0)>=.82)
                hits.extend(xs[accepted].tolist())
        boundaries=rd.cluster_x(hits,1.1*space)
        if boundaries and music_nodes is not None:
            last=boundaries[-1]
            trailing=[n for n in music_nodes if n['class']<=7 and last+.3*space<n['xy'][0]<right]
            if .85*space<right-last<4*space and not trailing:right=last
        if not boundaries or right-boundaries[-1]>.85*space:boundaries.append(float(right))
        else:boundaries[-1]=float(right)
        return [float(left)]+boundaries

    def bar_checked(staffs, bars, prob=None, image=None, music_nodes=None):
        r, o = bar_bounds(staffs, bars, prob, image, music_nodes), orig_bar(staffs, bars, prob, image, music_nodes)
        if r != o:
            _differ("bar_bounds", r, o)
        return r

    rd.bar_bounds = bar_checked if CHECK else bar_bounds
    report["installed"].append("bar_bounds (cropped dilations)")

    rd._RELATION_FAST_INFER = infer_checked if CHECK else infer_and_strokes
    src = srcs["recognize"].replace("\r\n", "\n").replace(OLD_BLOCK, NEW_BLOCK)
    exec(compile(src, rd.__file__, "exec"), rd.__dict__)          # redefines rd.recognize, same globals
    rd._relation_fast = True
    report["installed"].append("recognize: stroke tiles first, strokes() overlapping the symbol tiles")
    return report
