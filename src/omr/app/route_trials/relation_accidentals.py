"""Unique photographed accidental attachment and bounded bar-local state.

No alteration is inferred from harmony. A visual accidental seeds state;
later notes need an empty accidental strip before inheriting that state.
"""
import hashlib
from pathlib import Path
from collections import defaultdict
import cv2
import xml.etree.ElementTree as E
from omr_bar_geometry import measure_systems
from omr_event_records import records
from repair_pitch_context import no_accidental
from .symbol_cnn import FACTOR

NAMES={8:(1,'sharp'),9:(-1,'flat'),10:(0,'natural')}


def associations(row):
    heads=[h for g in row['groups'] if not g.get('grace') for h in g.get('heads',[])]
    accs=[a for a in row['nodes'] if a['class'] in NAMES and a['confidence']>=(.85 if a['class']==10 else .9)]
    result=defaultdict(list);space=row['space_px']
    for acc in accs:
        eligible=[]
        for h in heads:
            dx=(h['xy'][0]-acc['xy'][0])/space
            dy=(acc['xy'][1]-h['xy'][1])/space
            target=-.75 if acc['class']==9 else 0
            if .65<dx<3.3 and abs(dy-target)<(.45 if acc['class']==9 else .3):
                eligible.append((abs(dy-target)+.15*dx,h))
        eligible.sort(key=lambda v:v[0])
        if not eligible or len(eligible)>1 and eligible[1][0]-eligible[0][0]<.15:
            continue
        cost,h=eligible[0];result[h['id']].append((cost,acc))
    mapped={}
    for hid,items in result.items():
        items.sort(key=lambda v:v[0])
        if len(items)>1 and items[1][0]-items[0][0]<.15:
            continue
        mapped[hid]=items[0][1]
    return mapped


def fuse_accidentals(root,graph,photo,staff_geometry,local_alignment=False):
    digest=hashlib.sha256(Path(photo).read_bytes()).hexdigest()
    if graph.get('image_sha256')!=digest or staff_geometry.get('image_sha256')!=digest:
        return []
    systems=measure_systems(root)
    if len(systems)>64 or len(root.findall('part'))!=1:
        return []
    # The extension is additive. Its more conservative seed rule must never
    # suppress a repair accepted by the existing positional matcher.
    legacy=fuse_accidentals(root,graph,photo,staff_geometry,local_alignment=False) if local_alignment else []
    rec={m:r for m,r in zip(root.findall('part/measure'),records(root))}
    ink=None;changes=list(legacy)
    for si,ms in enumerate(systems,1):
        for row in [r for r in graph['systems'] if r['system']==si]:
            bounds=row['bounds'];space=row['space_px'];sn=row['staff']
            if len(bounds)!=len(ms)+1 or len(bounds)>129 or len(row['groups'])>2048 or len(row['nodes'])>2048:
                continue
            attached=associations(row)
            all_heads=[{'x':h['xy'][0],'y':h['xy'][1],'staff_space':space}
                       for g in row['groups'] for h in g.get('heads',[])]
            for mi,m in enumerate(ms):
                seen_note=False;inline=False
                for element in m:
                    if element.tag=='note':seen_note=True
                    elif element.tag=='attributes' and seen_note:inline=True
                if inline or any(e['grace'] for e in rec[m]):continue
                events=[e for e in rec[m] if e['staff']==sn and e['midi'] is not None and not e['grace']]
                if not events or len(events)>128 or len({e['voice'] for e in events})>2:
                    continue
                groups=[]
                for t in sorted({e['onset'] for e in events}):
                    groups.append(sorted([e['node'] for e in events if e['onset']==t],
                                         key=lambda n:(int(n.findtext('pitch/octave')),'CDEFGAB'.index(n.findtext('pitch/step')))))
                other=sorted([g for g in row['groups'] if bounds[mi]<g['x']<bounds[mi+1]
                              and not g.get('grace') and g.get('heads')],key=lambda g:g['x'])
                if len(groups)>64 or len(other)>64 or (not local_alignment and len(groups)!=len(other)):
                    continue
                # Exact ordered spelling/cardinality prevents shifted association.
                hs=[sorted(g['heads'],key=lambda h:(h['pitch'][1],'CDEFGAB'.index(h['pitch'][0]))) for g in other]
                state={}
                written=[tuple((n.findtext('pitch/step'),int(n.findtext('pitch/octave'))) for n in ns) for ns in groups]
                detected=[tuple((h['pitch'][0],h['pitch'][1]) for h in hh) for hh in hs]
                complete_alignment=written==detected
                if local_alignment:
                    from .relation_sequence import forced_pairs
                    pairs=forced_pairs(written,detected)
                else:pairs=list(zip(range(len(groups)),range(len(hs))))
                previous=(-1,-1)
                for ni,hi in pairs:
                    ns,hh=groups[ni],hs[hi]
                    # A skipped or ambiguous column may contain a reset sign.
                    # Never propagate state across that unobserved interval.
                    if local_alignment and (ni!=previous[0]+1 or hi!=previous[1]+1):state.clear()
                    previous=(ni,hi)
                    if [(n.findtext('pitch/step'),int(n.findtext('pitch/octave'))) for n in ns] != [(h['pitch'][0],h['pitch'][1]) for h in hh]:
                        for n in ns:
                            state[(n.findtext('pitch/step'),int(n.findtext('pitch/octave')))]=None
                        continue
                    updates={}
                    # The positional zipper cannot assign two different
                    # signs to identical written pitches. Preserve recognized
                    # chromatic unisons and invalidate scalar carry state;
                    # the later exclusive glyph/head decoder handles them.
                    contrasts={identity for identity in {(n.findtext('pitch/step'),int(n.findtext('pitch/octave'))) for n in ns} if len({int(n.findtext('pitch/alter','0')) for n in ns if (n.findtext('pitch/step'),int(n.findtext('pitch/octave')))==identity})>1}
                    for n,h in zip(ns,hh):
                        identity=(n.findtext('pitch/step'),int(n.findtext('pitch/octave')))
                        if identity in contrasts:updates[identity]=None;continue
                        p=n.find('pitch');old=int(p.findtext('alter','0'));acc=attached.get(h['id'])
                        # Signature glyphs occupy the opening prefix. Ambiguous
                        # first-column accidentals cannot seed measure state.
                        if acc and mi==0 and acc['xy'][0]<other[0]['x']-.9*space:
                            acc=None
                        if (n.find('tie') is not None or n.find('notations/tied') is not None
                                or n.get('print-object')=='no' or h['confidence']<.85
                                or n.findtext('accidental') in ('double-sharp','flat-flat')):
                            updates[identity]=None;continue
                        if ink is None:
                            raw=cv2.imread(str(photo),cv2.IMREAD_GRAYSCALE)
                            if raw is None or raw.size>16000000 or max(raw.shape)>8192:return []
                            ink=cv2.threshold(cv2.resize(raw,None,fx=FACTOR,fy=FACTOR),140,255,cv2.THRESH_BINARY_INV)[1]
                        head={'x':h['xy'][0],'y':h['xy'][1],'staff_space':space}
                        absent=no_accidental(head,all_heads,ink)
                        if acc:
                            value,label=NAMES[acc['class']];updates[identity]=value
                            method='unique_visual_accidental'
                            # An incomplete graph can omit the glyph's real
                            # target. New local anchors may seed only when the
                            # existing note agrees, never overwrite on one cue.
                            if local_alignment and not complete_alignment and value!=old:
                                updates[identity]=None;continue
                        elif absent and identity in state and state[identity] is not None and n.find('accidental') is None:
                            value,label=state[identity],None;method='bar_local_accidental_state'
                        else:
                            if not absent:updates[identity]=None
                            continue
                        if value!=old:
                            alter=p.find('alter')
                            if alter is None:
                                alter=E.Element('alter');p.insert(list(p).index(p.find('step'))+1,alter)
                            alter.text=str(value)
                            if label:
                                glyph=n.find('accidental')
                                if glyph is None:
                                    glyph=E.Element('accidental');at=list(n).index(n.find('type'))+1 if n.find('type') is not None else len(n);n.insert(at,glyph)
                                glyph.text=label
                            changes.append({'measure':m.get('number'),'system':si,'staff':sn,'pitch':identity,
                                            'old_alter':old,'new_alter':value,'xy':h['xy'],'method':method})
                    state.update(updates)
    return changes
