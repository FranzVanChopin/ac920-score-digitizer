"""Local visual attributes and complete predicted attribute state.

Only predicted headers and photographed ink are inputs. Reference scores,
engraving transforms and the expected measure count are not accepted.
"""
import copy
import xml.etree.ElementTree as E
import cv2
import numpy as np
from omr_bar_geometry import measure_systems
from repair_prefix_symbols import detect
from experimental_omr.staff import StaffGeometry
from experimental_omr.inline_clef import find_inline_bass_clefs

NAMES={('G',2):'treble',('F',4):'bass',('C',3):'alto',('C',4):'tenor'}


def key_signature_nodes(nodes,changes,measure_index,first_note_x,space,staff=None,clef=None):
    """Certify the ordered key glyphs against the actual five-line track."""
    changes=[c for c in changes if c.get('measure')==measure_index and 'key' in c
             and c.get('evidence')=='predicted_attributes_aligned_to_image_bars']
    if len(changes)!=1 or staff is None or clef is None:return set()
    change=changes[0];fifths=int(change['key'])
    cls=9 if fifths<0 else 8 if fifths>0 else None
    count=abs(fifths)
    if cls is None or count>7:return set()
    glyphs=sorted([n for n in nodes if n['class']==cls
                  and 0<(n['xy'][0]-change['x'])/space<=1.5*count
                  and n['xy'][0]<first_note_x],key=lambda n:n['xy'][0])
    if len(glyphs)!=count or any(n.get('confidence',0)<.75 for n in glyphs):return set()
    from .staff_preprocess import staff_step
    expected=('BEADGCF' if fifths<0 else 'FCGDAEB')[:count]
    for n,letter in zip(glyphs,expected):
        x,y=n['xy'];position=staff_step(staff,x,y+(.75*space if cls==9 else 0))
        if abs(position['residual'])>.3 or pitch_for_step(position['nearest_step'],clef)[0]!=letter:
            return set()
    return {n['id'] for n in glyphs}


def attributes(attr):
    result={}
    if attr.find('time') is not None:
        result['meter']=(int(attr.findtext('time/beats')),int(attr.findtext('time/beat-type')))
    if attr.findtext('key/fifths') is not None:result['key']=int(attr.findtext('key/fifths'))
    clefs={}
    for c in attr.findall('clef'):
        sign=c.findtext('sign');line=int(c.findtext('line','2'));octave=int(c.findtext('clef-octave-change','0'))
        if sign in ('G','F','C'):clefs[int(c.get('number','1'))]={'sign':sign,'line':line,'octave':octave,'name':NAMES.get((sign,line),'treble')}
    if clefs:result['clefs']=clefs
    return result


def update(state,change):
    for key in ('meter','key'):
        if key in change:state[key]=change[key]
    state['clefs'].update(change.get('clefs',{}))


def contexts(path,nsystems):
    systems=measure_systems(E.parse(path).getroot())
    if len(systems)!=nsystems:raise ValueError('Predicted attribute/system alignment unavailable')
    state={'clefs':{1:{'sign':'G','line':2,'octave':0,'name':'treble'},2:{'sign':'F','line':4,'octave':0,'name':'bass'}},'meter':(4,4),'key':0};rows=[]
    for system in systems:
        first=copy.deepcopy(state);changes=[]
        for mi,measure in enumerate(system):
            for attr in measure.findall('attributes'):
                change=attributes(attr)
                if mi==0:update(first,change)
                else:changes.append({'measure':mi,**change})
                update(state,change)
        rows.append({**first,'predicted_changes':changes,'predicted_measures':len(system)})
    return rows


def flatten(image,staff):
    left=max(0,round(staff['lines'][0]['start'][0]));right=min(image.shape[1],round(staff['lines'][0]['end'][0])+1)
    xs=np.arange(left,right,dtype='float32');ys=np.arange(192,dtype='float32')
    top=np.interp(xs,*np.array(staff['lines'][0]['polyline']).T);bottom=np.interp(xs,*np.array(staff['lines'][-1]['polyline']).T)
    localspace=(bottom-top)/4
    yy=(top[None]+(ys[:,None]-64)/16*localspace[None]).astype('float32');xx=np.broadcast_to(xs,yy.shape).copy()
    return cv2.remap(image,xx,yy,cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=255),left


def visual_changes(image,staff,bounds,ctx,number):
    flat,left=flatten(image,staff);space=16.
    ink=(flat<140).astype('uint8')
    horizontal=cv2.morphologyEx(ink,cv2.MORPH_OPEN,np.ones((1,48),np.uint8))
    clean=cv2.morphologyEx(cv2.subtract(ink,horizontal),cv2.MORPH_CLOSE,np.ones((3,1),np.uint8))
    geom=StaffGeometry(0,[64.,80.,96.,112.,128.],16.,0.,float(flat.shape[1]-1))
    bass=find_inline_bass_clefs(clean,geom)
    # Adapt the existing generic glyph reader to a single flattened staff.
    prefix={'systems':[{'staffs':[{'x_range':[0,flat.shape[1]-1],'grid':[{'x':0,'y':geom.line_ys}]}]}]}
    meter=detect(prefix,flat,full_width=True)[0]['meters']
    changes=[]
    for hit in meter:
        x=hit['x']+left;mi=next((i for i,(a,b) in enumerate(zip(bounds,bounds[1:])) if a<x<b),None)
        if mi is None:continue
        # A printed meter belongs to the beginning of its measure. Require a
        # nearby learned physical bar except at the opening system prefix.
        if mi and x-bounds[mi]>4*staff['space_px']:continue
        changes.append({'x':x,'measure':mi,'meter':(hit['beats'],hit['beat_type']),'confidence':hit['confidence'],'evidence':'generic_meter_glyphs'})
    for x in bass:
        x+=left;mi=next((i for i,(a,b) in enumerate(zip(bounds,bounds[1:])) if a<x<b),None)
        if mi is None:continue
        changes.append({'x':x,'measure':mi,'clefs':{number:{'sign':'F','line':4,'octave':0,'name':'bass'}},'evidence':'F_clef_body_and_two_dots'})
    # Predicted attributes can fill key changes and other clefs only when the
    # system's independently detected physical bars agree with its structure.
    # Mid-measure clefs require a visual position, never a fabricated onset.
    if len(bounds)-1==ctx['predicted_measures']:
        for change in ctx['predicted_changes']:
            mi=change['measure'];item={'x':bounds[mi]+.01,'measure':mi,'evidence':'predicted_attributes_aligned_to_image_bars'}
            for key in ('key','meter'):
                if key in change:item[key]=change[key]
            if number in change.get('clefs',{}):
                clef=change['clefs'][number]
                if clef['name']=='bass':
                    proofs=[c for c in changes if c['measure']==mi and c.get('clefs',{}).get(number,{}).get('name')=='bass']
                    if proofs:item['clefs']={number:clef};item['x']=proofs[0]['x']
                # Other inline clefs are retained as a predicted dependency,
                # with its scope explicitly reported rather than called visual.
                else:item['clefs']={number:clef}
            if len(item)>3:changes.append(item)
    return sorted(changes,key=lambda r:r['x'])


def pitch_for_step(step,clef):
    letters='CDEFGAB';letter,octave={'G':('G',4),'F':('F',3),'C':('C',4)}[clef['sign']]
    d=octave*7+letters.index(letter)-2*(clef['line']-1)+step
    return letters[d%7],d//7+clef.get('octave',0)
