"""Preserve decoded meter changes when using the legacy XML exporter."""
import xml.etree.ElementTree as E
import copy
from experimental_omr.export import to_musicxml


def apply_meters(root,meters):
    for part in root.findall('part'):
        measures=part.findall('measure')
        if len(measures)!=len(meters):raise ValueError('Decoded meter/measure alignment mismatch')
        previous=None
        for measure,meter in zip(measures,meters):
            meter=tuple(meter)
            if meter!=previous:
                attr=measure.find('attributes')
                if attr is None:attr=E.Element('attributes');measure.insert(0,attr)
                for time in list(attr.findall('time')):attr.remove(time)
                time=E.Element('time');E.SubElement(time,'beats').text=str(meter[0]);E.SubElement(time,'beat-type').text=str(meter[1])
                # Attributes order: divisions, key, time, staves, clef.
                position=sum(child.tag in ('divisions','key') for child in attr);attr.insert(position,time)
            previous=meter
    return root


def export_score(score):
    ordinary=copy.deepcopy(score)
    for part in ordinary.parts:
        for staff in part.staves:
            for measure in staff.measures:
                for voice in measure.voices:voice.events=[ev for ev in voice.events if not getattr(ev,'grace',False)]
    root=E.fromstring(to_musicxml(ordinary))
    for pi,part in enumerate(score.parts):
        selected=E.Element('score-partwise');selected.append(root.findall('part')[pi])
        apply_meters(selected,[m.time_signature for m in part.staves[0].measures])
    if any(getattr(m,'relation_export',False) for p in score.parts for s in p.staves for m in s.measures):
        export_relations(root,score)
    E.indent(root,space='  ')
    return '<?xml version="1.0" encoding="UTF-8"?>\n'+E.tostring(root,encoding='unicode')+'\n'


def clef_xml(attr,number,clef,nstaves):
    node=E.SubElement(attr,'clef',{'number':str(number)} if nstaves>1 else {})
    E.SubElement(node,'sign').text=clef['sign'];E.SubElement(node,'line').text=str(clef['line'])
    if clef.get('octave'):E.SubElement(node,'clef-octave-change').text=str(clef['octave'])


def physical_beam_marks(events):
    """Export observed beam chains instead of inventing beat-based beams."""
    from collections import defaultdict
    groups=defaultdict(list);previous=None
    for event in sorted(events,key=lambda e:e.onset_ticks):
        chord=previous is not None and event.chord_id is not None and event.chord_id==previous.chord_id and event.onset_ticks==previous.onset_ticks and getattr(event,'grace',False)==getattr(previous,'grace',False)
        chain=getattr(event,'beam_chain',None)
        if event.type=='note' and not chord and chain is not None:groups[chain].append(event)
        previous=event
    marks={}
    for group in groups.values():
        if len(group)<2:continue
        for level in (1,2,3):
            eligible=[(e.visual_tail_beams or 0)>=level for e in group]
            for i,event in enumerate(group):
                if not eligible[i]:continue
                before=i>0 and eligible[i-1];after=i+1<len(group) and eligible[i+1]
                value='continue' if before and after else 'begin' if after else 'end' if before else 'backward hook' if i==len(group)-1 else 'forward hook'
                marks.setdefault(id(event),[]).append((level,value))
    return marks


def export_relations(root,score):
    from experimental_omr.export import _event_xml
    for xml_part,part in zip(root.findall('part'),score.parts):
        previous={}
        for mi,xml_measure in enumerate(xml_part.findall('measure')):
            measures=[s.measures[mi] for s in part.staves];nstaves=len(part.staves)
            for node in list(xml_measure):
                if node.tag in ('note','backup','forward'):xml_measure.remove(node)
            attr=xml_measure.find('attributes')
            initials={s.number:getattr(m,'initial_clef',None) for s,m in zip(part.staves,measures)}
            for number,clef in initials.items():
                if clef is None:continue
                if previous.get(number)!=clef:
                    if attr is None:attr=E.Element('attributes');xml_measure.insert(0,attr)
                    for c in list(attr.findall('clef')):
                        if int(c.get('number','1'))==number:attr.remove(c)
                    clef_xml(attr,number,clef,nstaves)
                previous[number]=clef
            cursor=0
            for staff,measure in zip(part.staves,measures):
                for vi,voice in enumerate(measure.voices):
                    if cursor:
                        E.SubElement(E.SubElement(xml_measure,'backup'),'duration').text=str(cursor);cursor=0
                    changes=iter(sorted(getattr(measure,'context_events',[]) if vi==0 else [],key=lambda c:c['onset']))
                    change=next(changes,None);previous_event=None
                    marks=physical_beam_marks(voice.events)
                    events=sorted(voice.events,key=lambda e:e.onset_ticks)
                    for ei,event in enumerate(events):
                        grace=getattr(event,'grace',False)
                        if event.onset_ticks<0 or (not grace and event.duration_ticks<=0):raise ValueError('Invalid relation event timing')
                        chord=previous_event is not None and event.chord_id is not None and event.chord_id==previous_event.chord_id and event.onset_ticks==previous_event.onset_ticks and grace==getattr(previous_event,'grace',False)
                        while change is not None and change['onset']<=event.onset_ticks:
                            if change['onset']!=cursor:
                                tag='forward' if change['onset']>cursor else 'backup';E.SubElement(E.SubElement(xml_measure,tag),'duration').text=str(abs(change['onset']-cursor));cursor=change['onset']
                            a=E.SubElement(xml_measure,'attributes');clef_xml(a,staff.number,change['clef'],nstaves);previous[staff.number]=change['clef'];change=next(changes,None)
                        if not chord and event.onset_ticks!=cursor:
                            tag='forward' if event.onset_ticks>cursor else 'backup';E.SubElement(E.SubElement(xml_measure,tag),'duration').text=str(abs(event.onset_ticks-cursor));cursor=event.onset_ticks
                        printed=copy.copy(event)
                        if grace:printed.duration_ticks=getattr(event,'written_duration',240)
                        node=E.fromstring(_event_xml(printed,staff.number,voice.id,nstaves,chord,marks.get(id(event),[]),f'rP{xml_part.get("id")}M{mi+1}S{staff.number}V{voice.id}E{ei}'))
                        ratio=getattr(event,'time_modification',None)
                        if ratio and not grace:
                            from experimental_omr.export import _split_type_and_dots
                            written,dots=_split_type_and_dots(event.written_duration)
                            node.find('type').text=written
                            for child in list(node.findall('dot'))+list(node.findall('time-modification')):node.remove(child)
                            position=list(node).index(node.find('type'))+1
                            for _ in range(dots):node.insert(position,E.Element('dot'));position+=1
                            tm=E.Element('time-modification')
                            E.SubElement(tm,'actual-notes').text=str(ratio[0]);E.SubElement(tm,'normal-notes').text=str(ratio[1]);E.SubElement(tm,'normal-type').text=written
                            position=next((i for i,n in enumerate(node) if n.tag in ('staff','beam','notations')),len(node));node.insert(position,tm)
                        if grace:
                            node.remove(node.find('duration'));node.insert(0,E.Element('grace',{'slash':'yes'} if getattr(event,'slash',False) else {}))
                        # The XML has new automatic spacing, not the photographed
                        # page's stem lengths. Let the engraver lay out stems;
                        # measured directions remain available in the graph.
                        xml_measure.append(node)
                        if not chord and not grace:cursor=event.onset_ticks+event.duration_ticks
                        previous_event=event
                    while change is not None:
                        if change['onset']!=cursor:
                            tag='forward' if change['onset']>cursor else 'backup';E.SubElement(E.SubElement(xml_measure,tag),'duration').text=str(abs(change['onset']-cursor));cursor=change['onset']
                        a=E.SubElement(xml_measure,'attributes');clef_xml(a,staff.number,change['clef'],nstaves);previous[staff.number]=change['clef'];change=next(changes,None)
