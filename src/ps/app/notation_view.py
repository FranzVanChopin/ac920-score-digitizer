"""Complete optional engraving fields in a COPY of compact recognizer MusicXML.

Pitch, duration, voice, staff, backup/forward, meter and the original file stay
unchanged. Verovio needs explicit types to render/play the compact duration-only
exports correctly. Missing clefs use the project's standard piano grand staff.
"""
import math
import xml.etree.ElementTree as ET
from fractions import Fraction

VALUES=[('maxima',32),('long',16),('breve',8),('whole',4),('half',2),('quarter',1),
        ('eighth',Fraction(1,2)),('16th',Fraction(1,4)),('32nd',Fraction(1,8)),
        ('64th',Fraction(1,16)),('128th',Fraction(1,32)),('256th',Fraction(1,64)),
        ('512th',Fraction(1,128)),('1024th',Fraction(1,256))]


def duration_type(value):
    for name,base in VALUES:
        for dots in range(4):
            if Fraction(base)*(2-Fraction(1,2**dots))==value:return name,dots,None
    # Duration-only exports encode common triplets as thirds of a quarter.
    for name,base in VALUES:
        for dots in range(3):
            if Fraction(base)*(2-Fraction(1,2**dots))*Fraction(2,3)==value:
                return name,dots,(3,2)
    # An unusual rational value is still represented exactly as a tuplet.
    name,base=min(VALUES,key=lambda x:abs(math.log2(float(value/Fraction(x[1])))))
    ratio=Fraction(base)/value
    return name,0,(ratio.numerator,ratio.denominator)


def prepare(data):
    root=ET.fromstring(data)
    changes=dict(types=0,clefs=0)
    for part in root.findall('part'):
        divisions=1
        measures=part.findall('measure')
        if not measures:continue
        first=measures[0]
        attributes=first.find('attributes')
        if attributes is None:attributes=ET.Element('attributes');first.insert(0,attributes)
        staves=int(attributes.findtext('staves','1'))
        for staff in range(1,staves+1):
            if any((int(c.get('number','1'))==staff) for c in attributes.findall('clef')):continue
            # The recognition project's two-staff piano export omits clefs.
            clef=ET.SubElement(attributes,'clef',{'number':str(staff)})
            ET.SubElement(clef,'sign').text='F' if staves==2 and staff==2 else 'G'
            ET.SubElement(clef,'line').text='4' if staves==2 and staff==2 else '2'
            changes['clefs']+=1
        for measure in measures:
            for child in measure:
                if child.tag=='attributes':
                    divisions=int(child.findtext('divisions',str(divisions)))
                if child.tag!='note' or child.find('grace') is not None:continue
                if (child.findtext('type') or '').strip():continue
                rest=child.find('rest')
                if rest is not None and rest.get('measure')=='yes':continue
                value=Fraction(child.findtext('duration','0'))/divisions
                if value<=0:continue
                modification=child.find('time-modification')
                actual=normal=None
                if modification is not None:
                    actual=int(modification.findtext('actual-notes','1'))
                    normal=int(modification.findtext('normal-notes','1'))
                    value*=Fraction(actual,normal)
                name,dots,tuplet=duration_type(value)
                for element in list(child):
                    if element.tag in ('type','dot'):child.remove(element)
                before=next((i for i,e in enumerate(child) if e.tag in
                    ('accidental','time-modification','stem','notehead','staff','beam','notations','lyric')),len(child))
                element=ET.Element('type');element.text=name;child.insert(before,element)
                for n in range(dots):child.insert(before+1+n,ET.Element('dot'))
                if tuplet and modification is None:
                    modification=ET.Element('time-modification')
                    ET.SubElement(modification,'actual-notes').text=str(tuplet[0])
                    ET.SubElement(modification,'normal-notes').text=str(tuplet[1])
                    child.insert(before+1+dots,modification)
                changes['types']+=1
    return ET.tostring(root,encoding='unicode'),changes
