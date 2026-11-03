"""Fresh MusicXML event records with exact integer clock accumulation.

This is a drop-in equivalent of omr_event_records.records. It keeps live note
identities and fresh output containers, and does not cache XML or records.
Only the intermediate cursor uses integers. Public onsets, durations and keys
keep Fraction values, including division changes inside a measure.
"""
from fractions import Fraction
from math import gcd


def _advance(ticks,denominator,units,divisions):
    if divisions==0:
        # Same exception and arguments as the legacy Fraction constructor.
        Fraction(units,divisions)
    if not units:
        return ticks,denominator
    absolute=abs(divisions)
    if denominator%absolute:
        common=denominator*(absolute//gcd(denominator,absolute))
        ticks*=common//denominator
        denominator=common
    return ticks+units*(denominator//divisions),denominator


def records(root):
    divisions=1;result=[]
    for measure in root.find('part').findall('measure'):
        ticks=0;denominator=1;cursor=Fraction(0);previous=cursor;row=[]
        for node in measure:
            if node.tag=='attributes':
                divisions=int(node.findtext('divisions',str(divisions)))
            elif node.tag in ('backup','forward'):
                units=int(node.findtext('duration','0'))
                # Validate before applying the direction, preserving zero
                # divisor errors even when a backup amount itself is zero.
                if divisions==0:Fraction(units,divisions)
                ticks,denominator=_advance(ticks,denominator,units*(1 if node.tag=='forward' else -1),divisions)
                cursor=None
            elif node.tag=='note':
                grace=node.find('grace') is not None;chord=node.find('chord') is not None
                if chord:
                    onset=previous
                else:
                    if cursor is None:cursor=Fraction(ticks,denominator)
                    onset=cursor
                units=int(node.findtext('duration','0'));dur=Fraction(units,divisions)
                if not chord:
                    previous=onset
                    if not grace:
                        ticks,denominator=_advance(ticks,denominator,units,divisions)
                        cursor=None
                pitch=node.find('pitch');staff=int(node.findtext('staff','1'));voice=node.findtext('voice','1')
                midi=None if pitch is None else (int(pitch.findtext('octave'))+1)*12+{'C':0,'D':2,'E':4,'F':5,'G':7,'A':9,'B':11}[pitch.findtext('step')]+int(pitch.findtext('alter','0'))
                row.append({'node':node,'staff':staff,'voice':voice,'onset':onset,'duration':dur,'midi':midi,'grace':grace,'chord':chord,'key':(staff,onset,midi,dur)})
        result.append(row)
    return result
