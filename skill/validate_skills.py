"""Offline behavioral checks using bundled reports/XML and optional external XSA."""
from pathlib import Path
import argparse,ast,importlib.util,json,re,shutil,tempfile,zipfile
from fractions import Fraction
import xml.etree.ElementTree as ET

BASE=Path(__file__).resolve().parent
ROOT=BASE.parent
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
def require_error(fn):
    try:fn()
    except ValueError:return
    raise AssertionError('Expected invalid input to be rejected')
def run(xsa=None):
    checks=[]
    folders=sorted(p for p in BASE.iterdir() if p.is_dir() and (p/'SKILL.md').is_file())
    for p in folders:
        s=(p/'SKILL.md').read_text(encoding='utf-8')
        assert s.startswith('---\n') and f'name: {p.name}\n' in s
        assert re.fullmatch('[a-z0-9]+(?:-[a-z0-9]+)*',p.name)
        assert len(s.split('---',2))==3
    for p in BASE.rglob('*.py'):ast.parse(p.read_bytes(),filename=p.name)
    checks.append(dict(name='skill_format_and_python_syntax',passed=True,skills=len(folders)))
    fp=load('fpga',BASE/'fpga-report-review/scripts/review_reports.py')
    result=fp.review(ROOT/'build/reports')
    assert result['final_review_pass']
    assert result['implementation']['timing']['wns_ns']==.817
    assert result['implementation']['drc']['counts']['Warning']==8
    assert result['synthesis']['timing']['hold_failures']>0
    require_error(lambda:fp.timing('No timing data available'))
    assert fp.drc('| INBB-1 | Critical Warning | Black Box Instances | 15 |')['counts']['Critical Warning']==15
    checks.append(dict(name='fpga_real_reports_and_invalid_summary',passed=True,wns_ns=.817,drc_warnings=8,synthesis_and_implementation_separated=True))
    mx=load('musicxml',BASE/'musicxml-regression/scripts/compare_musicxml.py')
    # the recognition evidence lives under board/ after the 2026-11-02 restructure;
    # keep the other two candidates so this also runs in the original pack layout.
    sample=next((p for p in (
        ROOT/'board/recognition/card_20261103/Outputs/01.musicxml',
        ROOT/'recognition/card_20261103/Outputs/01.musicxml',
        ROOT/'data/samples/result/01.musicxml',
    ) if p.is_file()),None)
    assert sample is not None,'sample MusicXML not found; expected board/recognition/card_20261103/Outputs/01.musicxml'
    assert mx.compare(sample,sample)['semantic_events_equal']
    with tempfile.TemporaryDirectory() as tmp:
        tmp=Path(tmp)
        broken_reports=tmp/'reports';broken_reports.mkdir()
        for p in (ROOT/'build/reports').glob('*.rpt'):shutil.copy2(p,broken_reports/p.name)
        (broken_reports/'synthesis_drc.rpt').write_text('| INBB-1 | Critical Warning | Black Box Instances | 15 |')
        assert not fp.review(broken_reports)['final_review_pass']
        root=ET.fromstring(sample.read_bytes());reformatted=tmp/'reformatted.musicxml'
        ET.indent(root);reformatted.write_bytes(ET.tostring(root))
        assert mx.compare(sample,reformatted)['semantic_events_equal']
        step=root.find('.//pitch/step');step.text='A' if step.text!='A' else 'B'
        changed=tmp/'changed.musicxml';changed.write_bytes(ET.tostring(root))
        delta=mx.compare(sample,changed);assert delta['lost_events']>=1 and delta['added_events']>=1
        def synthetic(div):
            return f'<score-partwise><part id="P1"><measure><attributes><divisions>{div}</divisions></attributes><note><pitch><step>C</step><octave>4</octave></pitch><duration>{div}</duration><voice>1</voice></note><note><chord/><pitch><step>E</step><octave>4</octave></pitch><duration>{div}</duration><voice>1</voice></note><backup><duration>{div}</duration></backup><note><rest/><duration>{div}</duration><voice>2</voice><staff>2</staff></note></measure></part></score-partwise>'
        a,b=tmp/'a.musicxml',tmp/'b.musicxml';a.write_text(synthetic(1));b.write_text(synthetic(2))
        assert mx.compare(a,b)['semantic_events_equal'] and sum(mx.events(a).values())==3
        x=ET.fromstring(synthetic(1));note=ET.fromstring('<note><chord/><pitch><step>E</step><octave>4</octave></pitch><duration>1</duration><voice>1</voice></note>');x.find('part/measure').insert(3,note);b.write_bytes(ET.tostring(x))
        assert mx.compare(a,b)['added_events']==1
        checks.append(dict(name='musicxml_real_output_and_semantic_mutations',passed=True,real_score_events=sum(mx.events(sample).values()),cases=['format_only','changed_pitch','equivalent_divisions','chord_backup_voice','added_event']))
        py=load('overlay',BASE/'pynq-artifact-check/scripts/check_artifacts.py')
        pair=tmp/'overlay';pair.mkdir();bit=pair/'overlay.bit';hwh=pair/'overlay.hwh'
        if xsa:
            with zipfile.ZipFile(xsa) as z:
                hn=next(n for n in z.namelist() if n.endswith('.hwh') and '/' not in n and z.namelist().count(n)==1 and n.removesuffix('.hwh')+'.bit' in z.namelist())
                bit.write_bytes(z.read(hn.removesuffix('.hwh')+'.bit'));hwh.write_bytes(z.read(hn))
            evidence='real Vivado XSA pair'
        else:
            bit.write_bytes(b'non-empty synthetic presence fixture; not a valid FPGA bitstream')
            hwh.write_text('<EDKSYSTEM><SYSTEMINFO DEVICE="example-device"/><MEMRANGE BASEVALUE="0x1000" HIGHVALUE="0x1fff" INSTANCE="peripheral" MEMTYPE="REGISTER"/></EDKSYSTEM>')
            evidence='synthetic metadata/presence fixture; not a valid bitstream'
        info=py.inspect(bit,hwh);assert info['address_ranges'] and not info['hardware_tested']
        require_error(lambda:py.inspect(bit,hwh,'incorrect-device'))
        other=pair/'wrong.hwh';other.write_bytes(hwh.read_bytes());require_error(lambda:py.inspect(bit,other))
        checks.append(dict(name='pynq_offline_pair_and_failure_cases',passed=True,evidence=evidence,metadata_ranges=len(info['address_ranges']),hardware_tested=False,generation_match_proven=False))
        beat=load('beat',BASE/'beat-timeline/scripts/beat_frames.py')
        t=beat.timeline(90,30,84)
        assert t['last_frame']==1680 and all(x['frame']==x['beat']*20 for x in t['anchors'])
        t=beat.timeline(90,'30000/1001',1000,17)
        assert all(abs(Fraction(x['rounding_error_frames']))<=Fraction(1,2) for x in t['anchors'])
        checks.append(dict(name='beat_exact_and_fractional_fps',passed=True,normal_example_frames=1680,max_rounding_error_frames='1/2'))
        privacy=load('privacy',BASE/'private-release/scripts/audit_release.py')
        safe=tmp/'safe';safe.mkdir();(safe/'summary.md').write_text('Generic technical methods and evidence only.')
        assert privacy.audit(safe)['review_pass']
        (safe/'visible_messages.json').write_text(json.dumps({'role':'assistant','content':'synthetic test'}))
        found=privacy.audit(safe);assert not found['review_pass']
        assert {x['rule'] for x in found['findings']} >= {'raw_chat_role','raw_transcript_filename'}
        checks.append(dict(name='privacy_safe_and_raw_transcript_fixtures',passed=True))
    result=dict(all_checks_passed=True,checks=checks,scope='New offline skill behavior checks, not full historical algorithm regression, PYNQ deployment, tool installation or live-board/audio tests.',template_review=dict(skill='multi-project-handoff',method='Retrospective project contract review',external_team_reuse_tested=False))
    return result
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--xsa',type=Path);a=p.parse_args();r=run(a.xsa)
    (BASE/'VALIDATION.json').write_text(json.dumps(r,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(r,ensure_ascii=False,indent=2))
