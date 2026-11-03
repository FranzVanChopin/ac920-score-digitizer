"""Find likely private artifacts; review findings instead of silently deleting data."""
import argparse,hashlib,json,re
from pathlib import Path
TEXT={'.md','.txt','.json','.log','.rpt','.csv','.py','.tcl','.yaml','.yml','.xml','.musicxml','.complete'}
RULES={
 'personal_home':re.compile(r'(?i)[A-Z]:[/\\]+Users[/\\]+(?!Public\b|Default\b)[^/\\\r\n]+'),
 'session_identifier':re.compile(r'\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b',re.I),
 'raw_chat_role':re.compile(r'"role"\s*:\s*"(?:assistant|user|system)"'),
 'private_key':re.compile(r'-----BEGIN (?:OPENSSH |RSA |EC )?PRIVATE KEY-----'),
 'literal_credential':re.compile(r'''(?i)["']?(?:password|sudo_password|api_key|access_token)["']?\s*[:=]\s*["'][^"'\r\n]{4,}["']''')
}
def audit(folder):
    out=[];entries=[]
    for p in sorted(Path(folder).rglob('*')):
        if not p.is_file() or '__pycache__' in p.parts or '.git' in p.parts:continue
        rel=p.relative_to(folder).as_posix();raw=p.read_bytes()
        if any(x in p.name.lower() for x in ['visible_messages','visible_excerpts','rollout']) or p.suffix=='.jsonl':out.append(dict(path=rel,rule='raw_transcript_filename'))
        if p.suffix.lower() in TEXT:
            for i,line in enumerate(raw.decode('utf-8-sig',errors='replace').splitlines(),1):
                for rule,regex in RULES.items():
                    if regex.search(line):out.append(dict(path=rel,line=i,rule=rule))
        entries.append(dict(path=rel,bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest()))
    return dict(review_pass=not out,findings=out,files_checked=len(entries),scope='Heuristic privacy check, not a proof of anonymization. No file is changed or deleted.')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('folder',type=Path);a=p.parse_args();r=audit(a.folder)
    print(json.dumps(r,ensure_ascii=False,indent=2));raise SystemExit(0 if r['review_pass'] else 1)
