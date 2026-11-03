"""Join the page MusicXML files of one piece (run_page.py output: one part, measures numbered from 1 per page)
into one score.

* each page gets notation for display (engrave.py: clefs and key signature from the headers.json next to its
  page.musicxml, written accidentals, note types, beams); --plain leaves the pages as recognized;
* measures are renumbered 1..n across the pages;
* a page's <attributes> keep only what changes the running state (divisions, staves, key, time, clef per staff),
  an <attributes> left empty is dropped;
* every later page starts with <print new-page="yes"/>, so the page turns stay visible in a notation program.

  python merge_pages.py OUT.musicxml PAGE1.musicxml PAGE2.musicxml ... [--title TITLE] [--plain]
"""
import copy
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
try:
    import engrave as EN  # noqa: E402
except ImportError:          # a copy of this file without engrave.py next to it: join the pages as recognized
    EN = None
    print("merge_pages: engrave.py not found next to merge_pages.py; songs are written without notation",
          file=sys.stderr, flush=True)


def _state_key(el):
    """identity of an <attributes> child in the running state (clef / key / time per staff number)"""
    return el.tag, el.get("number")


def _same(a, b):
    return ET.tostring(a) == ET.tostring(b)


def merge(pages, title=None, engrave=True):
    """pages: MusicXML paths in page order -> (ElementTree of the joined score, measures per page)"""
    engrave = engrave and EN is not None
    root, part, state, counts = None, None, {}, []
    n = 0
    for k, path in enumerate(pages):
        page = ET.parse(path).getroot()
        src = page.find("part")
        headers = Path(path).with_name("headers.json")
        if engrave and src is not None and headers.is_file():
            EN.engrave_page(src, json.loads(headers.read_text(encoding="utf-8")))
        if root is None:
            root = ET.Element("score-partwise", page.attrib)
            if title:
                work = ET.SubElement(root, "work")
                ET.SubElement(work, "work-title").text = title
            root.append(copy.deepcopy(page.find("part-list")))
            if engrave:
                EN.name_part(root)
            part = ET.SubElement(root, "part", src.attrib)
        measures = src.findall("measure") if src is not None else []
        counts.append(len(measures))
        for j, m in enumerate(measures):
            m = copy.deepcopy(m)
            n += 1
            m.set("number", str(n))
            for att in m.findall("attributes"):
                for child in list(att):
                    key = _state_key(child)
                    if key in state and _same(state[key], child):
                        att.remove(child)
                    else:
                        state[key] = child
                if len(att) == 0:
                    m.remove(att)
            if k > 0 and j == 0:
                m.insert(0, ET.Element("print", {"new-page": "yes"}))
            part.append(m)
    if root is None:
        raise ValueError("no pages")
    return ET.ElementTree(root), counts


def write(tree, out):
    """atomic: a reader never sees a half-written file"""
    out = Path(out)
    tmp = out.with_name(out.name + ".tmp")
    tree.write(tmp, encoding="utf-8", xml_declaration=True)
    tmp.replace(out)


if __name__ == "__main__":
    args = sys.argv[1:]
    title = None
    plain = "--plain" in args
    if plain:
        args.remove("--plain")
    if "--title" in args:
        i = args.index("--title")
        title = args[i + 1]
        del args[i:i + 2]
    tree, counts = merge(args[1:], title, engrave=not plain)
    write(tree, args[0])
    print(f"{args[0]}: {sum(counts)} measures from {len(counts)} pages {counts}")
