"""Print only the *structure* of the provided XML files (tag paths, attribute names, channel
definitions, value vocabularies). No raw data rows are printed, by design: the T&C forbid sharing
extracts of the data with a hosted LLM, so everything that leaves this script is schema/aggregate.
"""
import sys, collections
import xml.etree.ElementTree as ET
sys.stdout.reconfigure(encoding="utf-8")

def schema(path, max_vals=8):
    root = ET.parse(path).getroot()
    paths = collections.Counter()
    attrs = collections.defaultdict(set)
    def walk(e, p):
        tag = e.tag.split('}')[-1]
        q = p + "/" + tag
        paths[q] += 1
        for k in e.attrib:
            attrs[q].add(k)
        for c in e:
            walk(c, q)
    walk(root, "")
    print("=" * 80); print(path)
    for q, n in paths.items():
        print("%6d  %s  attrs=%s" % (n, q, sorted(attrs[q])))

if __name__ == "__main__":
    for p in sys.argv[1:]:
        schema(p)
