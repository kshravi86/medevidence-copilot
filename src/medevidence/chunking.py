"""Three chunking strategies, all returning [(pmid, chunk_text)].

abstract : one chunk per abstract (bge-small truncates at 512 tokens; median abstract ~200 words)
section  : one chunk per structured-abstract section ("BACKGROUND: ...")
window   : sliding word window (default 100 words, stride 50)
"""


def chunk_abstract(docs):
    return [(d["pmid"], d["text"]) for d in docs]


def chunk_section(docs):
    out = []
    for d in docs:
        labs = d["labels"] if len(d["labels"]) == len(d["sections"]) else [""] * len(d["sections"])
        for l, s in zip(labs, d["sections"]):
            out.append((d["pmid"], f"{l}: {s}" if l else s))
    return out


def chunk_window(docs, size=100, stride=50):
    out = []
    for d in docs:
        w = d["text"].split()
        if len(w) <= size:
            out.append((d["pmid"], d["text"]))
            continue
        i = 0
        while True:
            out.append((d["pmid"], " ".join(w[i:i + size])))
            if i + size >= len(w):
                break
            i += stride
    return out


CHUNKERS = {"abstract": chunk_abstract, "section": chunk_section, "window": chunk_window}
