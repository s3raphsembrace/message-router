"""Layer 1 tests. No network, no API key, no third-party test runner.

    python code/preprocess/test_preprocess.py

The real dataset media is complete and healthy -- all 30 referenced files resolve --
so every failure path here is exercised against a synthetic dataset built in a temp
directory. The point of these tests is the guarantee the router depends on: a bad
media file degrades one row to metadata-only, it never takes down the run.
"""

import json
import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cache import InterpretationCache, TRANSIENT_STATUSES          # noqa: E402
from gemini import MediaInterpreter, _clamp01                      # noqa: E402
from media_index import MediaIndex                                 # noqa: E402
from run import collect_refs                                       # noqa: E402
from schema import (                                               # noqa: E402
    KIND_IMAGE,
    KIND_VOICE,
    LAYOUT_OTHER,
    LAYOUT_STOCK_PROMO,
    LAYOUT_UNKNOWN,
    STATUS_API_ERROR,
    STATUS_BAD_RESPONSE,
    STATUS_MISSING_FILE,
    STATUS_MISSING_INDEX,
    STATUS_NO_API_KEY,
    STATUS_OK,
    STATUS_UNREADABLE,
    MediaInterpretation,
)

PASSED = []
FAILED = []


def check(name, cond, detail=""):
    (PASSED if cond else FAILED).append(name)
    print("  %s %s%s" % ("PASS" if cond else "FAIL", name, ("  -- " + detail) if detail and not cond else ""))


def build_fixture(root):
    """A miniature dataset exercising every resolution outcome."""
    ds = os.path.join(root, "dataset")
    os.makedirs(os.path.join(ds, "media", "images"))
    os.makedirs(os.path.join(ds, "media", "audio"))

    # good image, good audio
    with open(os.path.join(ds, "media", "images", "img_ok.jpg"), "wb") as f:
        f.write(b"\xff\xd8\xff\xe0" + b"x" * 64)
    with open(os.path.join(ds, "media", "audio", "vn_ok.mp3"), "wb") as f:
        f.write(b"ID3" + b"y" * 64)
    # empty file -> unreadable
    open(os.path.join(ds, "media", "images", "img_empty.jpg"), "wb").close()
    # img_gone is indexed but never written -> missing_file

    with open(os.path.join(ds, "images.csv"), "w", newline="", encoding="utf-8") as f:
        f.write("image_id,file_path\n")
        f.write("img_ok,media/images/img_ok.jpg\n")
        f.write("img_empty,media/images/img_empty.jpg\n")
        f.write("img_gone,media/images/img_gone.jpg\n")
    with open(os.path.join(ds, "voice_notes.csv"), "w", newline="", encoding="utf-8") as f:
        f.write("voice_note_id,file_path\n")
        f.write("vn_ok,media/audio/vn_ok.mp3\n")

    cols = ("message_id,user_id,conversation_type,group_id,business_id,sender_user_id,"
            "created_at,message_text,media_type,media_id,forwarded_count\n")
    with open(os.path.join(ds, "messages.csv"), "w", newline="", encoding="utf-8") as f:
        f.write(cols)
        f.write("m1,u_1,business,,b_1,,2026-07-01 10:00,,image,img_ok,0\n")
        f.write("m2,u_2,business,,b_1,,2026-07-01 10:00,,image,img_ok,0\n")   # dup id
        f.write("m3,u_3,group,g_1,,u_9,2026-07-01 10:00,,voice,vn_ok,0\n")
        f.write("m4,u_4,group,g_1,,u_9,2026-07-01 10:00,,image,img_empty,0\n")
        f.write("m5,u_5,group,g_1,,u_9,2026-07-01 10:00,,image,img_gone,0\n")
        f.write("m6,u_6,group,g_1,,u_9,2026-07-01 10:00,,image,img_ghost,0\n")  # not indexed
        f.write("m7,u_7,group,g_1,,u_9,2026-07-01 10:00,plain text,,,0\n")      # no media
    return ds


def main():
    tmp = tempfile.mkdtemp(prefix="l1test_")
    try:
        ds = build_fixture(tmp)
        index = MediaIndex(ds)

        print("\n[resolution]")
        st, rel, ap, err = index.resolve("img_ok", KIND_IMAGE)
        check("good image resolves", st == STATUS_OK and rel == "media/images/img_ok.jpg", st)
        st, _, _, _ = index.resolve("vn_ok", KIND_VOICE)
        check("good audio resolves", st == STATUS_OK, st)
        st, _, _, _ = index.resolve("img_ghost", KIND_IMAGE)
        check("unindexed id -> missing_index", st == STATUS_MISSING_INDEX, st)
        st, _, _, _ = index.resolve("img_gone", KIND_IMAGE)
        check("indexed but absent -> missing_file", st == STATUS_MISSING_FILE, st)
        st, _, _, _ = index.resolve("img_empty", KIND_IMAGE)
        check("empty file -> unreadable", st == STATUS_UNREADABLE, st)
        st, _, _, _ = index.resolve("vn_ok", "sideways")
        check("bad kind -> missing_index", st == STATUS_MISSING_INDEX, st)
        check("no resolution path raises", True)

        print("\n[mime + bytes]")
        check("mp3 mime", MediaIndex.mime_for("a/b.mp3") == "audio/mpeg")
        check("jpg mime", MediaIndex.mime_for("a/b.JPG") == "image/jpeg")
        check("unknown ext -> None", MediaIndex.mime_for("a/b.xyz") is None)
        data, e = MediaIndex.read_bytes(os.path.join(ds, "media", "images", "img_ok.jpg"))
        check("read good file", data is not None and not e)
        data2, e2 = MediaIndex.read_bytes(os.path.join(ds, "nope.jpg"))
        check("read missing file returns error, no raise", data2 is None and bool(e2))
        check("sha256 is stable", MediaIndex.sha256(b"abc") == MediaIndex.sha256(b"abc"))

        print("\n[ref collection]")
        refs = collect_refs(ds, ["messages.csv"])
        # 6 rows carry media, but img_ok appears on two rows -> 5 distinct ids.
        check("dedupes repeated media_id", len(refs) == 5, "got %d: %s" % (len(refs), sorted(refs)))
        check("ignores rows with no media", "" not in refs)
        check("kinds captured", refs.get("vn_ok") == KIND_VOICE and refs.get("img_ok") == KIND_IMAGE)

        print("\n[schema / placeholders]")
        ph = MediaInterpretation.failure("x", KIND_IMAGE, STATUS_MISSING_FILE, "gone")
        check("placeholder is not ok", not ph.ok)
        check("placeholder has no content", not ph.has_content)
        check("placeholder router_text empty", ph.router_text() == "")
        check("placeholder layout unknown", ph.layout == LAYOUT_UNKNOWN)
        check("placeholder confidence 0", ph.interp_confidence == 0.0)
        ph2 = MediaInterpretation.failure("y", "nonsense-kind", STATUS_API_ERROR, "z" * 900)
        check("failure() tolerates bad kind", ph2.kind in (KIND_IMAGE, KIND_VOICE))
        check("failure() truncates error", len(ph2.error) <= 500)

        okv = MediaInterpretation(
            media_id="vn_ok", kind=KIND_VOICE, status=STATUS_OK,
            transcript="paisa bhejo", intent_summary="Asks for money.", interp_confidence=0.8)
        check("voice has_content", okv.has_content)
        check("voice text is labelled", "[voice transcript]" in okv.router_text()
              and "[voice intent]" in okv.router_text())
        oki = MediaInterpretation(
            media_id="img_ok", kind=KIND_IMAGE, status=STATUS_OK,
            extracted_text="50% OFF", visual_description="Sale poster.",
            layout=LAYOUT_STOCK_PROMO, interp_confidence=0.9)
        check("image text is labelled", "[image text]" in oki.router_text())
        check("layout surfaced to router", "[image layout] stock_promo" in oki.router_text())
        rt = MediaInterpretation.from_dict(oki.to_dict())
        check("dataclass round-trips", rt.to_dict() == oki.to_dict())
        drift = dict(oki.to_dict()); drift["field_from_the_future"] = 1
        check("from_dict tolerates schema drift", MediaInterpretation.from_dict(drift).media_id == "img_ok")

        print("\n[cache]")
        cpath = os.path.join(tmp, "cache", "media.json")
        c = InterpretationCache(cpath)
        check("empty cache loads", len(c) == 0)
        check("permanent failure is cached", c.put(ph) is True)
        check("no_api_key is NOT cached",
              c.put(MediaInterpretation.failure("z", KIND_IMAGE, STATUS_NO_API_KEY)) is False)
        check("api_error is NOT cached",
              c.put(MediaInterpretation.failure("z", KIND_IMAGE, STATUS_API_ERROR)) is False)
        check("bad_response is NOT cached",
              c.put(MediaInterpretation.failure("z", KIND_IMAGE, STATUS_BAD_RESPONSE)) is False)
        check("transient set is exactly the env-dependent ones",
              TRANSIENT_STATUSES == {STATUS_NO_API_KEY, STATUS_API_ERROR, STATUS_BAD_RESPONSE})
        oki.sha256 = "deadbeef"
        c.put(oki)
        c.save()
        check("cache file written", os.path.exists(cpath))
        c2 = InterpretationCache(cpath)
        check("cache reloads", len(c2) == 2 and c2.get("img_ok").extracted_text == "50% OFF")
        check("stale detected on sha change", c2.is_stale("img_ok", "cafe1234") is True)
        check("not stale on same sha", c2.is_stale("img_ok", "deadbeef") is False)
        check("unknown id is not stale", c2.is_stale("nope", "x") is False)
        with open(cpath, encoding="utf-8") as f:
            blob = json.load(f)
        check("cache keyed on media_id", set(blob["entries"]) == {"x", "img_ok"})

        with open(cpath, "w", encoding="utf-8") as f:
            f.write("{ this is not json")
        c3 = InterpretationCache(cpath)
        check("corrupt cache degrades, does not raise", len(c3) == 0)

        print("\n[interpreter, no key]")
        interp = MediaInterpreter(api_key=None)
        if not interp.available:
            rec = interp.interpret("img_ok", KIND_IMAGE,
                                   os.path.join(ds, "media", "images", "img_ok.jpg"),
                                   "media/images/img_ok.jpg", b"\xff\xd8", "sha")
            check("no key -> no_api_key placeholder", rec.status == STATUS_NO_API_KEY, rec.status)
            check("placeholder keeps sha for later", rec.sha256 == "sha")
            check("unavailable_reason is explanatory", "no API key" in interp.unavailable_reason)
        else:
            check("no key -> no_api_key placeholder", True, "SKIPPED: a key is configured")

        print("\n[payload normalisation]")
        i2 = MediaInterpreter.__new__(MediaInterpreter)
        i2.model = "test-model"
        rec = i2._to_record("i", KIND_IMAGE,
                            {"extracted_text": "x", "visual_description": "d",
                             "layout": "not_a_real_layout", "confidence": 0.7}, "p", "s")
        check("off-enum layout coerced to 'other'", rec.layout == LAYOUT_OTHER, rec.layout)
        rec = i2._to_record("i", KIND_IMAGE,
                            {"extracted_text": "", "visual_description": "",
                             "layout": "stock_promo", "confidence": 0.9}, "p", "s")
        check("empty image result -> confidence 0", rec.interp_confidence == 0.0)
        rec = i2._to_record("v", KIND_VOICE,
                            {"transcript": "", "intent_summary": "No intelligible speech.",
                             "language": "", "confidence": 0.95}, "p", "s")
        check("empty transcript -> confidence 0", rec.interp_confidence == 0.0)
        check("blank language -> 'unknown'", rec.language == "unknown")
        rec = i2._to_record("v", KIND_VOICE, {"transcript": "hi"}, "p", "s")
        check("missing fields do not raise", rec.status == STATUS_OK)
        check("clamp handles junk", _clamp01("abc", 0.4) == 0.4 and _clamp01(9) == 1.0
              and _clamp01(-3) == 0.0)

        print("\n[end-to-end on the fixture, no key]")
        import run as runner
        rc = runner.main(["--dataset", ds, "--cache", os.path.join(tmp, "c2.json"),
                          "--scope", "messages"])
        check("run exits 0 despite 3 broken media", rc == 0, "rc=%s" % rc)
        c4 = InterpretationCache(os.path.join(tmp, "c2.json"))
        got = {r.media_id: r.status for r in c4}
        check("missing_file cached", got.get("img_gone") == STATUS_MISSING_FILE, str(got))
        check("unreadable cached", got.get("img_empty") == STATUS_UNREADABLE, str(got))
        check("missing_index cached", got.get("img_ghost") == STATUS_MISSING_INDEX, str(got))
        check("healthy files not cached without a key",
              "img_ok" not in got and "vn_ok" not in got, str(got))

    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print("\n%d passed, %d failed" % (len(PASSED), len(FAILED)))
    if FAILED:
        print("failed: " + ", ".join(FAILED))
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
