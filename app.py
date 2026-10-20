"""Outlook-style Email Spam Classifier.

Run (after running spam_classifier.py once to create spam_model.joblib):
    python -m streamlit run app.py
"""
import html
from datetime import datetime
from pathlib import Path

import joblib
import streamlit as st

from spam_classifier import clean_text

MODEL_PATH = Path(__file__).resolve().parent / "spam_model.joblib"

st.set_page_config(page_title="Mail - Spam Guard", page_icon="📧", layout="wide")

# ---------- Outlook-like styling ----------
st.markdown(
    """
<style>
html, body, [class*="css"] { font-family: "Segoe UI", "Helvetica Neue", Arial, sans-serif; }
#MainMenu, footer, header[data-testid="stHeader"] { visibility: hidden; }
.block-container { padding-top: 0 !important; max-width: 100%; }
.topbar { background:#0F6CBD; color:#fff; padding:10px 22px; margin:0 -4rem 16px -4rem;
          display:flex; align-items:center; gap:14px; }
.topbar .logo { font-size:22px; font-weight:600; }
.topbar .sub  { opacity:.85; font-size:13px; }
[data-testid="stSidebar"] { background:#F3F2F1; border-right:1px solid #E1DFDD; }
.pane-title { font-size:15px; font-weight:600; color:#201F1E; margin:4px 0 8px 0;
              padding-bottom:6px; border-bottom:2px solid #0F6CBD; }
.reader { background:#fff; border:1px solid #E1DFDD; border-radius:6px; padding:20px 24px; }
.reader h3 { margin:0 0 10px 0; color:#201F1E; font-weight:600; }
.meta { color:#605E5C; font-size:13px; margin-bottom:12px; }
.body { color:#323130; font-size:14px; line-height:1.6; white-space:pre-wrap; }
.banner { padding:10px 14px; border-radius:4px; margin:10px 0 14px 0; font-size:14px; }
.banner.spam { background:#FDE7E9; border-left:4px solid #D13438; color:#7A1F22; }
.banner.ham  { background:#DFF6DD; border-left:4px solid #107C10; color:#0B4A0B; }
.badge { display:inline-block; padding:1px 8px; border-radius:10px; font-size:11px; font-weight:600; }
.badge.spam { background:#D13438; color:#fff; }
.badge.ham  { background:#107C10; color:#fff; }
div.stButton > button { width:100%; text-align:left; justify-content:flex-start; border-radius:4px;
                        border:1px solid #E1DFDD; background:#fff; }
div.stButton > button:hover { border-color:#0F6CBD; color:#0F6CBD; background:#EFF6FC; }
div.stButton > button[kind="primary"] { background:#0F6CBD; color:#fff; border-color:#0F6CBD;
                                        justify-content:center; text-align:center; }
</style>
<div class="topbar"><span class="logo">📧 Mail</span>
<span class="sub">Spam Guard &nbsp;|&nbsp; powered by machine learning</span></div>
""",
    unsafe_allow_html=True,
)


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


try:
    model = load_model()
except FileNotFoundError:
    st.error("spam_model.joblib not found. Run `python spam_classifier.py` first.")
    st.stop()


def classify(subject: str, body: str):
    cleaned = clean_text(f"{subject} {body}")
    proba = model.predict_proba([cleaned])[0]
    is_spam = bool(proba[1] >= 0.5)
    return is_spam, float(proba[1] if is_spam else proba[0])


# ---------- Sample inbox ----------
SAMPLES = [
    ("Sarah Johnson", "sarah.j@company.com", "Project review meeting tomorrow",
     "Hi team,\n\nPlease find attached the notes for tomorrow's project review. "
     "We will discuss the gas nomination schedule and next quarter's deliverables.\n\nThanks,\nSarah"),
    ("Prize Center", "winner@claim-now.biz", "CONGRATULATIONS!!! You won $1,000,000",
     "Dear winner, you have been selected to receive a FREE cash prize! Click here now "
     "and send your bank details to claim your money. Limited time offer!!!"),
    ("HR Department", "hr@company.com", "Updated leave policy",
     "Hello all,\n\nThe updated leave policy is now available on the intranet. "
     "Please review it before the end of the month and let us know if you have questions."),
    ("Cheap Meds Online", "sales@pharma-deals.net", "Buy cheap viagra online no prescription",
     "Best prices on medications, order now and save 80%! No prescription needed. "
     "Fast shipping worldwide. Click to order cheap pills today."),
    ("David Lee", "david.lee@company.com", "Lunch on Friday?",
     "Hey, are you free for lunch on Friday? There is a new place near the office "
     "that the team wants to try. Let me know!"),
    ("Online Loans", "approve@fast-cash-loans.info", "Get a loan approved in 5 minutes - no credit check",
     "Need money fast? Get approved instantly with no credit check. Apply now, "
     "guaranteed approval, low rates, earn extra income from home!"),
]

if "emails" not in st.session_state:
    items = []
    for i, (name, addr, subj, body) in enumerate(SAMPLES):
        is_spam, conf = classify(subj, body)
        items.append({"id": i, "name": name, "addr": addr, "subject": subj, "body": body,
                      "spam": is_spam, "conf": conf, "time": datetime.now().strftime("%I:%M %p")})
    st.session_state.emails = items
    st.session_state.selected = 0
    st.session_state.next_id = len(items)

emails = st.session_state.emails
inbox = [e for e in emails if not e["spam"]]
junk = [e for e in emails if e["spam"]]

# ---------- Sidebar: folders ----------
with st.sidebar:
    st.markdown("### Folders")
    folder = st.radio(
        "Folder",
        [f"📥 Inbox ({len(inbox)})", f"🗑️ Junk Email ({len(junk)})"],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.caption(f"Total scanned: {len(emails)}")
    if st.button("Reset mailbox"):
        for k in ("emails", "selected", "next_id"):
            st.session_state.pop(k, None)
        st.rerun()

current = junk if folder.startswith("🗑️") else inbox

tab_mail, tab_new = st.tabs(["📬 Mailbox", "✉️ New message / Scan email"])

# ---------- Mailbox view ----------
with tab_mail:
    left, right = st.columns([2, 3], gap="large")
    with left:
        st.markdown(f'<div class="pane-title">{folder.split(" (")[0]}</div>', unsafe_allow_html=True)
        if not current:
            st.info("No messages in this folder.")
        for e in current:
            label = f"{e['name']}  •  {e['time']}\n{e['subject'][:48]}"
            if st.button(label, key=f"msg_{e['id']}"):
                st.session_state.selected = e["id"]

    with right:
        st.markdown('<div class="pane-title">Reading pane</div>', unsafe_allow_html=True)
        chosen = next((e for e in emails if e["id"] == st.session_state.selected), None)
        if chosen is None:
            st.info("Select a message to read it.")
        else:
            kind = "spam" if chosen["spam"] else "ham"
            verdict = "Moved to Junk Email: this looks like spam" if chosen["spam"] \
                else "This message looks safe"
            icon = "🚨" if chosen["spam"] else "✅"
            st.markdown(
                f"""
<div class="reader">
  <h3>{html.escape(chosen['subject'])}</h3>
  <div class="meta"><b>{html.escape(chosen['name'])}</b> &lt;{html.escape(chosen['addr'])}&gt;
  &nbsp;<span class="badge {kind}">{'SPAM' if chosen['spam'] else 'NOT SPAM'}</span></div>
  <div class="banner {kind}">{icon} {verdict} ({chosen['conf']:.1%} confidence)</div>
  <div class="body">{html.escape(chosen['body'])}</div>
</div>""",
                unsafe_allow_html=True,
            )

# ---------- New message / scan ----------
with tab_new:
    st.markdown('<div class="pane-title">Scan an incoming email</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    sender = c1.text_input("From", placeholder="name@example.com")
    subject = c2.text_input("Subject", placeholder="Subject line")
    body = st.text_area("Message", height=220, placeholder="Paste the email body here...")
    if st.button("Scan and deliver", type="primary"):
        if not (subject.strip() or body.strip()):
            st.warning("Please enter a subject or message first.")
        else:
            is_spam, conf = classify(subject, body)
            new = {"id": st.session_state.next_id, "name": sender.split("@")[0] or "Unknown",
                   "addr": sender or "unknown@unknown", "subject": subject or "(no subject)",
                   "body": body, "spam": is_spam, "conf": conf,
                   "time": datetime.now().strftime("%I:%M %p")}
            st.session_state.next_id += 1
            st.session_state.emails.insert(0, new)
            st.session_state.selected = new["id"]
            if is_spam:
                st.error(f"🚨 SPAM ({conf:.1%} confidence). Delivered to Junk Email.")
            else:
                st.success(f"✅ NOT SPAM ({conf:.1%} confidence). Delivered to Inbox.")
            st.progress(conf if is_spam else 1 - conf, text="Spam probability")
            st.caption("Open the Mailbox tab and choose the folder in the sidebar to read it.")