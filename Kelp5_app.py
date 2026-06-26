
import re, os, warnings
import streamlit as st
import numpy as np

warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ─────────────────────────────────────────────────────────────
# KONSTANTA
# ─────────────────────────────────────────────────────────────
ALL_LABELS = [
    "PRODUCT_POSITIVE","PRODUCT_NEGATIVE","PRODUCT_NEUTRAL",
    "PRICE_POSITIVE","PRICE_NEGATIVE","PRICE_NEUTRAL",
    "PLACE_POSITIVE","PLACE_NEGATIVE","PLACE_NEUTRAL",
    "PROMOTION_POSITIVE","PROMOTION_NEGATIVE","PROMOTION_NEUTRAL",
    "OUT_OF_TOPIC",
]

LABEL_DESC = {
    "PRODUCT_POSITIVE":   "✅ Produk/jasa dinilai positif",
    "PRODUCT_NEGATIVE":   "❌ Produk/jasa dinilai negatif",
    "PRODUCT_NEUTRAL":    "➖ Penyebutan produk/jasa (netral)",
    "PRICE_POSITIVE":     "✅ Harga/nilai dinilai positif (terjangkau)",
    "PRICE_NEGATIVE":     "❌ Harga/nilai dinilai negatif (mahal)",
    "PRICE_NEUTRAL":      "➖ Penyebutan harga (netral)",
    "PLACE_POSITIVE":     "✅ Tempat/suasana/lokasi dinilai positif",
    "PLACE_NEGATIVE":     "❌ Tempat/suasana/lokasi dinilai negatif",
    "PLACE_NEUTRAL":      "➖ Penyebutan tempat (netral)",
    "PROMOTION_POSITIVE": "✅ Promosi/diskon dinilai positif",
    "PROMOTION_NEGATIVE": "❌ Promosi/diskon dinilai negatif",
    "PROMOTION_NEUTRAL":  "➖ Penyebutan promosi (netral)",
    "OUT_OF_TOPIC":       "⚪ Review tidak relevan dengan aspek ABSA",
}

ASPECT_COLORS = {
    "PRODUCT":   "#4F86C6",
    "PRICE":     "#F4A261",
    "PLACE":     "#2A9D8F",
    "PROMOTION": "#E76F51",
    "OUT":       "#9E9E9E",
}

TAG_COLORS = {
    "B-PRODUCT_POSITIVE":   "#C8E6C9","I-PRODUCT_POSITIVE":   "#C8E6C9",
    "B-PRODUCT_NEGATIVE":   "#FFCDD2","I-PRODUCT_NEGATIVE":   "#FFCDD2",
    "B-PRODUCT_NEUTRAL":    "#E3F2FD","I-PRODUCT_NEUTRAL":    "#E3F2FD",
    "B-PRICE_POSITIVE":     "#FFF9C4","I-PRICE_POSITIVE":     "#FFF9C4",
    "B-PRICE_NEGATIVE":     "#FFE0B2","I-PRICE_NEGATIVE":     "#FFE0B2",
    "B-PRICE_NEUTRAL":      "#F3E5F5","I-PRICE_NEUTRAL":      "#F3E5F5",
    "B-PLACE_POSITIVE":     "#B2EBF2","I-PLACE_POSITIVE":     "#B2EBF2",
    "B-PLACE_NEGATIVE":     "#FCE4EC","I-PLACE_NEGATIVE":     "#FCE4EC",
    "B-PLACE_NEUTRAL":      "#E8EAF6","I-PLACE_NEUTRAL":      "#E8EAF6",
    "B-PROMOTION_POSITIVE": "#DCEDC8","I-PROMOTION_POSITIVE": "#DCEDC8",
    "B-PROMOTION_NEGATIVE": "#FFCCBC","I-PROMOTION_NEGATIVE": "#FFCCBC",
    "B-PROMOTION_NEUTRAL":  "#D7CCC8","I-PROMOTION_NEUTRAL":  "#D7CCC8",
    "B-OUT_OF_TOPIC":       "#ECEFF1","I-OUT_OF_TOPIC":       "#ECEFF1",
}

MULTILABEL_PATH = os.path.join(os.path.dirname(__file__), "best_multilabel_model.joblib")
NER_DIR         = "ner_model_artifacts"
MAX_LEN         = 128
PAD_TAG_ID      = -100

# ─────────────────────────────────────────────────────────────
# TEXT CLEANING (harus sama persis dengan notebook)
# ─────────────────────────────────────────────────────────────
def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# ─────────────────────────────────────────────────────────────
# LOAD MULTILABEL MODEL
# ─────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Memuat model multilabel...")
def load_multilabel_model():
    import joblib
    bundle = joblib.load(MULTILABEL_PATH)
    return bundle  # keys: model, tfidf, ft_model, mlb, best_exp, repr_used

# ─────────────────────────────────────────────────────────────
# LOAD NER MODEL
# ─────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Memuat model NER (IndoBERT-CRF)...")
def load_ner_model():
    import torch
    import torch.nn as nn
    from transformers import AutoTokenizer, AutoModel
    try:
        from torchcrf import CRF
    except ImportError:
        import subprocess, sys
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "pytorch-crf"])
        from torchcrf import CRF

    # Load checkpoint metadata
    ckpt = torch.load(
        os.path.join(NER_DIR, "crf_head.pt"),
        map_location="cpu",
        weights_only=False,
    )
    tag2id  = ckpt["tag2id"]
    id2tag  = ckpt["id2tag"]
    num_tags = ckpt["num_tags"]
    max_len  = ckpt.get("max_len", MAX_LEN)

    # Rebuild model architecture (sama persis dengan notebook)
    class IndoBERTCRF(nn.Module):
        def __init__(self, encoder_path, num_tags, dropout=0.2):
            super().__init__()
            self.encoder    = AutoModel.from_pretrained(encoder_path)
            hidden_size     = self.encoder.config.hidden_size
            self.dropout    = nn.Dropout(dropout)
            self.classifier = nn.Linear(hidden_size, num_tags)
            self.crf        = CRF(num_tags, batch_first=True)

        def predict(self, input_ids, attention_mask, token_type_ids=None):
            self.eval()
            with torch.no_grad():
                outputs   = self.encoder(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                )
                emissions = self.classifier(self.dropout(outputs.last_hidden_state))
                crf_mask  = attention_mask.bool()
                decoded   = self.crf.decode(emissions, mask=crf_mask)
            return decoded

    encoder_path = os.path.join(NER_DIR, "encoder")
    model = IndoBERTCRF(encoder_path, num_tags)
    model.classifier.load_state_dict(ckpt["classifier_state"])
    model.crf.load_state_dict(ckpt["crf_state"])
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(NER_DIR)

    return model, tokenizer, tag2id, id2tag, max_len

# ─────────────────────────────────────────────────────────────
# PREDICT MULTILABEL
# ─────────────────────────────────────────────────────────────
def predict_multilabel(text: str, bundle: dict):
    cleaned = clean_text(text)
    model      = bundle["model"]
    tfidf      = bundle["tfidf"]
    ft_model   = bundle["ft_model"]
    mlb        = bundle["mlb"]
    repr_used  = bundle["repr_used"]

    if repr_used == "TF-IDF":
        X = tfidf.transform([cleaned])
    else:  # FastText
        tokens = cleaned.split()
        vecs   = [ft_model[t] for t in tokens if t in ft_model]
        vec    = np.mean(vecs, axis=0) if vecs else np.zeros(ft_model.vector_size)
        X      = vec.reshape(1, -1)

    pred = model.predict(X)
    if hasattr(pred, "toarray"):
        pred = pred.toarray()
    pred_labels = [ALL_LABELS[i] for i, v in enumerate(pred[0]) if v == 1]

    # Confidence/probability — tersedia untuk model dengan predict_proba
    proba_dict = {}
    try:
        proba = model.predict_proba(X)
        if hasattr(proba, "toarray"):
            proba = proba.toarray()
        proba_flat = np.array(proba).flatten()
        # Pastikan panjang proba sama dengan ALL_LABELS
        if len(proba_flat) == len(ALL_LABELS):
            proba_dict = {lbl: float(proba_flat[i]) for i, lbl in enumerate(ALL_LABELS)}
    except Exception:
        # OneVsRestClassifier dengan LinearSVC → estimasi dari decision_function
        try:
            df = model.decision_function(X)
            if hasattr(df, "toarray"):
                df = df.toarray()
            df_flat = np.array(df).flatten()
            if len(df_flat) == len(ALL_LABELS):
                # Sigmoid normalisasi agar interpretable sebagai confidence
                conf = 1 / (1 + np.exp(-df_flat))
                proba_dict = {lbl: float(conf[i]) for i, lbl in enumerate(ALL_LABELS)}
        except Exception:
            pass

    return pred_labels, proba_dict

# ─────────────────────────────────────────────────────────────
# PREDICT NER
# ─────────────────────────────────────────────────────────────
def predict_ner(text: str, model, tokenizer, id2tag: dict, max_len: int):
    import torch

    # Tokenisasi word-level sederhana (konsisten dengan Prodigy token)
    words = text.split()

    enc = tokenizer(
        words,
        is_split_into_words=True,
        padding="max_length",
        truncation=True,
        max_length=max_len,
        return_tensors="pt",
        return_token_type_ids=True,
    )

    word_ids = enc.word_ids(batch_index=0)

    input_ids      = enc["input_ids"]
    attention_mask = enc["attention_mask"]
    token_type_ids = enc.get("token_type_ids", torch.zeros_like(input_ids))

    decoded = model.predict(input_ids, attention_mask, token_type_ids)
    tag_ids = decoded[0]  # list of int (hanya posisi valid, bukan PAD)

    # Mapping: ambil tag subword pertama per word
    word_tags = {}
    prev_wid  = None
    tag_ptr   = 0
    for pos, wid in enumerate(word_ids):
        if wid is None:
            continue
        if wid != prev_wid:
            if tag_ptr < len(tag_ids):
                word_tags[wid] = id2tag.get(tag_ids[tag_ptr], "O")
                tag_ptr += 1
        prev_wid = wid

    result = []
    for i, word in enumerate(words):
        tag = word_tags.get(i, "O")
        result.append((word, tag))
    return result

# ─────────────────────────────────────────────────────────────
# RENDER NER HIGHLIGHT
# ─────────────────────────────────────────────────────────────
def infer_aspect_sentiments(pred_labels, proba_dict):
    """Pilih label aspek-sentimen terbaik untuk tiap aspek dari model multilabel."""
    aspect_sentiments = {}
    aspects = ["PRODUCT", "PRICE", "PLACE", "PROMOTION"]

    for aspect in aspects:
        candidates = [lbl for lbl in ALL_LABELS if lbl.startswith(f"{aspect}_")]
        active = [lbl for lbl in pred_labels if lbl in candidates]

        if active:
            best = max(active, key=lambda lbl: proba_dict.get(lbl, 1.0))
        elif proba_dict:
            best = max(candidates, key=lambda lbl: proba_dict.get(lbl, 0.0))
        else:
            best = f"{aspect}_NEUTRAL"

        aspect_sentiments[aspect] = best

    return aspect_sentiments


def enrich_ner_tags_with_sentiment(token_tags, aspect_sentiments):
    """Ubah tag aspect-only seperti B-PLACE menjadi B-PLACE_POSITIVE."""
    enriched = []
    prev_label = None

    for word, tag in token_tags:
        if tag == "O":
            enriched.append((word, tag))
            prev_label = None
            continue

        _, label = tag.split("-", 1)
        full_label = aspect_sentiments.get(label, label)
        prefix = "I" if full_label == prev_label else "B"
        enriched.append((word, f"{prefix}-{full_label}"))
        prev_label = full_label

    return enriched


def render_ner_html(token_tags):
    """Render highlighted spans dari token-tag pairs."""
    html_parts = []
    i = 0
    while i < len(token_tags):
        word, tag = token_tags[i]
        if tag == "O":
            html_parts.append(f'<span style="margin:0 2px">{word}</span>')
            i += 1
        else:
            # Kumpulkan span B + semua I yang menyusul
            prefix, label = tag.split("-", 1)
            span_words  = [word]
            span_color  = TAG_COLORS.get(tag, ASPECT_COLORS.get(label, "#E0E0E0"))
            i += 1
            while i < len(token_tags):
                next_word, next_tag = token_tags[i]
                if next_tag == f"I-{label}":
                    span_words.append(next_word)
                    i += 1
                else:
                    break
            span_text   = " ".join(span_words)
            short_label = label.replace("_", " ")
            html_parts.append(
                f'<mark style="background:{span_color};border-radius:4px;'
                f'padding:2px 6px;margin:0 2px;font-weight:500">'
                f'{span_text}'
                f'<sup style="font-size:0.65em;margin-left:4px;color:#333">{short_label}</sup>'
                f'</mark>'
            )
    return '<p style="line-height:2.4;font-size:1.05em">' + " ".join(html_parts) + "</p>"

# ─────────────────────────────────────────────────────────────
# MAIN APP
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ABSA Review Analyzer",
    page_icon="🔍",
    layout="wide",
)

st.title("🔍 ABSA Review Analyzer")
st.caption(
    "Aspect-Based Sentiment Analysis dari Review Google Places — "
    "Multilabel Text Classification & Named Entity Recognition"
)

tab_multi, tab_ner, tab_about = st.tabs([
    "📊 Prediksi Multilabel ABSA",
    "🏷️ Prediksi NER ABSA",
    "ℹ️ Tentang Aplikasi",
])

# ═══════════════════════════════════════
# TAB 1 — MULTILABEL
# ═══════════════════════════════════════
with tab_multi:
    st.header("Prediksi Label ABSA (Review-Level)")
    st.markdown(
        "Model memprediksi satu atau lebih label aspek-sentimen dari teks review. "
        "Input teks bebas bahasa Indonesia."
    )

    col_in, col_out = st.columns([1, 1], gap="large")

    with col_in:
        text_multi = st.text_area(
            "Masukkan teks review:",
            height=180,
            placeholder="Contoh: Tempatnya luas, harganya juga terjangkau. Tempatnya nyaman tapi parkirnya susah.",
            key="text_multi",
        )
        btn_multi = st.button("🔮 Prediksi Label", type="primary", key="btn_multi")

    with col_out:
        if btn_multi:
            if not text_multi.strip():
                st.warning("Masukkan teks review terlebih dahulu.")
            else:
                try:
                    bundle = load_multilabel_model()
                    with st.spinner("Memproses..."):
                        pred_labels, proba_dict = predict_multilabel(text_multi, bundle)

                    if not pred_labels:
                        st.info("Model tidak mendeteksi label ABSA yang signifikan pada review ini.")
                    else:
                        st.success(f"**{len(pred_labels)} label terdeteksi**")
                        for lbl in pred_labels:
                            desc = LABEL_DESC.get(lbl, lbl)
                            st.markdown(
                                f'<div style="background:#F0F4FF;border-left:4px solid #4F86C6;'
                                f'padding:8px 12px;margin:4px 0;border-radius:4px">'
                                f'<b>{lbl}</b><br><span style="font-size:0.88em;color:#555">{desc}</span></div>',
                                unsafe_allow_html=True,
                            )

                    # Confidence / probability
                    if proba_dict:
                        st.markdown("---")
                        st.subheader("Confidence Score per Label")
                        st.caption(
                            "Confidence dihitung dari decision function model (sigmoid-normalized). "
                            "Nilai > 0.5 cenderung diprediksi sebagai label aktif."
                        )
                        import pandas as pd
                        df_conf = pd.DataFrame([
                            {"Label": lbl, "Confidence": proba_dict[lbl],
                             "Aktif": "✅" if lbl in pred_labels else ""}
                            for lbl in ALL_LABELS
                        ]).sort_values("Confidence", ascending=False)

                        st.dataframe(
                            df_conf.style.format({"Confidence": "{:.3f}"})
                                   .background_gradient(subset=["Confidence"], cmap="Blues"),
                            use_container_width=True,
                            height=380,
                        )

                except FileNotFoundError:
                    st.error(
                        f"File model `{MULTILABEL_PATH}` tidak ditemukan. "
                        "Pastikan file ada di direktori yang sama dengan `app.py`."
                    )
                except Exception as e:
                    st.error(f"Error: {e}")

    # Interpretasi
    with st.expander("📖 Interpretasi Label ABSA"):
        cols = st.columns(2)
        for i, (lbl, desc) in enumerate(LABEL_DESC.items()):
            cols[i % 2].markdown(f"**{lbl}**  \n{desc}")

# ═══════════════════════════════════════
# TAB 2 — NER
# ═══════════════════════════════════════
with tab_ner:
    st.header("Prediksi NER ABSA (Token-Level Span Extraction)")
    st.markdown(
        "Model mengenali **span** dalam teks yang merujuk pada aspek ABSA "
        "(Product, Price, Place, Promotion) beserta sentimennya."
    )

    col_in2, col_out2 = st.columns([1, 1], gap="large")

    with col_in2:
        text_ner = st.text_area(
            "Masukkan teks review:",
            height=180,
            placeholder="Contoh: Bahan bajunya enak banget, harga standar, tempatnya nyaman tapi sempit.",
            key="text_ner",
        )
        btn_ner = st.button("🏷️ Ekstrak Entitas", type="primary", key="btn_ner")

    with col_out2:
        if btn_ner:
            if not text_ner.strip():
                st.warning("Masukkan teks review terlebih dahulu.")
            else:
                try:
                    ner_model, ner_tokenizer, tag2id, id2tag, max_len = load_ner_model()
                    multilabel_bundle = load_multilabel_model()
                    with st.spinner("Memproses..."):
                        token_tags = predict_ner(text_ner, ner_model, ner_tokenizer, id2tag, max_len)
                        pred_labels, proba_dict = predict_multilabel(text_ner, multilabel_bundle)
                        aspect_sentiments = infer_aspect_sentiments(pred_labels, proba_dict)
                        token_tags = enrich_ner_tags_with_sentiment(token_tags, aspect_sentiments)

                    # Highlight HTML
                    st.subheader("Visualisasi Span")
                    ner_html = render_ner_html(token_tags)
                    st.markdown(ner_html, unsafe_allow_html=True)

                    # Tabel entitas
                    entities = []
                    i = 0
                    while i < len(token_tags):
                        word, tag = token_tags[i]
                        if tag.startswith("B-"):
                            label     = tag[2:]
                            span_words = [word]
                            i += 1
                            while i < len(token_tags) and token_tags[i][1] == f"I-{label}":
                                span_words.append(token_tags[i][0])
                                i += 1
                            entities.append({
                                "Span": " ".join(span_words),
                                "Label": label,
                                "Aspek": label.rsplit("_", 1)[0],
                                "Sentimen": label.rsplit("_", 1)[1] if "_" in label else "-",
                            })
                        else:
                            i += 1

                    if entities:
                        st.markdown("---")
                        st.subheader("Entitas Terdeteksi")
                        import pandas as pd
                        df_ent = pd.DataFrame(entities)
                        st.dataframe(df_ent, use_container_width=True, hide_index=True)
                    else:
                        st.info("Tidak ada entitas ABSA yang terdeteksi pada review ini.")

                except FileNotFoundError:
                    st.error(
                        f"Direktori model `{NER_DIR}/` tidak ditemukan. "
                        "Pastikan folder `ner_model_artifacts/` ada di direktori yang sama dengan `app.py`."
                    )
                except Exception as e:
                    st.error(f"Error: {e}")

    # Legend warna
    with st.expander("🎨 Legenda Warna Span"):
        legend_items = [
            ("PRODUCT_POSITIVE", "#C8E6C9"),("PRODUCT_NEGATIVE", "#FFCDD2"),("PRODUCT_NEUTRAL", "#E3F2FD"),
            ("PRICE_POSITIVE",   "#FFF9C4"),("PRICE_NEGATIVE",   "#FFE0B2"),("PRICE_NEUTRAL",   "#F3E5F5"),
            ("PLACE_POSITIVE",   "#B2EBF2"),("PLACE_NEGATIVE",   "#FCE4EC"),("PLACE_NEUTRAL",   "#E8EAF6"),
            ("PROMOTION_POSITIVE","#DCEDC8"),("PROMOTION_NEGATIVE","#FFCCBC"),("PROMOTION_NEUTRAL","#D7CCC8"),
        ]
        cols_leg = st.columns(3)
        for idx, (lbl, color) in enumerate(legend_items):
            cols_leg[idx % 3].markdown(
                f'<span style="background:{color};padding:3px 8px;border-radius:3px;font-size:0.85em">{lbl}</span>',
                unsafe_allow_html=True,
            )

# ═══════════════════════════════════════
# TAB 3 — ABOUT
# ═══════════════════════════════════════
with tab_about:
    st.header("Tentang Aplikasi")
    st.markdown("""
    ### Proyek UAS Pengolahan Bahasa Alami
    **Universitas Atma Jaya Yogyakarta — Genap 2025/2026**

    Aplikasi ini merupakan deployment dari dua model NLP yang dikembangkan untuk analisis review Google Places berbahasa Indonesia menggunakan pendekatan **Aspect-Based Sentiment Analysis (ABSA)**.

    ---
    ### Proyek A — Multilabel Text Classification
    | Komponen | Detail |
    |---|---|
    | **Task** | Memprediksi label ABSA pada level review (multi-hot) |
    | **Representasi** | TF-IDF bigram (10K fitur) & FastText pretrained Indonesia (cc.id.300) |
    | **Pendekatan** | Binary Relevance & Classifier Chain |
    | **Algoritma** | Logistic Regression, LinearSVC |
    | **Output** | Kombinasi 13 label aspek-sentimen |

    ### Proyek B — Named Entity Recognition (NER)
    | Komponen | Detail |
    |---|---|
    | **Task** | Mengenali span aspek ABSA pada level token |
    | **Model** | IndoBERT (`indobenchmark/indobert-base-p1`) + CRF |
    | **Skema tag** | BIO (B-/I-/O) dengan 13 label entitas |
    | **Decoding** | Viterbi via CRF layer |
    | **Output** | Span teks dengan label aspek-sentimen |

    ---
    ### Hubungan Output Multilabel ↔ NER
    Output kedua model saling melengkapi:
    - **Multilabel** menjawab *"aspek apa saja yang dibicarakan?"* pada level review
    - **NER** menjawab *"di bagian mana tepatnya?"* pada level token/span

    Contoh: jika Multilabel memprediksi `PRODUCT_POSITIVE`, NER akan menunjukkan
    span kalimat spesifik seperti *"tempatnya luas"* yang menjadi dasar prediksi tersebut.
    """)
