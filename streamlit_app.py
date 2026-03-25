import streamlit as st
import requests
import time
import os
import json
import pandas as pd
from pathlib import Path

# ============================================================
# Configuration
# ============================================================
API_BASE_URL = "https://a5af-34-143-214-246.ngrok-free.app"
API_PREFIX   = "/api/v1/rag"

st.set_page_config(
    page_title="RAG Anything UI",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# Custom CSS
# ============================================================
st.markdown("""
    <style>
    .main { background-color: #0e1117; color: #ffffff; }
    .stButton>button {
        width: 100%; border-radius: 5px; height: 3em;
        background-color: #262730; color: white; border: 1px solid #4a4a4a;
    }
    .stButton>button:hover { border-color: #ff4b4b; color: #ff4b4b; }
    .log-container {
        font-family: 'Courier New', Courier, monospace;
        background-color: #000; color: #0f0;
        padding: 10px; border-radius: 5px;
        height: 200px; overflow-y: auto; font-size: 0.8em;
    }
    .metric-card {
        background: linear-gradient(135deg, #1e1e2e, #2a2a3e);
        border: 1px solid #444; border-radius: 12px;
        padding: 18px; text-align: center;
    }
    .metric-score { font-size: 2.5em; font-weight: bold; }
    .score-good  { color: #00d26a; }
    .score-ok    { color: #f7c948; }
    .score-poor  { color: #ff4b4b; }
    </style>
    """, unsafe_allow_html=True)

st.title("🤖 RAG Anything — Production UI")
st.markdown("---")

# ============================================================
# Sidebar
# ============================================================
with st.sidebar:
    st.header("⚙️ Settings")
    base_url = st.text_input("Ngrok/API Host", value=API_BASE_URL)
    api_url  = f"{base_url.rstrip('/')}{API_PREFIX}"
    st.info(f"API: {api_url}")

    st.markdown("---")
    st.header("🗂️ Workspace Manager")

    try:
        proj_resp        = requests.get(f"{api_url}/projects", timeout=5)
        existing_projects = proj_resp.json().get("projects", []) if proj_resp.status_code == 200 else []
    except:
        existing_projects = []

    new_project = st.text_input("Create New Project", placeholder="e.g. legal_docs")
    if st.button("➕ Create"):
        if new_project:
            st.session_state.project_id = new_project
            st.success(f"Project '{new_project}' ready!")
            st.rerun()

    if "project_id" in st.session_state and st.session_state.project_id not in existing_projects:
        existing_projects.append(st.session_state.project_id)

    if existing_projects:
        existing_projects.sort()
        selected_project = st.selectbox(
            "Select Workspace", options=existing_projects,
            index=existing_projects.index(st.session_state.project_id)
                  if st.session_state.get("project_id") in existing_projects else 0
        )
        st.session_state.project_id = selected_project
    else:
        st.warning("No projects found. Create one above!")
        if "project_id" not in st.session_state:
            st.session_state.project_id = "default"

    st.markdown(f"**Current Workspace:** `{st.session_state.project_id}`")
    st.markdown("---")
    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

# ============================================================
# 3 TABS
# ============================================================
tab1, tab2, tab3 = st.tabs(["📤 Upload & Process", "💬 Chat with RAG", "📊 Evaluate (RAGAS)"])

# ============================================================================
# TAB 1 — Upload & Process
# ============================================================================
with tab1:
    st.header("📁 Document Upload")

    # ── Upload mode toggle ──────────────────────────────────────
    upload_mode = st.radio(
        "Upload mode",
        ["Single file", "Multiple files (folder)"],
        horizontal=True,
        key="upload_mode",
    )

    if upload_mode == "Single file":
        uploaded_files = st.file_uploader(
            "Choose a file (PDF, TXT, DOCX, MD, Images)",
            type=["pdf", "txt", "docx", "doc", "md", "png", "jpg", "jpeg"],
            accept_multiple_files=False,
        )
        uploaded_files = [uploaded_files] if uploaded_files else []
    else:
        uploaded_files = st.file_uploader(
            "Choose multiple files — select all files from your folder at once",
            type=["pdf", "txt", "docx", "doc", "md", "png", "jpg", "jpeg"],
            accept_multiple_files=True,
        )
        if uploaded_files:
            st.info(f"📂 **{len(uploaded_files)} file(s)** selected: "
                    + ", ".join(f"`{f.name}`" for f in uploaded_files[:5])
                    + ("..." if len(uploaded_files) > 5 else ""))

    # ── Upload button ────────────────────────────────────────────
    if uploaded_files and st.button("🚀 Start Processing", type="primary"):

        # ── SINGLE FILE: use original /upload endpoint ──────────
        if len(uploaded_files) == 1:
            file_obj = uploaded_files[0]
            with st.status(f"Processing `{file_obj.name}`...", expanded=True) as status:
                try:
                    resp = requests.post(
                        f"{api_url}/upload",
                        files={"file": (file_obj.name, file_obj.getvalue(), file_obj.type)},
                        data={"project_id": st.session_state.project_id},
                    )
                    if resp.status_code == 200:
                        task_id = resp.json()["task_id"]
                        st.success(f"Uploaded! Task: `{task_id}`")
                        progress_bar    = st.progress(0, text="Initializing...")
                        log_placeholder = st.empty()
                        while True:
                            sr = requests.get(f"{api_url}/status/{task_id}")
                            if sr.status_code == 200:
                                d = sr.json()
                                progress_bar.progress(d.get("percentage", 0) / 100,
                                                      text=f"Status: {d.get('message')}")
                                with log_placeholder.container():
                                    st.markdown("**Logs:**")
                                    log_html = "".join(f"<div>> {l}</div>" for l in d.get("logs", []))
                                    st.markdown(f'<div class="log-container">{log_html}</div>',
                                                unsafe_allow_html=True)
                                if d.get("status") == "completed":
                                    st.balloons()
                                    status.update(label="✅ Complete!", state="complete", expanded=False)
                                    break
                                elif d.get("status") == "failed":
                                    st.error(f"Failed: {d.get('error')}")
                                    status.update(label="❌ Failed", state="error")
                                    break
                            time.sleep(2)
                    else:
                        st.error(f"Upload error: {resp.text}")
                        status.update(label="Upload Failed", state="error")
                except Exception as e:
                    st.error(f"Connection error: {e}")
                    status.update(label="Connection Error", state="error")

        # ── BATCH FILES: use /upload_batch endpoint ─────────────
        else:
            st.info(f"Uploading **{len(uploaded_files)}** files to workspace "
                    f"`{st.session_state.project_id}`...")
            try:
                multi_files = [
                    ("files", (f.name, f.getvalue(), f.type))
                    for f in uploaded_files
                ]
                batch_resp = requests.post(
                    f"{api_url}/upload_batch",
                    files=multi_files,
                    data={"project_id": st.session_state.project_id},
                )
                if batch_resp.status_code != 200:
                    st.error(f"Batch upload error: {batch_resp.text}")
                    st.stop()

                batch_data = batch_resp.json()
                results    = batch_data["results"]
                task_ids   = {r["filename"]: r["task_id"] for r in results if r["task_id"]}
                skipped    = [r for r in results if r["status"] == "skipped"]

                if skipped:
                    st.warning("⚠️ Skipped files (unsupported/too large): "
                               + ", ".join(f"`{r['filename']}`" for r in skipped))

                st.success(f"✅ **{len(task_ids)}/{len(uploaded_files)}** files queued "
                           "— processing concurrently...")

                # ── Live dashboard: poll all tasks until all done ──
                st.markdown("### 📊 Processing Dashboard")
                status_placeholder = st.empty()
                all_done = False

                while not all_done:
                    rows = []
                    all_done = True
                    for r in results:
                        if r["status"] == "skipped":
                            rows.append({"File": r["filename"], "Status": "⏭️ Skipped",
                                         "Progress": "—", "Message": r["message"]})
                            continue
                        tid = r["task_id"]
                        try:
                            sr = requests.get(f"{api_url}/status/{tid}", timeout=5)
                            d  = sr.json() if sr.status_code == 200 else {}
                        except Exception:
                            d = {}
                        s = d.get("status", "unknown")
                        pct = d.get("percentage", 0)
                        msg = d.get("message", "")
                        icon = {"completed": "✅", "failed": "❌", "pending": "⏳",
                                "parsing": "🔍", "indexing": "📊"}.get(s, "🔄")
                        rows.append({"File": r["filename"],
                                     "Status": f"{icon} {s.capitalize()}",
                                     "Progress": f"{pct:.0f}%",
                                     "Message": msg[:60]})
                        if s not in ("completed", "failed"):
                            all_done = False

                    with status_placeholder.container():
                        st.dataframe(
                            pd.DataFrame(rows),
                            use_container_width=True,
                            hide_index=True,
                        )

                    if not all_done:
                        time.sleep(3)

                # Final summary
                completed_count = sum(1 for r in rows if "✅" in r["Status"])
                failed_count    = sum(1 for r in rows if "❌" in r["Status"])
                if failed_count == 0:
                    st.balloons()
                    st.success(f"🎉 All {completed_count} files indexed successfully!")
                else:
                    st.warning(f"Done: {completed_count} ✅ completed, {failed_count} ❌ failed.")

            except Exception as e:
                st.error(f"Batch upload connection error: {e}")


# ============================================================================
# TAB 2 — Chat with RAG
# ============================================================================
with tab2:
    st.header("Query the RAG Engine")

    MODE_DESCRIPTIONS = {
        "hybrid": "🔀 Hybrid — Kết hợp Graph + Vector. Tốt cho câu hỏi tổng hợp.",
        "local":  "📍 Local   — Tìm theo vector chunk. Tốt cho truy vấn nguyên văn điều luật cụ thể.",
        "global": "🌐 Global  — Tìm trên Knowledge Graph. Tốt cho câu hỏi khái niệm, mối quan hệ.",
        "naive":  "📄 Naive   — Truy vấn thẳng vào text chunk, không dùng Graph.",
    }
    mode_options  = list(MODE_DESCRIPTIONS.keys())
    selected_mode = st.radio(
        "🎛️ Chọn chế độ Query:",
        options=mode_options,
        format_func=lambda m: MODE_DESCRIPTIONS[m],
        horizontal=False,
        index=mode_options.index(st.session_state.get("query_mode", "hybrid")),
        key="query_mode_selector",
    )
    st.session_state.query_mode = selected_mode

    # ── Query Settings (top_k + response_type) ──────────────────────────────
    with st.expander("⚙️ Query Settings", expanded=False):
        col_topk, col_rtype = st.columns(2)
        with col_topk:
            top_k = st.slider(
                "🔢 Top-K (số entity/chunk tìm kiếm)",
                min_value=10, max_value=500, value=100, step=10,
                help="Số lượng entity/relation/chunk LightRAG lấy ra trước khi lọc. "
                     "Cao hơn → recall tốt hơn nhưng chậm hơn.",
                key="chat_top_k",
            )
        with col_rtype:
            response_type = st.selectbox(
                "📝 Response Type",
                options=[
                    "Structured List",
                    "Single Paragraph",
                    "Multiple Paragraphs",
                    "Bullet Points",
                    "Plain Text",
                ],
                index=0,
                help="Định dạng câu trả lời LLM sẽ trả về.\n"
                     "• Structured List: danh sách có cấu trúc\n"
                     "• Single/Multiple Paragraphs: đoạn văn liên tục\n"
                     "• Bullet Points: dạng gạch đầu dòng",
                key="chat_response_type",
            )
    st.markdown("---")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask anything about your documents..."):
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("assistant"):
            with st.spinner(f"Thinking... (mode: {st.session_state.query_mode}, top_k: {top_k})"):
                try:
                    payload = {
                        "query":         prompt,
                        "mode":          st.session_state.query_mode,
                        "project_id":    st.session_state.project_id,
                        "top_k":         top_k,
                        "response_type": response_type,
                    }
                    response = requests.post(f"{api_url}/query", json=payload)
                    if response.status_code == 200:
                        full_response = response.json().get("response", "No response received.")
                        st.markdown(full_response)
                        st.session_state.messages.append({"role": "assistant", "content": full_response})
                    else:
                        st.error(f"API Error: {response.status_code}")
                except Exception as e:
                    st.error(f"Error: {str(e)}")


# ============================================================================
# TAB 3 — Evaluate (RAGAS)
# ============================================================================
with tab3:
    st.header("📊 RAGAS Evaluation")
    st.markdown("""
    Đánh giá chất lượng hệ thống RAG theo 4 metrics của RAGAS:
    **Faithfulness** · **Answer Relevancy** · **Context Recall** · **Context Precision**

    **Cách dùng:**
    1. Nhập các cặp câu hỏi + câu trả lời chuẩn (ground truth) vào bảng bên dưới
    2. Nhấn **"🔍 Fetch RAG Answers"** → hệ thống tự chạy query và điền `answer` + `contexts`
    3. Nhấn **"🚀 Run RAGAS"** → xem điểm đánh giá chi tiết
    """)
    st.markdown("---")

    # ── 3.1: OpenAI Key for RAGAS judge ──────────────────────────────────────
    with st.expander("🔑 OpenAI API Key (dùng để RAGAS chấm điểm)", expanded=False):
        ragas_openai_key = st.text_input(
            "OpenAI API Key", type="password",
            placeholder="sk-...",
            help="RAGAS dùng GPT-4o-mini để chấm điểm. Chỉ cần nhập khi chạy evaluation.",
            key="ragas_openai_key"
        )

    # ── 3.2: Query mode for fetching answers ────────────────────────────────
    col_mode, col_info = st.columns([1, 2])
    with col_mode:
        eval_mode = st.selectbox(
            "Query Mode khi fetch answers:",
            options=["hybrid", "local", "global", "naive"],
            index=0,
            key="eval_query_mode"
        )
    with col_info:
        st.info("ℹ️ Contexts được thu thập qua `naive` mode (raw chunks), answer qua mode đã chọn.")

    st.markdown("---")

    # ── 3.3: Dataset editor ─────────────────────────────────────────────────
    st.subheader("📝 Test Dataset")
    st.caption("Nhập câu hỏi và câu trả lời chuẩn. Cột `answer` và `contexts` sẽ được tự động điền.")

    # Default empty dataset
    if "eval_dataset" not in st.session_state:
        st.session_state.eval_dataset = pd.DataFrame({
            "question":     ["", "", ""],
            "ground_truth": ["", "", ""],
            "answer":       ["", "", ""],
            "contexts":     ["", "", ""],
            "status":       ["⬜ Chưa fetch", "⬜ Chưa fetch", "⬜ Chưa fetch"],
        })

    col_add, col_clear, col_import, col_export = st.columns(4)

    with col_add:
        if st.button("➕ Thêm dòng"):
            new_row = pd.DataFrame({
                "question": [""], "ground_truth": [""],
                "answer": [""], "contexts": [""], "status": ["⬜ Chưa fetch"]
            })
            st.session_state.eval_dataset = pd.concat(
                [st.session_state.eval_dataset, new_row], ignore_index=True
            )
            st.rerun()

    with col_clear:
        if st.button("🗑️ Xóa tất cả"):
            st.session_state.eval_dataset = pd.DataFrame({
                "question": [""], "ground_truth": [""],
                "answer": [""], "contexts": [""], "status": ["⬜ Chưa fetch"]
            })
            st.rerun()

    with col_import:
        uploaded_csv = st.file_uploader("📥 Import CSV", type=["csv"], label_visibility="collapsed", key="csv_import")
        # Chỉ import khi file MỚI (tránh vòng lặp vô hạn do rerun giữ lại file uploader)
        if uploaded_csv and st.session_state.get("_last_imported_csv") != uploaded_csv.name:
            try:
                imported = pd.read_csv(
                    uploaded_csv,
                    engine="python",
                    on_bad_lines="skip",
                    quotechar='"',
                    encoding="utf-8-sig",
                )
                for col in ["answer", "contexts", "status"]:
                    if col not in imported.columns:
                        imported[col] = "" if col != "status" else "⬜ Chưa fetch"
                st.session_state.eval_dataset = imported
                # Ép kiểu tất cả cột về string để tránh lỗi .str accessor với NaN
                for str_col in ["question", "ground_truth", "answer", "contexts", "status"]:
                    if str_col in st.session_state.eval_dataset.columns:
                        st.session_state.eval_dataset[str_col] = (
                            st.session_state.eval_dataset[str_col].fillna("").astype(str)
                        )
                st.session_state._last_imported_csv = uploaded_csv.name  # Đánh dấu đã import
                st.success(f"✅ Imported {len(imported)} rows từ '{uploaded_csv.name}'!")
                st.rerun()
            except Exception as e:
                st.error(f"❌ Lỗi đọc CSV: {e}")

    with col_export:
        csv_data = st.session_state.eval_dataset.to_csv(index=False).encode("utf-8-sig")
        st.download_button("📤 Export CSV", data=csv_data, file_name="eval_dataset.csv", mime="text/csv")

    # Editable table — only question and ground_truth editable by user
    edited_df = st.data_editor(
        st.session_state.eval_dataset[["question", "ground_truth", "status"]],
        use_container_width=True,
        num_rows="dynamic",
        column_config={
            "question":     st.column_config.TextColumn("❓ Câu hỏi", width="large"),
            "ground_truth": st.column_config.TextColumn("✅ Câu trả lời chuẩn (ground truth)", width="large"),
            "status":       st.column_config.TextColumn("📌 Trạng thái", disabled=True, width="small"),
        },
        key="dataset_editor"
    )

    # Sync edited rows back
    st.session_state.eval_dataset["question"]     = edited_df["question"]
    st.session_state.eval_dataset["ground_truth"] = edited_df["ground_truth"]

    # ── Xóa từng dòng ───────────────────────────────────────────────────────
    df_cur = st.session_state.eval_dataset
    non_empty = df_cur[df_cur["question"].astype(str).str.strip() != ""]

    if not non_empty.empty:
        # Tạo nhãn hiển thị dạng "Dòng 1: Câu hỏi..." cho multiselect
        row_labels = {
            idx: f"#{idx+1}: {str(row['question'])[:60]}..."
            for idx, row in non_empty.iterrows()
        }
        selected_labels = st.multiselect(
            "🗑️ Chọn dòng cần xóa:",
            options=list(row_labels.keys()),
            format_func=lambda i: row_labels[i],
            placeholder="Chọn 1 hoặc nhiều dòng...",
            key="rows_to_delete",
        )
        if selected_labels and st.button("❌ Xóa các dòng đã chọn", type="secondary"):
            st.session_state.eval_dataset = (
                st.session_state.eval_dataset
                .drop(index=selected_labels)
                .reset_index(drop=True)
            )
            st.rerun()

    st.markdown("---")


    # ── 3.4: Fetch RAG Answers ───────────────────────────────────────────────
    if st.button("🔍 Fetch RAG Answers", type="primary", use_container_width=True):
        df = st.session_state.eval_dataset.copy()
        valid_rows = df[df["question"].str.strip() != ""]

        if valid_rows.empty:
            st.warning("Vui lòng nhập ít nhất 1 câu hỏi!")
        else:
            progress = st.progress(0, text="Đang query RAG system...")
            total    = len(valid_rows)

            for i, (idx, row) in enumerate(valid_rows.iterrows()):
                progress.progress((i) / total, text=f"Đang fetch {i+1}/{total}: {row['question'][:60]}...")
                try:
                    payload  = {
                        "query":      row["question"],
                        "mode":       eval_mode,
                        "project_id": st.session_state.project_id,
                    }
                    resp = requests.post(f"{api_url}/query_eval", json=payload, timeout=180)

                    if resp.status_code == 200:
                        data = resp.json()
                        st.session_state.eval_dataset.at[idx, "answer"]   = data.get("answer", "")
                        st.session_state.eval_dataset.at[idx, "contexts"] = " ||| ".join(data.get("contexts", []))
                        st.session_state.eval_dataset.at[idx, "status"]   = "✅ Đã fetch"
                    else:
                        st.session_state.eval_dataset.at[idx, "status"] = f"❌ Lỗi {resp.status_code}"
                except Exception as e:
                    st.session_state.eval_dataset.at[idx, "status"] = f"❌ {str(e)[:40]}"

            progress.progress(1.0, text="✅ Fetch hoàn tất!")
            time.sleep(0.5)
            st.rerun()

    # Show current state of full dataset (with answer + contexts)
    with st.expander("🔎 Xem đầy đủ bảng (cả answer & contexts)", expanded=False):
        st.dataframe(st.session_state.eval_dataset, use_container_width=True)

    st.markdown("---")

    # ── 3.5: Run RAGAS ──────────────────────────────────────────────────────
    st.subheader("🚀 Run RAGAS Evaluation")

    ready_rows = st.session_state.eval_dataset[
        (st.session_state.eval_dataset["question"].astype(str).str.strip() != "") &
        (st.session_state.eval_dataset["question"].astype(str).str.strip() != "nan") &
        (st.session_state.eval_dataset["answer"].astype(str).str.strip() != "") &
        (st.session_state.eval_dataset["answer"].astype(str).str.strip() != "nan")
    ]

    if len(ready_rows) == 0:
        st.info("👆 Fetch RAG answers trước, sau đó chạy RAGAS evaluation.")
    else:
        st.success(f"✅ {len(ready_rows)} câu hỏi đã sẵn sàng để evaluate.")

        if st.button("📊 Run RAGAS", type="primary", use_container_width=True):
            if not ragas_openai_key:
                st.error("⚠️ Vui lòng nhập OpenAI API Key để RAGAS chạy được!")
            else:
                try:
                    with st.spinner("⏳ Đang cài đặt RAGAS..."):
                        import subprocess
                        subprocess.run(["pip", "install", "-q", "ragas", "langchain-openai"], check=True)

                    import os as _os
                    _os.environ["OPENAI_API_KEY"] = ragas_openai_key

                    from datasets import Dataset as HFDataset
                    from ragas import evaluate
                    from ragas.metrics import faithfulness, answer_relevancy, context_recall, context_precision
                    from langchain_openai import ChatOpenAI, OpenAIEmbeddings

                    # Prepare RAGAS dataset
                    ragas_data = []
                    for _, row in ready_rows.iterrows():
                        contexts_list = [c.strip() for c in str(row["contexts"]).split("|||") if c.strip()]
                        if not contexts_list:
                            contexts_list = [row["answer"]]
                        ragas_data.append({
                            "question":     row["question"],
                            "answer":       row["answer"],
                            "contexts":     contexts_list,
                            "ground_truth": row["ground_truth"],
                        })

                    ragas_dataset = HFDataset.from_list(ragas_data)

                    llm        = ChatOpenAI(model="gpt-4o-mini", openai_api_key=ragas_openai_key)
                    embeddings = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=ragas_openai_key)

                    with st.spinner(f"🧠 RAGAS đang chấm điểm {len(ragas_data)} câu hỏi..."):
                        result = evaluate(
                            dataset=ragas_dataset,
                            metrics=[faithfulness, answer_relevancy, context_recall, context_precision],
                            llm=llm,
                            embeddings=embeddings,
                            raise_exceptions=False,
                        )

                    # ── Display Results ─────────────────────────────────────
                    st.success("🎉 Evaluation hoàn tất!")
                    st.markdown("### 📈 Kết quả tổng hợp")

                    # Convert to pandas first — works với mọi version RAGAS (0.1.x và 0.2.x)
                    detail_df = result.to_pandas()

                    def safe_mean(col):
                        """Lấy mean score từ DataFrame, trả về 0 nếu cột không tồn tại"""
                        if col in detail_df.columns:
                            return float(detail_df[col].dropna().mean())
                        return 0.0

                    scores = {
                        "Faithfulness":      safe_mean("faithfulness"),
                        "Answer Relevancy":  safe_mean("answer_relevancy"),
                        "Context Recall":    safe_mean("context_recall"),
                        "Context Precision": safe_mean("context_precision"),
                    }

                    def score_color(s):
                        if s >= 0.8: return "score-good"
                        if s >= 0.5: return "score-ok"
                        return "score-poor"

                    def score_emoji(s):
                        if s >= 0.8: return "🟢"
                        if s >= 0.5: return "🟡"
                        return "🔴"

                    cols = st.columns(4)
                    metric_info = {
                        "Faithfulness":      "Câu trả lời có bịa thông tin không?",
                        "Answer Relevancy":  "Câu trả lời có đúng trọng tâm không?",
                        "Context Recall":    "Retrieved đủ ngữ cảnh không?",
                        "Context Precision": "Context có nhiều nhiễu không?",
                    }
                    for col, (name, score) in zip(cols, scores.items()):
                        color = score_color(score)
                        emoji = score_emoji(score)
                        col.markdown(f"""
                        <div class="metric-card">
                            <div style="font-size:0.85em; color:#aaa;">{emoji} {name}</div>
                            <div class="metric-score {color}">{score:.3f}</div>
                            <div style="font-size:0.75em; color:#888; margin-top:4px;">{metric_info[name]}</div>
                        </div>""", unsafe_allow_html=True)

                    overall = sum(scores.values()) / len(scores)
                    grade   = "🟢 Excellent" if overall >= 0.8 else "🟡 Good" if overall >= 0.6 else "🟠 Fair" if overall >= 0.4 else "🔴 Poor"
                    st.markdown(f"\n**Overall Score:** {overall:.3f} &nbsp;&nbsp; **Grade:** {grade}")

                    # Progress bars
                    st.markdown("### 📊 Chi tiết từng metric")
                    for name, score in scores.items():
                        st.metric(label=name, value=f"{score:.4f} ({score*100:.1f}%)")
                        st.progress(min(score, 1.0))

                    # Per-question table
                    # RAGAS v0.2+ không bao gồm cột 'question' trong result df → ghép thủ công
                    st.markdown("### 🔬 Kết quả từng câu hỏi")

                    # Lấy các cột metric có sẵn trong detail_df
                    metric_cols = [c for c in ["faithfulness", "answer_relevancy", "context_recall", "context_precision"] if c in detail_df.columns]

                    # Ghép question, answer, ground_truth từ ragas_data gốc
                    questions_df = pd.DataFrame([{
                        "question":     r["question"],
                        "ground_truth": r["ground_truth"],
                    } for r in ragas_data]).reset_index(drop=True)

                    display_df = pd.concat(
                        [questions_df, detail_df[metric_cols].reset_index(drop=True)],
                        axis=1
                    )

                    st.dataframe(display_df, use_container_width=True)

                    # Download results (dùng display_df có đủ question + metrics)
                    result_csv = display_df.to_csv(index=False).encode("utf-8-sig")
                    st.download_button(
                        "💾 Download kết quả CSV",
                        data=result_csv,
                        file_name="ragas_results.csv",
                        mime="text/csv",
                    )

                    # Top worst questions (dùng display_df đã ghép question)
                    if "faithfulness" in display_df.columns:
                        st.markdown("### ⚠️ Câu hỏi có Faithfulness thấp nhất (cần cải thiện)")
                        worst = display_df.sort_values("faithfulness").head(5)
                        for _, row in worst.iterrows():
                            faith_val = row.get("faithfulness", 0) or 0
                            with st.expander(f'❓ {str(row["question"])[:80]}... — Faithfulness: {faith_val:.3f}'):
                                st.write("**Ground Truth:**", row.get("ground_truth", ""))

                except ImportError as e:
                    st.error(f"❌ Thiếu thư viện RAGAS: {e}. Hãy cài: `pip install ragas langchain-openai datasets`")
                except Exception as e:
                    st.error(f"❌ RAGAS evaluation lỗi: {str(e)}")

# ============================================================
# Footer
# ============================================================
st.markdown("---")
st.caption("Powered by RAG Anything · LightRAG · RAGAS · GPT-4o")
