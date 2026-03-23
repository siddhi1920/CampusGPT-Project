import streamlit as st
from PyPDF2 import PdfReader
import pandas as pd
import asyncio
import logging
import time  

logging.basicConfig(
    filename="campusgpt.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
from sqlalchemy import create_engine, text as sql_text

# ===== LangChain imports =====
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq

# ================= DATABASE =================
DB_USER = "siddhi"
DB_PASS = "siddhi"
DB_HOST = "localhost"
DB_NAME = "campusgpt"

engine = create_engine(
    f"mysql+mysqlconnector://{DB_USER}:{DB_PASS}@{DB_HOST}/{DB_NAME}"
)

try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

# ================= DB FUNCTIONS =================
def validate_user(username, password):
    if username == "admin" and password == "admin123":
        return True, 0

    with engine.connect() as conn:
        row = conn.execute(
            sql_text("""
                SELECT user_id FROM users 
                WHERE username=:u AND password=:p
            """),
            {"u": username, "p": password}
        ).fetchone()

        return (True, row[0]) if row else (False, None)


def register_user(username, email, password):
    with engine.connect() as conn:
        existing = conn.execute(
            sql_text("SELECT user_id FROM users WHERE username=:u OR email=:e"),
            {"u": username, "e": email}
        ).fetchone()

        if existing:
            return False

        conn.execute(
            sql_text("""
                INSERT INTO users (username, email, password)
                VALUES (:u, :e, :p)
            """),
            {"u": username, "e": email, "p": password}
        )
        conn.commit()
        return True


def save_chat(user_id, q, a):
    if user_id == 0:
        return

    logging.info(f"User {user_id} asked question: {q}")  # 👈 ADD HERE

    with engine.connect() as conn:
        conn.execute(
            sql_text("""
                INSERT INTO chat_history(user_id,question,answer)
                VALUES(:u,:q,:a)
            """),
            {"u": user_id, "q": q, "a": a}
        )
        conn.commit()


def get_all_chat_history():
    with engine.connect() as conn:
        return conn.execute(
            sql_text("""
                SELECT id, user_id, question, answer, timestamp
                FROM chat_history
                ORDER BY timestamp DESC
            """)
        ).fetchall()


def get_user_chat_history(user_id):
    with engine.connect() as conn:
        return conn.execute(
            sql_text("""
                SELECT id, user_id, question, answer, timestamp
                FROM chat_history
                WHERE user_id = :uid
                ORDER BY timestamp DESC
            """),
            {"uid": user_id}
        ).fetchall()


# ================= PDF =================
def extract_text(pdfs):
    text = ""
    for pdf in pdfs:
        reader = PdfReader(pdf)
        for page in reader.pages:
            text += page.extract_text() or ""
    return text


def split_text(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )
    return splitter.split_text(text)


# ================= AI =================
def build_chain(mode="offline"):
    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0.3
    )

    pdf_prompt = PromptTemplate.from_template("""
You are an academic assistant.

Use the provided context to answer the question.

If the answer is partially available,
combine the context information logically.

Only say "The document does not contain sufficient information"
if the topic is completely unrelated.

Context:
{context}

Question:
{question}

Answer clearly and completely:
""")

    online_prompt = PromptTemplate.from_template("""
    Answer the question clearly and accurately.

    Question:
    {question}

    Answer:
    """)

    def run_chain(docs, question):
        if mode == "offline":
            context = "\n\n".join([doc.page_content for doc in docs])
            response = llm.invoke(
                pdf_prompt.format(context=context, question=question)
            )
        else:
            response = llm.invoke(
                online_prompt.format(question=question)
            )

        return response.content

    return run_chain



# ================= MAIN =================
def main():
    st.set_page_config("CampusGPT", page_icon="📄")

    # -------- SESSION INITIALIZATION --------
    if "page" not in st.session_state:
        st.session_state.page = "login"

    if "username" not in st.session_state:
        st.session_state.username = None

    if "user_id" not in st.session_state:
        st.session_state.user_id = None

    # ================= LOGIN =================
    if st.session_state.page == "login":
        st.title("🔐 CampusGPT Login")

        username = st.text_input("Username")
        password = st.text_input("Password", type="password")

        col1, col2 = st.columns(2)

        with col1:
            if st.button("Login"):
                ok, uid = validate_user(username, password)

                if ok:
                    logging.info(f"User {username} logged in successfully")
                    st.session_state.user_id = uid
                    st.session_state.username = username
                    st.session_state.page = "chat"
                    st.rerun()
                else:
                    logging.warning(f"Failed login attempt for {username}")
                    st.error("Invalid credentials")

        with col2:
            if st.button("Register"):
                st.session_state.page = "register"
                st.rerun()

    # ================= REGISTER =================
    elif st.session_state.page == "register":
        st.title("📝 Register Account")

        username = st.text_input("Username")
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")

        if st.button("Create Account"):
            if register_user(username, email, password):
                st.success("Account created successfully!")
                st.session_state.page = "login"
                st.rerun()
            else:
                st.error("Username or Email already exists")

        if st.button("Back"):
            st.session_state.page = "login"
            st.rerun()

    # ================= HISTORY =================
    elif st.session_state.page == "history":
        st.title("📜 My Chat History")

        rows = get_user_chat_history(st.session_state.user_id)

        if rows:
            df = pd.DataFrame(rows, columns=[
                "ID", "User ID", "Question", "Answer", "Timestamp"
            ])
            st.dataframe(df, use_container_width=True)

        if st.button("Back"):
            st.session_state.page = "chat"
            st.rerun()

    # ================= CHAT =================
    elif st.session_state.page == "chat":

        if not st.session_state.username:
            st.session_state.page = "login"
            st.rerun()

        
                # ================= ADMIN =================
        if st.session_state.username == "admin":

            st.title("🛠 Admin Dashboard")

            # ===== Dashboard Metrics =====
            total_users = pd.read_sql(
                "SELECT COUNT(*) as count FROM users",
                engine
            )["count"][0]

            total_chats = pd.read_sql(
                "SELECT COUNT(*) as count FROM chat_history",
                engine
            )["count"][0]

            col1, col2 = st.columns(2)
            col1.metric("Total Users", total_users)
            col2.metric("Total Chats", total_chats)

            st.divider()

            # ===== All Chat History =====
            rows = get_all_chat_history()

            if rows:
                df = pd.DataFrame(rows, columns=[
                    "ID", "User ID", "Question", "Answer", "Timestamp"
                ])
                st.dataframe(df, use_container_width=True)

                st.download_button(
                    "📥 Download Full History CSV",
                    df.to_csv(index=False),
                    "admin_history.csv",
                    "text/csv"
                )
            else:
                st.info("No chat history available.")

            # ===== Analytics =====
            st.subheader("📊 User Activity Chart")

            activity = pd.read_sql("""
                SELECT u.username, COUNT(c.id) AS total_chats
                FROM users u
                LEFT JOIN chat_history c ON u.user_id = c.user_id
                GROUP BY u.username
            """, engine)

            if not activity.empty:
                st.bar_chart(activity.set_index("username"))
            else:
                st.info("No user activity found.")

            # ===== Delete User =====
            st.subheader("🗑 Delete User")

            users = pd.read_sql(
                "SELECT user_id, username FROM users WHERE username != 'admin'",
                engine
            )

            if not users.empty:
                selected_user = st.selectbox(
                    "Select User to Delete",
                    users["username"]
                )

                if st.button("Delete Selected User"):
                    uid = int(
                        users[users["username"] == selected_user]["user_id"].values[0]
                    )

                    logging.info(f"Admin deleted user {selected_user}")

                    with engine.connect() as conn:
                        conn.execute(
                            sql_text("DELETE FROM users WHERE user_id=:uid"),
                            {"uid": uid}
                        )
                        conn.commit()

                    st.success("User deleted successfully")
                    st.rerun()
            else:
                st.info("No users available to delete.")

            if st.button("Logout", key="admin_logout"):
                st.session_state.clear()
                st.session_state.page = "login"
                st.rerun()

            return  # STOP HERE FOR ADMIN


        # ================= NORMAL USER =================
        st.sidebar.write(f"👤 {st.session_state.username}")

        mode = st.sidebar.radio(
            "🌐 Select Mode",
            ["Offline (PDF Only)", "Online (General AI)"]
        )

        if st.sidebar.button("📜 My History"):
            st.session_state.page = "history"
            st.rerun()

        if st.sidebar.button("Logout"):
            st.session_state.clear()
            st.session_state.page = "login"
            st.rerun()

        pdfs = st.sidebar.file_uploader(
            "Upload PDFs",
            type="pdf",
            accept_multiple_files=True
        )

        if st.sidebar.button("Process PDFs") and pdfs:
            pdf_text = extract_text(pdfs)
            chunks = split_text(pdf_text)

            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )

            st.session_state.vs = FAISS.from_texts(chunks, embeddings)
            st.success("PDFs processed successfully")

        st.subheader("💬 Ask Your Question")
        q = st.text_input("Type your question here...")

        if q:

            # ===== OFFLINE MODE =====
            if mode == "Offline (PDF Only)":

                if "vs" in st.session_state:

                    docs_with_score = st.session_state.vs.similarity_search_with_score(q, k=5)
                    docs = [doc for doc, score in docs_with_score]

                    if docs_with_score:
                        st.caption(f"🔎 Similarity Score: {round(docs_with_score[0][1],4)}")

                    with st.spinner("Generating answer..."):
                        start = time.time()
                        answer = build_chain("offline")(docs, q)
                        end = time.time()

                    save_chat(st.session_state.user_id, q, answer)
                    st.success(answer)
                    st.caption(f"⏱ Response Time: {round(end-start,2)} seconds")

                else:
                    st.warning("Please upload and process PDFs first.")

            # ===== ONLINE MODE =====
            else:
                with st.spinner("Generating answer..."):
                    start = time.time()
                    answer = build_chain("online")([], q)
                    end = time.time()

                save_chat(st.session_state.user_id, q, answer)
                st.success(answer)
                st.caption(f"⏱ Response Time: {round(end-start,2)} seconds")
                  

if __name__ == "__main__":
    main()
