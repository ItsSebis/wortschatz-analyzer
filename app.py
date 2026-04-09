import os
import re
import hashlib
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import nltk
from nltk.stem.snowball import GermanStemmer
from lexical_diversity import lex_div as ld

stemmer = GermanStemmer();

from flask import Flask, request, render_template, jsonify
import mysql.connector

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

CURRENT_DB = "sebi"

DB_CONFIG = {
    "host": os.getenv("DB_HOST"),
    "port": int(os.getenv("DB_PORT", 3306)),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
    "database": os.getenv("DB_NAME")
}

DB_BLACKLIST = {
    "information_schema",
    "mysql",
    "performance_schema",
    "sys"
}

# === UTILS ===

def get_db():
    config = DB_CONFIG.copy()
    config["database"] = CURRENT_DB
    return mysql.connector.connect(**config)

def hash_file(filepath):
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            sha256.update(chunk)
    return sha256.hexdigest()

def normalize_word(word):
    return stemmer.stem(word)

def process_text(text):
    text = text.lower()
    words = re.findall(r'\b\w+\b', text)

    normalized = [normalize_word(w) for w in words]

    counts = Counter(normalized)
    return counts, len(words), len(counts)

def save_to_db(file_hash, filename, counts, total, unique):
    conn = get_db()
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO files (file_hash, filename, total_words, unique_words)
        VALUES (%s, %s, %s, %s)
        ON DUPLICATE KEY UPDATE
            filename = VALUES(filename),
            total_words = VALUES(total_words),
            unique_words = VALUES(unique_words)
    """, (file_hash, filename, total, unique))

    conn.commit()

    cursor.execute("SELECT id FROM files WHERE file_hash=%s", (file_hash,))
    file_id = cursor.fetchone()[0]

    cursor.execute("DELETE FROM words WHERE file_id=%s", (file_id,))

    data = [(file_id, w, c, len(w)) for w, c in counts.items()]

    cursor.executemany(
        "INSERT INTO words (file_id, word, count, word_length) VALUES (%s, %s, %s, %s)",
        data
    )

    conn.commit()
    cursor.close()
    conn.close()

def process_file(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read()

    counts, total, unique = process_text(text)
    file_hash = hash_file(filepath)
    filename = os.path.basename(filepath)

    save_to_db(file_hash, filename, counts, total, unique)

def init_db_schema(db_name):
    conn = mysql.connector.connect(
        host=DB_CONFIG["host"],
        user=DB_CONFIG["user"],
        password=DB_CONFIG["password"],
        database=db_name
    )
    cursor = conn.cursor()

    # files table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS files (
            id INT AUTO_INCREMENT PRIMARY KEY,
            file_hash CHAR(64) UNIQUE,
            filename VARCHAR(255),
            total_words INT,
            unique_words INT,
            mtld FLOAT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # words table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS words (
            id INT AUTO_INCREMENT PRIMARY KEY,
            file_id INT,
            word VARCHAR(255),
            count INT,
            word_length INT,
            FOREIGN KEY (file_id) REFERENCES files(id) ON DELETE CASCADE,
            INDEX(word),
            INDEX(file_id),
            FULLTEXT(word)
        )
    """)

    conn.commit()
    cursor.close()
    conn.close()

def safe_db_name(name):
    if not re.match(r'^[a-zA-Z0-9_]+$', name):
        raise ValueError("Invalid DB name")
    return name

# === ROUTES ===

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload():
    files = request.files.getlist("files")

    paths = []
    for f in files:
        path = os.path.join(UPLOAD_FOLDER, f.filename)
        f.save(path)
        paths.append(path)

    with ThreadPoolExecutor(max_workers=4) as executor:
        executor.map(process_file, paths)

    return jsonify({"status": "ok", "files": len(paths)})

@app.route('/uploadExtension', methods=['POST'])
def uploadExtension():
    file = request.files.get('file')

    if not file:
        return "No file", 400

    content = file.read().decode('utf-8', errors='ignore')

    path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(path)
    process_file(path)

    return "OK"

@app.route("/search")
def search():
    word = request.args.get("word", "")

    conn = get_db()
    cursor = conn.cursor(dictionary=True)

    cursor.execute("""
        SELECT f.filename, w.word, w.count,
               MATCH(w.word) AGAINST(%s IN NATURAL LANGUAGE MODE) AS score
        FROM words w
        JOIN files f ON w.file_id = f.id
        WHERE MATCH(w.word) AGAINST(%s IN NATURAL LANGUAGE MODE)
        ORDER BY score DESC
        LIMIT 50
    """, (word, word))

    results = cursor.fetchall()

    cursor.close()
    conn.close()

    return jsonify(results)

@app.route("/files")
def files():
    conn = get_db()
    cursor = conn.cursor(dictionary=True)

    cursor.execute("""
        SELECT filename, total_words, unique_words, created_at, mtld, file_hash
        FROM files
        ORDER BY created_at DESC
    """)

    data = cursor.fetchall()

    cursor.close()
    conn.close()

    return jsonify(data)

@app.route("/stats")
def stats():
    conn = get_db()
    cursor = conn.cursor()

    # Gesamt Wörter
    cursor.execute("SELECT SUM(total_words) FROM files")
    total_words = cursor.fetchone()[0] or 0

    # globaler Wortschatz (alle einzigartigen Wörter über alle Dateien)
    cursor.execute("SELECT COUNT(DISTINCT word) FROM words")
    unique_words = cursor.fetchone()[0] or 0

    # Gesamt MTLD
    cursor.execute("SELECT SUM(mtld) / COUNT(filename) FROM files")
    mtld = round(cursor.fetchone()[0] or 0, 5)

    cursor.execute("""
        SELECT SUM(word_length * count) / SUM(count)
        FROM words
    """)
    avg = cursor.fetchone()[0] or 0

    cursor.close()
    conn.close()

    ttr = 0;
    if (total_words > 0):
        ttr = round(unique_words/total_words, 5)

    return jsonify({
        "total_words": total_words,
        "unique_words": unique_words,
        "ttr": ttr,
        "mtld": mtld
    })

@app.route("/top/global")
def top_global():
    conn = get_db()
    cursor = conn.cursor(dictionary=True)

    cursor.execute("""
        SELECT word, SUM(count) as total_count
        FROM words
        GROUP BY word
        ORDER BY total_count DESC
        LIMIT 20
    """)

    data = cursor.fetchall()

    cursor.close()
    conn.close()

    return jsonify(data)

@app.route("/top/file/<filename>")
def top_file(filename):
    conn = get_db()
    cursor = conn.cursor(dictionary=True)

    cursor.execute("""
        SELECT w.word, w.count
        FROM words w
        JOIN files f ON w.file_id = f.id
        WHERE f.filename = %s
        ORDER BY w.count DESC
        LIMIT 20
    """, (filename,))

    data = cursor.fetchall()

    cursor.close()
    conn.close()

    return jsonify(data)

@app.route("/avg/global")
def avg_global():
    conn = get_db()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT SUM(word_length * count) / SUM(count)
        FROM words
    """)

    avg = cursor.fetchone()[0] or 0

    cursor.close()
    conn.close()

    return jsonify({"avg_word_length": float(avg)})

@app.route("/avg/file/<filename>")
def avg_file(filename):
    conn = get_db()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT SUM(w.word_length * w.count) / SUM(w.count)
        FROM words w
        JOIN files f ON w.file_id = f.id
        WHERE f.filename = %s
    """, (filename,))

    avg = cursor.fetchone()[0] or 0

    cursor.close()
    conn.close()

    return jsonify({"avg_word_length": float(avg)})

@app.route("/tfidf/<filename>")
def tfidf(filename):
    conn = get_db()
    cursor = conn.cursor(dictionary=True)

    # Gesamtanzahl Dateien
    cursor.execute("SELECT COUNT(*) FROM files")
    total_docs = cursor.fetchone()["COUNT(*)"]

    # Wörter der Datei
    cursor.execute("""
        SELECT w.word, w.count
        FROM words w
        JOIN files f ON w.file_id = f.id
        WHERE f.filename = %s
    """, (filename,))
    words = cursor.fetchall()

    results = []

    for w in words:
        word = w["word"]
        tf = w["count"]

        # in wie vielen Dateien kommt das Wort vor
        cursor.execute("""
            SELECT COUNT(DISTINCT file_id) as df
            FROM words
            WHERE word = %s
        """, (word,))
        df = cursor.fetchone()["df"] or 1

        import math
        idf = math.log(total_docs / df)
        score = tf * idf

        results.append({
            "word": word,
            "tf": tf,
            "idf": idf,
            "score": score
        })

    # sortieren
    results = sorted(results, key=lambda x: x["score"], reverse=True)[:20]

    cursor.close()
    conn.close()

    return jsonify(results)

@app.route("/delete/<file_hash>", methods=["DELETE"])
def delete_file(file_hash):
    conn = get_db()
    cursor = conn.cursor()

    cursor.execute("DELETE FROM files WHERE file_hash = %s", (file_hash,))
    conn.commit()

    cursor.close()
    conn.close()

    return jsonify({"status": "deleted"})

@app.route("/chart/files")
def chart_files():
    conn = get_db()
    cursor = conn.cursor(dictionary=True)

    cursor.execute("""
        SELECT filename, total_words, unique_words
        FROM files
    """)

    data = cursor.fetchall()

    cursor.close()
    conn.close()

    return jsonify(data)

@app.route("/chart/wordlength/<filename>")
def wordlength(filename):
    conn = get_db()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT w.word_length, SUM(w.count)
        FROM words w
        JOIN files f ON w.file_id = f.id
        WHERE f.filename = %s
        GROUP BY w.word_length
        ORDER BY w.word_length
    """, (filename,))

    data = cursor.fetchall()

    cursor.close()
    conn.close()

    return jsonify(data)

@app.route("/db/switch", methods=["POST"])
def switch_db():
    global CURRENT_DB
    db_name = request.json.get("db")

    CURRENT_DB = db_name

    # 👉 sicherstellen, dass schema existiert
    init_db_schema(CURRENT_DB)

    return jsonify({"status": "ok", "current": CURRENT_DB})

@app.route("/db/create", methods=["POST"])
def create_db():
    db_name = safe_db_name(request.json.get("db"))

    conn = mysql.connector.connect(
        host=DB_CONFIG["host"],
        user=DB_CONFIG["user"],
        password=DB_CONFIG["password"]
    )
    cursor = conn.cursor()

    cursor.execute(f"CREATE DATABASE IF NOT EXISTS {db_name}")

    cursor.close()
    conn.close()

    # 👉 HIER WICHTIG
    init_db_schema(db_name)

    return jsonify({"status": "created"})

@app.route("/db/delete", methods=["POST"])
def delete_db():
    db_name = safe_db_name(request.json.get("db"))

    if db_name == CURRENT_DB:
        return jsonify({"error": "Cannot delete active DB"}), 400

    if db_name in DB_BLACKLIST:
        return jsonify({"error": "DB is protected"}), 403

    conn = mysql.connector.connect(
        host=DB_CONFIG["host"],
        user=DB_CONFIG["user"],
        password=DB_CONFIG["password"]
    )
    cursor = conn.cursor()

    cursor.execute(f"DROP DATABASE {db_name}")

    cursor.close()
    conn.close()

    return jsonify({"status": "deleted"})

@app.route("/db/list")
def list_db():
    conn = mysql.connector.connect(
        host=DB_CONFIG["host"],
        user=DB_CONFIG["user"],
        password=DB_CONFIG["password"]
    )
    cursor = conn.cursor()

    cursor.execute("SHOW DATABASES")
    dbs = [d[0] for d in cursor.fetchall()]

    cursor.close()
    conn.close()

    # Blacklist entfernen
    dbs = [db for db in dbs if db not in DB_BLACKLIST]

    # aktuelle DB nach oben ziehen
    dbs.sort()
    if CURRENT_DB in dbs:
        dbs.remove(CURRENT_DB)
        dbs.insert(0, CURRENT_DB)

    return jsonify({
        "current": CURRENT_DB,
        "databases": dbs
    })

# === START ===
if __name__ == "__main__":
    init_db_schema(CURRENT_DB)
    app.run