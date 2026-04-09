import os
import re
import math
import hashlib
import logging
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import nltk
from nltk.stem.snowball import GermanStemmer
from lexical_diversity import lex_div as ld
import subprocess
import json
from faster_whisper import WhisperModel
#import ffprobe
import ffmpeg

stemmer = GermanStemmer();

from flask import Flask, request, render_template, jsonify
import mysql.connector

model = WhisperModel("base", compute_type="int8")

app = Flask(__name__)
UPLOAD_FOLDER = "/data/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
DOWNLOAD_DIR = "/data/downloads/youtube"
os.makedirs(DOWNLOAD_DIR, exist_ok=True)

CURRENT_DB = os.getenv("DB_NAME")

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

def normalize_dict(d):
    total = sum(d.values()) or 1
    return {k: v / total for k, v in d.items()}

def process_text(text):
    text = text.lower()

    flt = ld.flemmatize(text)
    mtld = ld.mtld(flt)

    words = re.findall(r'\b\w+\b', text)

    normalized = [normalize_word(w) for w in words]

    counts = Counter(normalized)
    return counts, len(words), len(counts), mtld

def save_to_db(file_hash, filename, counts, total, unique, mtld):
    conn = get_db()
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO files (file_hash, filename, total_words, unique_words, mtld)
        VALUES (%s, %s, %s, %s, %s)
        ON DUPLICATE KEY UPDATE
            filename = VALUES(filename),
            total_words = VALUES(total_words),
            unique_words = VALUES(unique_words),
            mtld = VALUES(mtld)
    """, (file_hash, filename, total, unique, mtld))

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

    counts, total, unique, mtld = process_text(text)
    file_hash = hash_file(filepath)
    filename = os.path.basename(filepath)

    save_to_db(file_hash, filename, counts, total, unique, mtld)

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

def get_video_ids(url):
    result = subprocess.run(
        ["yt-dlp", "--flat-playlist", "-J", url],
        capture_output=True,
        text=True
    )

    data = json.loads(result.stdout)

    # einzelnes Video fallback
    if "entries" not in data:
        return [data["id"]]

    return [entry["id"] for entry in data["entries"] if entry]

def get_videos_with_limit(url, limit=20):
    result = subprocess.run(
        [
            "yt-dlp",
            "--flat-playlist",
            "--playlist-end", str(limit),
            "-J",
            url
        ],
        capture_output=True,
        text=True
    )

    data = json.loads(result.stdout)

    return [entry["id"] for entry in data.get("entries", []) if entry]

def get_video_info(video_id):
    result = subprocess.run(
        ["yt-dlp", "-j", f"https://youtube.com/watch?v={video_id}"],
        capture_output=True,
        text=True
    )

    data = json.loads(result.stdout)

    return {
        "title": data.get("title"),
        "channel": data.get("channel"),
        "uploader": data.get("uploader"),
        "upload_date": data.get("upload_date")
    }

def get_latest_video(url):
    result = subprocess.run(
        [
            "yt-dlp",
            "--flat-playlist",
            "--playlist-end", "1",
            "-J",
            url
        ],
        capture_output=True,
        text=True
    )

    data = json.loads(result.stdout)

    entry = data["entries"][0]
    return entry["id"]

def download_audio(video_id):
    output = f"{DOWNLOAD_DIR}/{video_id}.mp3"

    subprocess.run([
        "yt-dlp",
        "-x",
        "--download-archive", f"{DOWNLOAD_DIR}/archive.txt",
        "--audio-format", "mp3",
        "-o", f"{DOWNLOAD_DIR}/{video_id}.%(ext)s",
        f"https://youtube.com/watch?v={video_id}"
    ])

    return output

def transcribe(file):
    segments, _ = model.transcribe(file)

    result = []
    for i, seg in enumerate(segments):
        app.logger.info(f"[{i}] {seg.start:.2f}s -> {seg.end:.2f}s")
        result.append(seg.text)

    return " ".join(result)

def clean_text(text):
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def import_youtube(url, mode="all"):
    app.logger.info(f"YouTube import {url} with mode {mode}")

    video_ids = []
    if mode == "latest":
        video_ids = [get_latest_video(url)]
    elif mode == "limit":
        video_ids = get_videos_with_limit(url, limit=50)
    else:
        video_ids = get_video_ids(url)

    for vid in video_ids:
        meta = get_video_info(vid)
        app.logger.info("Processing " + meta["title"])

        try:
            audio = download_audio(vid)
            app.logger.info('Started transcribe')
            text = transcribe(audio)
            app.logger.info('Ended transcribe')
            text = clean_text(text)
            path = UPLOAD_FOLDER+"/"+meta["channel"] + " - " + meta["title"] +".txt"
            app.logger.info('Saving to file '+path)

            with open(path, "w", encoding="utf-8") as f:
                f.write(text)
                f.close()

            app.logger.info('Processing file...')
            process_file(path)
            app.logger.info('Processed file.')

            # Speicher sparen
            os.remove(audio)

        except Exception as e:
            print(f"Error with {vid}:", e)

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

IMPORT_RUNNING = False

@app.route("/admin/youtube_import", methods=["POST"])
def youtube_import():
    global IMPORT_RUNNING

    if IMPORT_RUNNING:
        return {"error": "import already running"}, 400

    IMPORT_RUNNING = True

    url = request.json.get("url")
    mode = request.json.get("mode", "all")

    t = threading.Thread(target=import_youtube, args=(url,mode,))
    t.start()

    return {"status": "started"}

# === PUBLIC ===

CATEGORIES = {
    "politik": [
        "regierung","politik","staat","wahl","partei","gesetz","demokratie",
        "kanzler","minister","bundestag","eu","verfassung"
    ],
    "gesellschaft": [
        "gesellschaft","menschen","kultur","sozial","gemeinschaft","leben",
        "werte","normen","alltag","identität"
    ],
    "kritisch": [
        "problem","kritik","fehler","schlecht","versagen","risiko",
        "krise","konflikt","schwierigkeit"
    ],
    "humor": [
        "witz","lustig","haha","lol","spaß","ironie","sarkasmus",
        "witzig","lachen"
    ],
    "wissenschaft": [
        "studie","analyse","daten","forschung","theorie","modell",
        "experiment","beweis","statistik"
    ],
    "wirtschaft": [
        "markt","geld","wirtschaft","unternehmen","preis","kosten",
        "investition","profit","wachstum"
    ],
    "technologie": [
        "technik","software","hardware","ai","algorithmus","internet",
        "system","entwicklung"
    ],
    "emotional": [
        "liebe","angst","wut","freude","traurig","glücklich",
        "emotional","gefühl"
    ],
    "kontrovers": [
        "umstritten","provokant","skandal","kritisch","extrem",
        "debattiert","polarisierend"
    ],

    # 🔥 neue
    "philosophisch": [
        "sinn","existenz","denken","bewusstsein","realität",
        "wahrheit","ethik","moral"
    ],
    "persönlich": [
        "mir","mein","erfahrung","leben","fühle",
        "denke","glaube"
    ],
    "erzählend": [
        "geschichte","erzählen","damals","passiert",
        "erlebnis","story"
    ],
    "argumentativ": [
        "argument","begründung","deshalb","weil",
        "logisch","schlussfolgerung"
    ]
}

# === PUBLIC ROUTES ===

@app.route("/public")
def public_ui():
    return render_template("public.html")

@app.route("/public/overview")
def public_overview():

    conn = mysql.connector.connect(
        host=DB_CONFIG["host"],
        user=DB_CONFIG["user"],
        password=DB_CONFIG["password"]
    )
    cursor = conn.cursor()

    cursor.execute("SHOW DATABASES")
    dbs = [d[0] for d in cursor.fetchall() if d[0] not in DB_BLACKLIST]

    result = []

    print("test")

    #selected_dbs = request.json.get("dbs", Array.array(dbs))

    print("test")

    for db in dbs:
        try:
            c = mysql.connector.connect(**{**DB_CONFIG, "database": db})
            cur = c.cursor()

            cur.execute("SELECT COUNT(*), SUM(total_words) FROM files")
            count, total = cur.fetchone()

            result.append({
                "db": db,
                "files": count or 0,
                "words": total or 0
            })

            cur.close()
            c.close()
        except:
            continue

    cursor.close()
    conn.close()

    return jsonify(result)

@app.route("/public/compare/<db>")
def compare(db):
    conn = mysql.connector.connect(**{**DB_CONFIG, "database": db})
    cursor = conn.cursor(dictionary=True)

    cursor.execute("""
        SELECT filename, total_words, unique_words
        FROM files
    """)

    data = cursor.fetchall()

    cursor.close()
    conn.close()

    return jsonify(data)

@app.route("/public/categories/<db>/<filename>")
def categories(db, filename):
    conn = mysql.connector.connect(**{**DB_CONFIG, "database": db})
    cursor = conn.cursor()

    cursor.execute("""
        SELECT word, count
        FROM words w
        JOIN files f ON w.file_id = f.id
        WHERE f.filename = %s
    """, (filename,))

    words = cursor.fetchall()

    scores = {k: 0 for k in CATEGORIES}

    for word, count in words:
        for cat, keywords in CATEGORIES.items():
            if word in keywords:
                scores[cat] += count

    cursor.close()
    conn.close()

    return jsonify(scores)

@app.route("/public/db_stats", methods=["POST"])
def db_stats():
    selected_dbs = request.json.get("dbs", [])

    if not selected_dbs:
        return jsonify([])

    conn = mysql.connector.connect(
        host=DB_CONFIG["host"],
        user=DB_CONFIG["user"],
        password=DB_CONFIG["password"]
    )
    cursor = conn.cursor()

    cursor.execute("SHOW DATABASES")
    dbs = [d[0] for d in cursor.fetchall() if d[0] not in DB_BLACKLIST]

    results = []

    for db in selected_dbs:
        try:
            c = mysql.connector.connect(**{**DB_CONFIG, "database": db})
            cur = c.cursor()

            cur.execute("""
                SELECT 
                    AVG(mtld),
                    AVG(unique_words / total_words),
                    AVG(total_words),
                    COUNT(*)
                FROM files
            """)

            mtld, diversity, avg_len, count = cur.fetchone()

            results.append({
                "db": db,
                "mtld": float(mtld or 0),
                "diversity": float(diversity or 0),
                "avg_words": float(avg_len or 0),
                "files": count
            })

            cur.close()
            c.close()
        except:
            continue

    cursor.close()
    conn.close()

    return jsonify(results)

@app.route("/public/categories_db/<db>")
def categories_db(db):
    conn = mysql.connector.connect(**{**DB_CONFIG, "database": db})
    cursor = conn.cursor()

    cursor.execute("""
        SELECT word, SUM(count)
        FROM words
        GROUP BY word
    """)

    words = cursor.fetchall()

    scores = {k: 0 for k in CATEGORIES}

    for word, count in words:
        for cat, keywords in CATEGORIES.items():
            if word in keywords:
                scores[cat] += count

    cursor.close()
    conn.close()

    scores = normalize_dict(scores)

    return jsonify(scores)

@app.route("/public/tfidf_db/<db>")
def tfidf_db(db):
    import math

    conn = mysql.connector.connect(**{**DB_CONFIG, "database": db})
    cursor = conn.cursor()

    cursor.execute("SELECT COUNT(*) FROM files")
    total_docs = cursor.fetchone()[0]

    cursor.execute("""
        SELECT word, SUM(count)
        FROM words
        GROUP BY word
    """)
    words = cursor.fetchall()

    results = []

    cursor.execute("""
        SELECT word, SUM(count) as tf, COUNT(DISTINCT file_id) as df
        FROM words
        GROUP BY word
    """)

    rows = cursor.fetchall()

    results = []
    for word, tf, df in rows:
        score = float(tf) * math.log(total_docs / (df or 1))
        results.append((word, score))

    results = sorted(results, key=lambda x: x[1], reverse=True)[:20]

    cursor.close()
    conn.close()

    return jsonify(results)

@app.route("/public/dna/<db>")
def podcast_dna(db):
    conn = mysql.connector.connect(**{**DB_CONFIG, "database": db})
    cursor = conn.cursor()

    cursor.execute("""
        SELECT 
            AVG(mtld),
            AVG(unique_words / total_words),
            AVG(word_length)
        FROM files f
        JOIN words w ON f.id = w.file_id
    """)

    mtld, diversity, word_len = cursor.fetchone()

    # Kategorien
    cursor.execute("SELECT word, SUM(count) FROM words GROUP BY word")
    words = cursor.fetchall()

    cat_scores = {k: 0 for k in CATEGORIES}
    for word, count in words:
        for cat, keys in CATEGORIES.items():
            if word in keys:
                cat_scores[cat] += count
    
    cat_scores = normalize_dict(cat_scores)

    cursor.close()
    conn.close()

    return jsonify({
        "mtld": float(mtld or 0),
        "diversity": float(diversity or 0),
        "word_length": float(word_len or 0),
        "categories": cat_scores
    })

@app.route("/public/tfidf_cross", methods=["POST"])
def tfidf_cross():
    selected_dbs = request.json.get("dbs", [])

    if not selected_dbs:
        return jsonify([])

    conn = mysql.connector.connect(
        host=DB_CONFIG["host"],
        user=DB_CONFIG["user"],
        password=DB_CONFIG["password"]
    )
    cursor = conn.cursor()

    cursor.execute("SHOW DATABASES")
    dbs = [d[0] for d in cursor.fetchall() if d[0] not in DB_BLACKLIST]

    db_word_counts = {}
    df_counts = {}
    total_dbs = len(dbs)

    # 🔹 Daten sammeln
    for db in selected_dbs:
        try:
            c = mysql.connector.connect(**{**DB_CONFIG, "database": db})
            cur = c.cursor()

            cur.execute("""
                SELECT word, SUM(count)
                FROM words
                GROUP BY word
            """)

            words = cur.fetchall()
            db_word_counts[db] = dict(words)

            # Document Frequency (über DBs)
            for word, _ in words:
                df_counts[word] = df_counts.get(word, 0) + 1

            cur.close()
            c.close()
        except:
            continue

    # 🔥 TF-IDF berechnen
    result = {}

    for db, words in db_word_counts.items():
        scores = []

        for word, tf in words.items():
            df = df_counts.get(word, 1)
            score = float(tf) * math.log(total_dbs / df)

            scores.append((word, score))

        scores = sorted(scores, key=lambda x: x[1], reverse=True)[:20]
        result[db] = scores

    return jsonify(result)

# === START ===

if __name__ == "__main__":
    init_db_schema(CURRENT_DB)
    app.run(host="0.0.0.0", port=5000)

if __name__ != '__main__':
    gunicorn_logger = logging.getLogger('gunicorn.error')
    app.logger.handlers = gunicorn_logger.handlers
    app.logger.setLevel(gunicorn_logger.level)