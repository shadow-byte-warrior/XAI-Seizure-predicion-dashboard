import sqlite3
import json
import os
from datetime import datetime

DB_PATH = "neuroguard.db"

def init_db():
    """Initializes the SQLite database."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Analyses table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS analyses (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            filename TEXT,
            model_name TEXT,
            risk_score REAL,
            detections INTEGER,
            total_segments INTEGER,
            accuracy REAL,
            top_channels TEXT,
            report_content TEXT
        )
    ''')
    
    # Settings table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS settings (
            key TEXT PRIMARY KEY,
            value TEXT
        )
    ''')
    
    conn.commit()
    conn.close()

def save_analysis(filename, model_name, risk_score, detections, total_segments, accuracy, top_channels, report_content):
    """Saves an analysis result to the database."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    top_channels_json = json.dumps(top_channels)
    
    cursor.execute('''
        INSERT INTO analyses (timestamp, filename, model_name, risk_score, detections, total_segments, accuracy, top_channels, report_content)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (timestamp, filename, model_name, risk_score, detections, total_segments, accuracy, top_channels_json, report_content))
    
    conn.commit()
    conn.close()

def get_history():
    """Retrieves all past analyses."""
    if not os.path.exists(DB_PATH):
        return []
        
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    cursor.execute('SELECT * FROM analyses ORDER BY id DESC')
    rows = cursor.fetchall()
    
    history = [dict(row) for row in rows]
    for item in history:
        item['top_channels'] = json.loads(item['top_channels'])
        
    conn.close()
    return history

def delete_analysis(analysis_id):
    """Deletes an analysis from the database."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('DELETE FROM analyses WHERE id = ?', (analysis_id,))
    conn.commit()
    conn.close()

def set_setting(key, value):
    """Sets a persistent setting."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('INSERT OR REPLACE INTO settings (key, value) VALUES (?, ?)', (key, value))
    conn.commit()
    conn.close()

def get_setting(key, default=None):
    """Gets a persistent setting."""
    if not os.path.exists(DB_PATH):
        return default
        
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('SELECT value FROM settings WHERE key = ?', (key,))
    row = cursor.fetchone()
    conn.close()
    
    return row[0] if row else default
