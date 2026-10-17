"""
Attendance database (SQLite). One row per person per day.

Used by face_decet.py:   attendance_db.mark_present("Talha")

Command line (run from the same folder):
  python attendance_db.py today              # who is present today
  python attendance_db.py date 2026-10-17    # who was present on a date
  python attendance_db.py person Talha       # one person's full history
  python attendance_db.py summary            # days present + attendance % per person
  python attendance_db.py export [file.csv]  # export everything to CSV (opens in Excel)
"""
import os
import sys
import csv
import sqlite3
from datetime import datetime

# stored next to this script, so it is the same file no matter where you run from
DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "attendance.db")


def _connect():
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """CREATE TABLE IF NOT EXISTS attendance (
               id         INTEGER PRIMARY KEY AUTOINCREMENT,
               name       TEXT NOT NULL,
               date       TEXT NOT NULL,   -- YYYY-MM-DD
               first_seen TEXT NOT NULL,   -- check-in  (HH:MM:SS)
               last_seen  TEXT NOT NULL,   -- check-out (HH:MM:SS), updated while the person is seen
               UNIQUE(name, date)          -- one row per person per day
           )"""
    )
    return conn


def _query(sql, params=()):
    conn = _connect()
    try:
        return conn.execute(sql, params).fetchall()
    finally:
        conn.close()


def mark_present(name):
    """Marks `name` present today. Returns True if this is their FIRST sighting today."""
    now = datetime.now()
    day, t = now.strftime("%Y-%m-%d"), now.strftime("%H:%M:%S")
    conn = _connect()
    try:
        with conn:  # commits on success
            cur = conn.execute(
                "INSERT OR IGNORE INTO attendance (name, date, first_seen, last_seen) VALUES (?, ?, ?, ?)",
                (name, day, t, t),
            )
            if cur.rowcount == 1:
                return True
            conn.execute(
                "UPDATE attendance SET last_seen = ? WHERE name = ? AND date = ?", (t, name, day)
            )
            return False
    finally:
        conn.close()


def _print_rows(rows):
    if not rows:
        print("No records.")
        return
    print(f"{'Name':15s} {'Date':12s} {'Check-in':10s} {'Last seen':10s}")
    print("-" * 50)
    for name, day, first, last in rows:
        print(f"{name:15s} {day:12s} {first:10s} {last:10s}")


def show_date(day):
    rows = _query(
        "SELECT name, date, first_seen, last_seen FROM attendance WHERE date = ? ORDER BY first_seen", (day,)
    )
    _print_rows(rows)
    print(f"\n{len(rows)} present on {day}")


def show_today():
    show_date(datetime.now().strftime("%Y-%m-%d"))


def show_person(name):
    rows = _query(
        "SELECT name, date, first_seen, last_seen FROM attendance WHERE LOWER(name) = LOWER(?) ORDER BY date",
        (name,),
    )
    _print_rows(rows)


def show_summary():
    total_days = _query("SELECT COUNT(DISTINCT date) FROM attendance")[0][0]
    if not total_days:
        print("No records yet.")
        return
    rows = _query("SELECT name, COUNT(*) FROM attendance GROUP BY name ORDER BY COUNT(*) DESC, name")
    print(f"Days with any attendance recorded: {total_days}\n")
    print(f"{'Name':15s} {'Days present':13s} {'Attendance %'}")
    print("-" * 42)
    for name, days in rows:
        print(f"{name:15s} {days:<13d} {100 * days / total_days:.0f}%")


def export_csv(path="attendance_export.csv"):
    rows = _query("SELECT name, date, first_seen, last_seen FROM attendance ORDER BY date, first_seen")
    with open(path, "w", newline="", encoding="utf-8-sig") as f:   # utf-8-sig: Excel opens it cleanly
        w = csv.writer(f)
        w.writerow(["Name", "Date", "Check-in", "Last seen"])
        w.writerows(rows)
    print(f"Exported {len(rows)} rows to {os.path.abspath(path)}")


if __name__ == "__main__":
    args = sys.argv[1:]
    cmd = args[0] if args else ""
    if cmd == "today":
        show_today()
    elif cmd == "date" and len(args) == 2:
        show_date(args[1])
    elif cmd == "person" and len(args) == 2:
        show_person(args[1])
    elif cmd == "summary":
        show_summary()
    elif cmd == "export":
        export_csv(*args[1:2])
    else:
        print(__doc__)