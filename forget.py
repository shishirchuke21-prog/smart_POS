"""List or remove saved people. Usage: python forget.py [name]"""
import sys

from common import load_db, save_db

db = load_db()
if len(sys.argv) < 2:
    print("Known people:", list(db) or "none")
elif sys.argv[1] in db:
    del db[sys.argv[1]]
    save_db(db)
    print(f"Removed {sys.argv[1]}")
else:
    print(f"{sys.argv[1]} not found. Known: {list(db)}")
