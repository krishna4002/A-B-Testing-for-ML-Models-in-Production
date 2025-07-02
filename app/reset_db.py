import sqlite3

def reset_predictions():
    conn = sqlite3.connect("logs/predict.db")
    cursor = conn.cursor()
    cursor.execute("DELETE FROM predictions")
    conn.commit()
    conn.close()
    print("✅ All predictions deleted successfully.")

if __name__ == "__main__":
    reset_predictions()
