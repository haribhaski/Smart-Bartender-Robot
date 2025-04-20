from flask import Flask, request, jsonify
import json
import os
from filelock import FileLock  # Install with `pip install filelock`

app = Flask(__name__)
DATA_FILE = 'data.json'
LOCK_FILE = 'data.json.lock'

# Initialize data.json if it doesn’t exist
if not os.path.exists(DATA_FILE):
    with open(DATA_FILE, 'w') as f:
        json.dump({"orders": []}, f)

@app.route('/')
def serve_html():
    return app.send_static_file('index.html')

@app.route('/data.json')
def serve_data():
    with FileLock(LOCK_FILE):  # Lock the file during read
        with open(DATA_FILE, 'r') as f:
            return jsonify(json.load(f))

@app.route('/add_order', methods=['POST'])
def add_order():
    new_order = request.get_json()
    try:
        with FileLock(LOCK_FILE):  # Lock the file during write
            with open(DATA_FILE, 'r') as f:
                data = json.load(f)
            data['orders'].append(new_order)
            with open(DATA_FILE, 'w') as f:
                json.dump(data, f)
                f.flush()  # Ensure write is committed to disk
                os.fsync(f.fileno())  # Force disk sync
        return jsonify({"success": True})
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"success": False}), 500

if __name__ == '__main__':
    app.static_folder = '.'
    app.run(debug=True, port=5000)