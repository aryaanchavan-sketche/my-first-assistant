from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/api/chat', methods=['post'])
def chat():
    user_message = request.json.get('message', '')
    response = f'AI Responsr to:{user_message}'
    return jsonify({'response': response})

if __name__== '__main__':
    app.run(port=5000)