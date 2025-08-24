// Binary animation
function randomBinary(length) {
    let s = '';
    for (let i = 0; i < length; i++) {
        s += Math.random() > 0.5 ? '0' : '1';
    }
    return s;
}
setInterval(() => {
    document.getElementById('binary').textContent = randomBinary(40);
}, 120);

const chatlog = document.getElementById('chatlog');
const aiDetails = document.getElementById('aiDetails');
let sessionActive = true;
const micBtn = document.getElementById('micBtn');
if ('webkitSpeechRecognition' in window || 'SpeechRecognition' in window) {
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    const recognition = new SpeechRecognition();
    recognition.lang = 'en-US';
    recognition.continuous = false;
    recognition.interimResults = false;

    micBtn.onclick = function() {
        recognition.start();
        micBtn.disabled = true;
        micBtn.textContent = '🎙️ Listening...';
    };

    recognition.onresult = function(event) {
        const transcript = event.results[0][0].transcript;
        document.getElementById('userInput').value = transcript;
        micBtn.disabled = false;
        micBtn.textContent = '🎤';
        sendMessage();
    };
    recognition.onerror = function() {
        micBtn.disabled = false;
        micBtn.textContent = '🎤';
    };
} else {
    micBtn.style.display = 'none';
}

function sendMessage() {
    if (!sessionActive) return;
    const input = document.getElementById('userInput');
    const msg = input.value.trim();
    if (!msg) return;
    chatlog.innerHTML += `<div><b>You:</b> ${msg}</div>`;
    input.value = '';
    fetch('http://localhost:5000/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: msg })
    })
    .then(res => {
        if (!res.ok) throw new Error('Backend error');
        return res.json();
    })
    .then(data => {
        let aiResp = data.response;
        let entities = '';
        let sentiment = '';
        if (aiResp.includes('Entities:')) {
            let entMatch = aiResp.match(/Entities: (\[.*?\])/);
            if (entMatch) entities = entMatch[1];
        }
        if (aiResp.includes('Sentiment:')) {
            let sentMatch = aiResp.match(/Sentiment: (\[.*?\])/);
            if (sentMatch) sentiment = sentMatch[1];
        }
        chatlog.innerHTML += `<div><b>Machine-Man:</b> <span style=\"color:#4f8cff\">${aiResp}</span></div>`;
        aiDetails.innerHTML = `<b>Entities:</b> ${entities}<br><b>Sentiment:</b> ${sentiment}`;
        chatlog.scrollTop = chatlog.scrollHeight;
        if (data.shutdown) {
            input.disabled = true;
            input.placeholder = "Lakshmi has ended the session.";
            document.querySelector('button').disabled = true;
            sessionActive = false;
        }
    })
    .catch((err) => {
        chatlog.innerHTML += `<div><b>Machine-Man:</b> <span style=\"color:#f00\">Error connecting to backend. Please ensure the backend is running.</span></div>`;
        sessionActive = false;
        document.getElementById('userInput').disabled = true;
        document.querySelector('button').disabled = true;
    });
}
document.getElementById('userInput').addEventListener('keydown', function(e) {
    if (e.key === 'Enter') sendMessage();
});