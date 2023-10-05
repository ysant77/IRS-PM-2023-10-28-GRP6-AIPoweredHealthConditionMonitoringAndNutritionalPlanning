function generateSessionId() {
    return Date.now() + Math.floor(Math.random() * 1000);
}


let sessionId = generateSessionId();
let socket = new WebSocket("ws://" + window.location.host + "/ws/chat/" + sessionId + "/");


socket.onmessage = function(event) {
    let data = JSON.parse(event.data);
    let chatBox = document.getElementById("chatBox");
    chatBox.innerHTML += "<div><strong>Bot:</strong> " + data["message"] + "</div>";
};

function sendMessage() {
    let userMessageInput = document.getElementById("userMessage");
    let chatBox = document.getElementById("chatBox");
    let message = userMessageInput.value;
    
    chatBox.innerHTML += "<div><strong>You:</strong> " + message + "</div>";

    socket.send(JSON.stringify({
        "message": message
    }));
    userMessageInput.value = "";
}
