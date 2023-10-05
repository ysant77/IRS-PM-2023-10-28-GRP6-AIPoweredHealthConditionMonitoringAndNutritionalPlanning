function generateSessionId() {
    return Date.now() + Math.floor(Math.random() * 1000);
}


let sessionId = generateSessionId();
let socket = new WebSocket("ws://" + window.location.host + "/ws/chat/" + sessionId + "/");


socket.onmessage = function(event) {
    let data = JSON.parse(event.data);
    let chatBox = document.getElementById("chatBox");
    
    // Create a new div for the bot's message
    let botMessageDiv = document.createElement("div");
    botMessageDiv.innerHTML = "<strong>Bot:</strong> ";
    
    // Check if the bot's message is a table by looking for the <table> tag
    if (data["bot_message"].includes("<table")) {
        let tempDiv = document.createElement("div");
        tempDiv.innerHTML = data["bot_message"];
        
        // Assuming DOMPurify is loaded, sanitize the HTML
        let sanitizedContent = DOMPurify.sanitize(tempDiv.innerHTML);
        botMessageDiv.innerHTML += sanitizedContent;
    } else {
        botMessageDiv.innerText += data["bot_message"];
    }

    chatBox.appendChild(botMessageDiv);
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
