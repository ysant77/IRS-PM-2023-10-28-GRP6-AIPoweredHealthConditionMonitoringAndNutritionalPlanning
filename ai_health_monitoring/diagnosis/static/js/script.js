function generateSessionId() {
    return Date.now() + Math.floor(Math.random() * 1000);
}


let sessionId = generateSessionId();
let socket = new WebSocket("ws://" + window.location.host + "/ws/chat/" + sessionId + "/");

socket.onmessage = function(event) {
    let data = JSON.parse(event.data);
    let chatBox = document.getElementById("chatBox");
    
    let botMessageDiv = document.createElement("div");

    if(data.type === "multi_select") {
        let botMessage = document.createElement("div");
        botMessage.innerHTML = "<strong>Bot:</strong> " + data["bot_message"];
        
        let dynamicContentDiv = document.createElement("div");
        dynamicContentDiv.id = 'dynamicContent';  // Setting the ID for the div

        for(let symptom of data.options) {
            let label = document.createElement("label");
            let checkbox = document.createElement("input");
            checkbox.type = 'checkbox';
            checkbox.value = symptom;
            label.appendChild(checkbox);
            label.append(symptom);
            dynamicContentDiv.appendChild(label);
        }

        botMessageDiv.appendChild(botMessage);
        botMessageDiv.appendChild(dynamicContentDiv);
    } else {
        botMessageDiv.innerHTML += "<strong>Bot:</strong> ";
        if (data["bot_message"].includes("<table")) {
            let tempDiv = document.createElement("div");
            tempDiv.innerHTML = data["bot_message"];
            
            let sanitizedContent = DOMPurify.sanitize(tempDiv.innerHTML);
            botMessageDiv.innerHTML += sanitizedContent;
        } else {
            botMessageDiv.innerText += data["bot_message"];
        }
    }

    chatBox.appendChild(botMessageDiv);
};


function sendMessage() {
    let userMessageInput = document.getElementById("userMessage");
    let chatBox = document.getElementById("chatBox");
    let message = userMessageInput.value;
    if(document.getElementById("dynamicContent")) {
        let checkboxes = document.querySelectorAll("#dynamicContent input[type='checkbox']:checked");
        let selectedOptions = [];
        for(let checkbox of checkboxes) {
            selectedOptions.push(checkbox.value);
        }
        if (selectedOptions.length !== 0){
        message = selectedOptions.join(",");
        }
    }
    let radios = document.getElementsByName("askMealPlan");
        
        if(radios){
        for(let i = 0; i < radios.length; i++) {
            if(radios[i].checked) {
                message = radios[i].value;
                radios[i].checked = false;
                break;
            }
        }
    }
    chatBox.innerHTML += "<div><strong>You:</strong> " + message + "</div>";
    socket.send(JSON.stringify({
        "message": message
    }));
    userMessageInput.value = "";
}
