function generateText() {
    const prompt = document.getElementById("prompt").value;
    const temperature = document.getElementById("temperature").value;
    const max_tokens = document.getElementById("max_tokens").value;
    const outputDiv = document.getElementById("output");

    if (!prompt) return;

    // Clear previous output
    outputDiv.innerText = "Generating...";

    fetch("/generate", {
        method: "POST",
        headers: {
            "Content-Type": "application/json"
        },
        body: JSON.stringify({
            prompt: prompt,
            temperature: temperature,
            max_tokens: max_tokens
        })
    })
    .then(response => response.json())
    .then(data => {
        outputDiv.innerText = data.response;
    })
    .catch(() => {
        outputDiv.innerText = "Something went wrong!";
    });

    // Clear textarea after sending
    document.getElementById("prompt").value = "";
}