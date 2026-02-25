function generateText() {
    const prompt = document.getElementById("prompt").value;
    const output = document.getElementById("output");

    output.innerText = "Generating... ⏳";

    fetch("/generate", {
        method: "POST",
        headers: {
            "Content-Type": "application/json",
        },
        body: JSON.stringify({ prompt: prompt }),
    })
    .then(response => response.json())
    .then(data => {
        output.innerText = data.response;
    })
    .catch(() => {
        output.innerText = "Something went wrong!";
    });
}