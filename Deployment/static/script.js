let modelsData = {};

// Load models.json from backend
async function loadModels() {
    const response = await fetch("/models");
    modelsData = await response.json();
    updateModelInfo();
}

// Update table when model changes
function updateModelInfo() {
    const selected = document.getElementById("model").value;
    const config = modelsData[selected];

    if (!config) return;

    const table = document.getElementById("model-table");

    table.innerHTML = `
        <tr><td>Parameters</td><td>${config.parameters} M</td></tr>
        <tr><td>Embedding</td><td>${config.n_embd}</td></tr>
        <tr><td>Heads</td><td>${config.n_head}</td></tr>
        <tr><td>Layers</td><td>${config.n_layer}</td></tr>
        <tr><td>Batch Size</td><td>${config.batch_size}</td></tr>
        <tr><td>Block Size</td><td>${config.block_size}</td></tr>
        <tr><td>Dropout</td><td>${config.dropout}</td></tr>
        <tr><td>Learning Rate</td><td>${config.learning_rate}</td></tr>
        <tr><td>Max Iters</td><td>${config.max_iters}</td></tr>
        <tr><td>Train Loss</td><td>${config.train_loss ?? "N/A"}</td></tr>
        <tr><td>Val Loss</td><td>${config.val_loss ?? "N/A"}</td></tr>
    `;
}

// Generate text
async function generateText() {

    const model = document.getElementById("model").value;
    const prompt = document.getElementById("prompt").value;
    const temperature = parseFloat(document.getElementById("temperature").value);
    const max_tokens = parseInt(document.getElementById("max_tokens").value);

    document.getElementById("output").innerText = "Generating...";

    const response = await fetch("/generate", {
        method: "POST",
        headers: {
            "Content-Type": "application/json"
        },
        body: JSON.stringify({
            model: model,
            prompt: prompt,
            temperature: temperature,
            max_tokens: max_tokens
        })
    });

    const data = await response.json();
    document.getElementById("output").innerText = data.response;
}

// Listen for model change
document.getElementById("model")
    .addEventListener("change", updateModelInfo);

window.onload = loadModels;