document.getElementById("learningStyleForm").addEventListener("submit", async function(event) {
    event.preventDefault();

    // Collect form data (convert responses to numerical features)
    let features = [];
    for (let i = 1; i <= 17; i++) {
        features.push(parseInt(document.querySelector(`[name="q${i}"]`).value));
    }

    try {
        // Send data to FastAPI backend
        let response = await fetch("http://127.0.0.1:8000/predict/", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ features: features })
        });

        if (!response.ok) {
            throw new Error("Server error, please try again later.");
        }

        let result = await response.json();
        document.getElementById("predictionResult").innerText = result.prediction;
    } catch (error) {
        document.getElementById("predictionResult").innerText = "Error: " + error.message;
    }
});




