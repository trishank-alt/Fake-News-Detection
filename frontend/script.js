async function checkNews(){

    const text = document.getElementById("newsText").value;

    const response = await fetch("http://127.0.0.1:8000/predict",{
        method:"POST",
        headers:{
            "Content-Type":"application/json"
        },
        body: JSON.stringify({text:text})
    });

    const data = await response.json();

    document.getElementById("prediction").innerText =
        "Prediction: " + data.prediction;

    document.getElementById("confidence").innerText =
        "Confidence: " + (data.confidence*100).toFixed(2) + "%";
}