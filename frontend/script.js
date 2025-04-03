document.getElementById('predictButton').addEventListener('click', function() {
    // Get the email text input
    const emailText = document.getElementById('emailText').value;

    // Check if the input is empty
    if (emailText.trim() === "") {
        alert("Please enter email content.");
        return;
    }

    // Prepare the data to send to the backend
    const data = {
        text: emailText
    };

    // Call the Flask API using Fetch
    fetch('http://127.0.0.1:5000/predict', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(data)
    })
    .then(response => response.json())
    .then(data => {
        // Display the prediction result
        document.getElementById('result').innerHTML = `
            <p><strong>Prediction:</strong> ${data.message}</p>
        `;
    })
    .catch(error => {
        console.error('Error:', error);
        alert('An error occurred while making the prediction.');
    });
});
