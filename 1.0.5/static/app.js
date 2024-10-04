document.getElementById('predictionForm').addEventListener('submit', async function (e) {
    e.preventDefault();

    const year = document.getElementById('year').value;
    const month = document.getElementById('month').value;

    // Clear previous results and errors
    document.getElementById('results').style.display = 'none';
    document.getElementById('error').style.display = 'none';
    document.getElementById('categoryExpenses').innerHTML = '';
    document.getElementById('totalExpense').innerText = '';

    try {
        // Send the data to the backend API
        const response = await fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ year: parseInt(year), month: parseInt(month) })
        });

        const result = await response.json();

        if (response.ok) {
            // Display the total monthly expense
            document.getElementById('totalExpense').innerText = result.total_monthly_expense.toFixed(2);

            // Display category-wise expenses
            const categoryExpenses = result.category_wise_predictions;
            for (const category in categoryExpenses) {
                const listItem = document.createElement('li');
                listItem.textContent = `${category}: ${categoryExpenses[category].toFixed(2)}`;
                document.getElementById('categoryExpenses').appendChild(listItem);
            }

            // Show the results section
            document.getElementById('results').style.display = 'block';
        } else {
            // Handle error
            document.getElementById('error').textContent = 'Error: Unable to fetch predictions.';
            document.getElementById('error').style.display = 'block';
        }
    } catch (error) {
        document.getElementById('error').textContent = 'Error: ' + error.message;
        document.getElementById('error').style.display = 'block';
    }
});
