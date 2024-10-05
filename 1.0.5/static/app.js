let chartInstance = null; // Global variable for the chart instance

document.getElementById('predictionForm').addEventListener('submit', async function (e) {
    e.preventDefault();

    const year = document.getElementById('year').value;
    const month = document.getElementById('month').value;

    // Clear previous results and errors
    document.getElementById('results').style.display = 'none';
    document.getElementById('error').style.display = 'none';

    try {
        // Fetch prediction data from API
        const response = await fetch('/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ year: parseInt(year), month: parseInt(month) })
        });

        if (response.ok) {
            const data = await response.json();
            const totalExpense = data.total_monthly_expense;
            document.getElementById('totalExpense').innerText = totalExpense.toFixed(2);

            const categories = Object.keys(data.category_wise_predictions);
            const expenses = Object.values(data.category_wise_predictions);

            // Remove "category_" prefix from each category name
            const formattedCategories = categories.map(category => category.replace('category_', ''));

            // Sort by expense amount to get categories ordered by highest expense
            const sortedData = formattedCategories.map((category, index) => {
                return { category, expense: expenses[index] };
            }).sort((a, b) => b.expense - a.expense);

            const sortedCategories = sortedData.map(item => item.category);
            const sortedExpenses = sortedData.map(item => item.expense);

            // Destroy previous chart instance if it exists
            if (chartInstance) {
                chartInstance.destroy();
            }

            // Create a new chart
            const ctx = document.getElementById('expenseChart').getContext('2d');
            chartInstance = new Chart(ctx, {
                type: 'bar', // Bar chart
                data: {
                    labels: sortedCategories,
                    datasets: [{
                        label: 'Expenses',
                        data: sortedExpenses,
                        backgroundColor: 'rgba(75, 192, 192, 0.2)',
                        borderColor: 'rgba(75, 192, 192, 1)',
                        borderWidth: 1
                    }]
                },
                options: {
                    responsive: true,
                    plugins: {
                        tooltip: {
                            callbacks: {
                                label: function (context) {
                                    const total = totalExpense;
                                    const value = context.raw;
                                    const percentage = ((value / total) * 100).toFixed(2);
                                    return `${value.toFixed(2)} (${percentage}%)`;
                                }
                            }
                        }
                    },
                    scales: {
                        y: {
                            beginAtZero: true
                        }
                    }
                }
            });

            document.getElementById('results').style.display = 'block';
        } else {
            // Display error message
            document.getElementById('error').textContent = 'Error: Unable to fetch predictions.';
            document.getElementById('error').style.display = 'block';
        }
    } catch (error) {
        document.getElementById('error').textContent = 'Error: ' + error.message;
        document.getElementById('error').style.display = 'block';
    }
});
