import matplotlib.pyplot as plt


class ExpenseVisualizer:
    """
    Handles the visualization of expense data using charts (Matplotlib).
    """

    def __init__(self, expenses):
        self.expenses = expenses

    def plot_expenses(self):
        """
        Aggregates expenses by category and displays a pie chart.
        """
        # 1. Check if there is data to visualize
        if not self.expenses:
            print("❌ No data available to visualize!")
            return

        # 2. Aggregate expenses by category (Data Aggregation)
        # Example result: { 'Food': 50, 'Transport': 20 }
        category_totals = {}

        for expense in self.expenses:
            # We use .strip() to ensure 'Food ' and 'Food' are treated as the same category
            category = expense.category.strip()

            if category in category_totals:
                category_totals[category] += expense.amount
            else:
                category_totals[category] = expense.amount

        # 3. Prepare data lists for plotting
        categories = list(category_totals.keys())
        amounts = list(category_totals.values())

        # 4. Create the Pie Chart
        # figsize sets the size of the window
        plt.figure(figsize=(8, 6))
        plt.pie(amounts, labels=categories, autopct='%1.1f%%', startangle=140)
        plt.title('Expenses Breakdown by Category')

        # 5. Show the chart
        print("📈 Generating chart...")
        plt.show()