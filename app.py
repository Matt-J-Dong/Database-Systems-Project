from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
import os

app = Flask(__name__)

def get_correlation_variable():
    df = pd.read_csv('correlation_data.csv')
    correlation_variable = df["correlation_variable"].iloc[0]
    return correlation_variable

def calculate_price(correlation_variable):
    base_price = 5000  # base price in dollars
    price = base_price + (correlation_variable * 1000)
    return price

def save_purchase(first_name, last_name, price):
    file_exists = os.path.isfile("purchases.csv")
    purchase_data = {"First Name": first_name, "Last Name": last_name, "Price": price}
    df = pd.DataFrame([purchase_data])
    df.to_csv('purchases.csv', mode='a', index=False, header=not file_exists)

@app.route('/', methods=['GET', 'POST'])
def index():
    correlation_variable = get_correlation_variable()
    price = calculate_price(correlation_variable)

    if request.method == 'POST':
        first_name = request.form['first_name']
        last_name = request.form['last_name']
        # Save the purchase details
        save_purchase(first_name, last_name, price)
        return redirect(url_for('success', first_name=first_name, last_name=last_name))

    return render_template('index.html', price=price)

@app.route('/success')
def success():
    first_name = request.args.get('first_name')
    last_name = request.args.get('last_name')
    return f"Thank you, {first_name} {last_name}, for purchasing our consultancy package!"


# Textbook Lookup Route
@app.route("/textbooks", methods=["GET", "POST"])
def textbooks():
    search_result = None
    if request.method == "POST":
        title = request.form["title"].strip().lower()
        df = pd.read_csv("textbooks.csv")
        search_result = df[df["Title"].str.lower() == title]
        if search_result.empty:
            search_result = None
    return render_template("textbooks.html", search_result=search_result)


if __name__ == '__main__':
    app.run(debug=True)
