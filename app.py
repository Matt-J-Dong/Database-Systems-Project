from flask import Flask, render_template, request, redirect, url_for
from flask_sqlalchemy import SQLAlchemy
import pandas as pd
import os

app = Flask(__name__)

# Update the SQLAlchemy Database URI to connect to Azure SQL Server
server_name = "database-systems-project-server.database.windows.net"
database_name = "Project_Database"
username = "Project"
password = "Testing1"
driver = "ODBC Driver 18 for SQL Server"

app.config["SQLALCHEMY_DATABASE_URI"] = (
    f"mssql+pyodbc://{username}:{password}@{server_name}:1433/{database_name}?driver={driver}"
)
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db = SQLAlchemy(app)

class Textbook(db.Model):
    __tablename__ = "textbooks"
    ISBN = db.Column(db.String(20), primary_key=True, nullable=False)
    Topic_ID = db.Column(db.Integer, primary_key=True, nullable=False)
    Title = db.Column(db.String(255), nullable=False)
    Author = db.Column(db.String(255), nullable=False)
    Publisher = db.Column(db.String(255), nullable=False)
    Edition = db.Column(db.Integer, nullable=False)
    Publication_Year = db.Column(db.Integer, nullable=False)
    Topic = db.Column(db.String(255), nullable=False)

def get_correlation_variable():
    df = pd.read_csv('correlation_data.csv')
    correlation_variable = df["correlation_variable"].iloc[0]
    return correlation_variable

def calculate_price(correlation_variable):
    base_price = 5000  # base price in dollars
    price = base_price + (correlation_variable * 1000)
    return price

@app.route('/', methods=['GET', 'POST'])
def index():
    correlation_variable = get_correlation_variable()
    price = calculate_price(correlation_variable)

    if request.method == 'POST':
        first_name = request.form['first_name']
        last_name = request.form['last_name']
        school_name = request.form["school_name"]
        contact_email = request.form["contact_email"]
        save_purchase(first_name, last_name, school_name, contact_email, price)
        return redirect(
            url_for(
                "success",
                first_name=first_name,
                last_name=last_name,
                school_name=school_name,
                contact_email=contact_email,
            )
        )

    return render_template('index.html', price=price)

@app.route('/success')
def success():
    first_name = request.args.get('first_name')
    last_name = request.args.get('last_name')
    school_name = request.args.get("school_name")
    contact_email = request.args.get("contact_email")
    return f"Thank you, {first_name} {last_name}, for purchasing our consultancy package! We will contact your school {school_name} at {contact_email} for further details shortly."


def save_purchase(first_name, last_name, school_name, contact_email, price):
    file_exists = os.path.isfile("purchases.csv")
    purchase_data = {
        "First Name": first_name,
        "Last Name": last_name,
        "School Name": school_name,
        "Contact Email": contact_email,
        "Price": price,
    }
    df = pd.DataFrame([purchase_data])
    df.to_csv("purchases.csv", mode="a", index=False, header=not file_exists)


@app.route("/textbooks", methods=["GET", "POST"])
def textbooks():
    search_result = None
    if request.method == "POST":
        title = request.form["title"].strip().lower()
        search_result = Textbook.query.filter(
            db.func.lower(Textbook.Title) == title
        ).all()
    return render_template("textbooks.html", search_result=search_result)


def load_csv_to_db():
    with app.app_context():
        if not Textbook.query.first():
            df = pd.read_csv("textbooks.csv")
            for index, row in df.iterrows():
                if not Textbook.query.filter_by(
                    ISBN=row["ISBN"], Topic_ID=row["Topic_ID"]
                ).first():
                    textbook = Textbook(
                        ISBN=row["ISBN"],
                        Title=row["Title"],
                        Author=row["Author"],
                        Publisher=row["Publisher"],
                        Edition=row["Edition"],
                        Publication_Year=row["Publication_Year"],
                        Topic=row["Topic"],
                        Topic_ID=row["Topic_ID"],
                    )
                    db.session.add(textbook)
            db.session.commit()


def remove_duplicates():
    with app.app_context():
        textbooks = Textbook.query.all()
        unique_books = {}
        for book in textbooks:
            key = (book.ISBN, book.Topic_ID)
            if key not in unique_books:
                unique_books[key] = book
            else:
                db.session.delete(book)
        db.session.commit()


if __name__ == '__main__':
    remove_duplicates()
    load_csv_to_db()
    app.run(debug=True)
