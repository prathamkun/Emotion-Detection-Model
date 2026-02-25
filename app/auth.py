from flask import Blueprint, render_template, request, redirect, url_for, session
from werkzeug.security import generate_password_hash, check_password_hash

from .models import db, User

auth = Blueprint("auth", __name__)

# 🔥 SIGNUP ROUTE
@auth.route("/signup", methods=["GET", "POST"])
def signup():

    if request.method == "POST":

        email = request.form["email"]
        password = request.form["password"]

        # check if user already exists
        existing_user = User.query.filter_by(email=email).first()

        if existing_user:
            return "User already exists"

        # hash password (IMPORTANT SECURITY)
        hashed_password = generate_password_hash(password)

        new_user = User(email=email, password=hashed_password)

        db.session.add(new_user)
        db.session.commit()

        return redirect(url_for("auth.login"))

    return render_template("signup.html")


# 🔥 LOGIN ROUTE
@auth.route("/login", methods=["GET", "POST"])
def login():

    if request.method == "POST":

        email = request.form["email"]
        password = request.form["password"]

        user = User.query.filter_by(email=email).first()

        if user and check_password_hash(user.password, password):

            session["user"] = user.email
            return redirect(url_for("main.dashboard"))

        return "Invalid login"

    return render_template("login.html")


# 🔥 LOGOUT
@auth.route("/logout")
def logout():

    session.pop("user", None)

    return redirect(url_for("auth.login"))