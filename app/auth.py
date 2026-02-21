from flask import Blueprint, render_template, request, redirect, session, url_for

auth = Blueprint("auth", __name__)

# temporary user database (later we use real database)
users = {
    "admin": "123"
}

@auth.route("/login", methods=["GET", "POST"])
def login():

    if request.method == "POST":

        username = request.form["username"]
        password = request.form["password"]

        if username in users and users[username] == password:
            session["user"] = username
            return redirect(url_for("main.dashboard"))

    return render_template("login.html")


@auth.route("/logout")
def logout():

    session.pop("user", None)
    return redirect(url_for("auth.login"))