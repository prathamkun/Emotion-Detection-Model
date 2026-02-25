from flask import Flask
from .models import db

def create_app():

    app = Flask(__name__)

    app.config["SECRET_KEY"] = "supersecretkey"

    import os

    BASE_DIR = os.path.abspath(os.path.dirname(__file__))
    app.config["SQLALCHEMY_DATABASE_URI"] = \
        "sqlite:///" + os.path.join(BASE_DIR, "..", "database.db")

    db.init_app(app)

    from .routes import main
    from .auth import auth
    from .api import api_bp

    app.register_blueprint(main)
    app.register_blueprint(auth)
    app.register_blueprint(api_bp)

    return app