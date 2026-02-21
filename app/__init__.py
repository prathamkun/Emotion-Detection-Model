from flask import Flask

def create_app():

    app = Flask(__name__)
    app.config["SECRET_KEY"] = "supersecretkey"

    # Import and register routes
    from .routes import main
    from .auth import auth
    from .api import api_bp

    app.register_blueprint(main)
    app.register_blueprint(auth)
    app.register_blueprint(api_bp)

    return app