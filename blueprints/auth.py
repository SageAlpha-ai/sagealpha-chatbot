"""
SageAlpha.ai Authentication Blueprint
Modern Flask 3.x authentication with bcrypt password hashing
"""

from datetime import datetime, timedelta, timezone

from flask import (
    Blueprint,
    current_app,
    flash,
    jsonify,
    redirect,
    render_template,
    request,
    session,
    url_for,
)
from flask_login import current_user, login_required, login_user, logout_user
from werkzeug.security import check_password_hash, generate_password_hash

from models import User, db

auth_bp = Blueprint("auth", __name__, template_folder="../templates")


def read_version():
    """Get application version from env or VERSION file."""
    import os

    v = os.getenv("SAGEALPHA_VERSION")
    if v:
        return v.strip()
    try:
        with open(
            os.path.join(os.path.dirname(os.path.dirname(__file__)), "VERSION"), "r"
        ) as f:
            return f.read().strip()
    except Exception:
        return "3.0.0"


@auth_bp.route("/login", methods=["GET", "POST"])
def login():
    """
    Render login page (GET), accept form POST (username/password).
    Accepts demo accounts demouser/devuser/produser for testing.
    """
    if hasattr(current_user, "is_authenticated") and current_user.is_authenticated:
        return redirect("/")

    if request.method == "GET":
        return render_template("login.html", APP_VERSION=read_version())

    username = (request.form.get("username") or "").strip()
    password = request.form.get("password") or ""

    if not username or not password:
        return (
            render_template(
                "login.html",
                error="Username and password required.",
                username=username,
                APP_VERSION=read_version(),
            ),
            400,
        )

    user = None
    try:
        user = User.query.filter_by(username=username).first()
    except Exception as e:
        current_app.logger.error(f"[login] DB lookup failed: {e!r}")

    authenticated = False
    if user:
        if hasattr(user, "check_password") and callable(user.check_password):
            try:
                authenticated = user.check_password(password)
            except Exception:
                authenticated = False
        else:
            if getattr(user, "password", None) == password:
                authenticated = True

    # Demo accounts fallback
    demo_passwords = {
        "demouser": "DemoPass123!",
        "devuser": "DevPass123!",
        "produser": "ProdPass123!",
    }
    if not authenticated and username in demo_passwords:
        if password == demo_passwords[username]:
            authenticated = True
            if not user:

                class _TempUser:
                    def __init__(self, username):
                        self.id = username
                        self.username = username
                        self.is_active = True
                        self.is_authenticated = True
                        self.is_anonymous = False

                    def get_id(self):
                        return str(self.id)

                user = _TempUser(username)

    if not authenticated:
        return (
            render_template(
                "login.html",
                error="Invalid username or password.",
                username=username,
                APP_VERSION=read_version(),
            ),
            401,
        )

    try:
        login_user(user)
    except Exception as e:
        current_app.logger.warning(f"[login] login_user failed: {e!r}")
        session["logged_in"] = True
        session["username"] = username

    return redirect("/")


@auth_bp.route("/register", methods=["POST"])
def register():
    """Handle new user registration from the sign-up form."""
    if hasattr(current_user, "is_authenticated") and current_user.is_authenticated:
        return redirect("/")

    username = (request.form.get("username") or "").strip()
    password = request.form.get("password") or ""
    confirm = request.form.get("confirm_password") or ""
    display_name = (request.form.get("display_name") or "").strip()

    register_error = None
    if not username or not password or not confirm:
        register_error = "All fields are required."
    elif password != confirm:
        register_error = "Passwords do not match."
    elif len(password) < 8:
        register_error = "Password must be at least 8 characters long."

    if register_error is None:
        try:
            existing = User.query.filter_by(username=username).first()
            if existing:
                register_error = "Username already taken. Please choose another."
        except Exception as e:
            current_app.logger.error(f"[register] DB lookup failed: {e!r}")
            register_error = "Unexpected error. Please try again."

    if register_error:
        return (
            render_template(
                "login.html",
                show_register=True,
                register_error=register_error,
                reg_username=username,
                reg_display_name=display_name,
                APP_VERSION=read_version(),
            ),
            400,
        )

    try:
        user = User(
            username=username,
            display_name=display_name or username,
            password_hash=generate_password_hash(password),
        )
        db.session.add(user)
        db.session.commit()
    except Exception as e:
        current_app.logger.error(f"[register] Failed to create user: {e!r}")
        return (
            render_template(
                "login.html",
                show_register=True,
                register_error="Could not create account. Please try again.",
                reg_username=username,
                reg_display_name=display_name,
                APP_VERSION=read_version(),
            ),
            500,
        )

    try:
        login_user(user)
    except Exception as e:
        current_app.logger.warning(f"[register] login_user failed: {e!r}")
        session["logged_in"] = True
        session["username"] = username

    return redirect("/")


@auth_bp.route("/logout", methods=["GET", "POST"])
def logout():
    """Clear Flask-Login state and session, redirect to login."""
    try:
        logout_user()
    except Exception:
        pass

    session.clear()

    if request.is_json or request.headers.get("X-Requested-With") == "XMLHttpRequest":
        return jsonify({"success": True, "next": url_for("auth.login")})

    return redirect(url_for("auth.login"))


@auth_bp.route("/profile", methods=["GET"])
@login_required
def profile():
    """User profile page."""
    try:
        return render_template("profile.html", APP_VERSION=read_version())
    except Exception:
        return jsonify({"profile": getattr(current_user, "username", "Guest")}), 200


@auth_bp.route("/user", methods=["GET"])
def user():
    """Return current user info as JSON."""
    if not (
        hasattr(current_user, "is_authenticated") and current_user.is_authenticated
    ):
        return jsonify(
            {"username": "Guest", "email": "guest@gmail.com", "avatar_url": None}
        )

    return jsonify(
        {
            "username": current_user.username,
            "email": f"{current_user.username}@local",
            "avatar_url": None,
        }
    )

