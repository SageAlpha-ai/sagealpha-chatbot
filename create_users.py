"""
SageAlpha.ai User Creation Script
Creates/updates demo users with proper schema migration handling
"""

import os
import sys

from werkzeug.security import generate_password_hash

# Add the project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from db_migrate import run_migrations
from models import User, db


def get_db_path():
    """Determine the database path based on environment."""
    if os.getenv("WEBSITE_SITE_NAME"):
        db_dir = "/home/data"
    else:
        db_dir = os.path.dirname(__file__)
    return os.path.join(db_dir, "sagealpha.db")


def create_users():
    """Create or update demo users."""
    # Import app here to avoid circular imports
    from app import app
    
    with app.app_context():
        # Run migrations first to ensure schema is up to date
        db_path = get_db_path()
        if os.path.exists(db_path):
            print(f"[create_users] Running migrations on: {db_path}")
            try:
                run_migrations(db_path)
            except Exception as e:
                print(f"[create_users][WARN] Migration failed: {e}")
        
        # Ensure tables exist
        db.create_all()
        
        # Demo users with all fields
        users = [
            {
                "username": "demouser",
                "display_name": "DemoUser",
                "password": "DemoPass123!",
                "email": "demouser@sagealpha.ai",
            },
            {
                "username": "devuser",
                "display_name": "DevUser",
                "password": "DevPass123!",
                "email": "devuser@sagealpha.ai",
            },
            {
                "username": "produser",
                "display_name": "ProductionUser",
                "password": "ProdPass123!",
                "email": "produser@sagealpha.ai",
            },
        ]
        
        created = 0
        updated = 0
        
        for user_data in users:
            username = user_data["username"]
            existing = User.query.filter_by(username=username).first()
            
            if not existing:
                # Create new user
                u = User(
                    username=username,
                    display_name=user_data["display_name"],
                    password_hash=generate_password_hash(user_data["password"]),
                    email=user_data["email"],
                    is_active=True,
                )
                db.session.add(u)
                created += 1
                print(f"  Created user: {username}")
            else:
                # Update existing user with new fields if missing
                needs_update = False
                
                if not getattr(existing, "email", None):
                    existing.email = user_data["email"]
                    needs_update = True
                
                if getattr(existing, "is_active", None) is None:
                    existing.is_active = True
                    needs_update = True
                
                if needs_update:
                    updated += 1
                    print(f"  Updated user: {username}")
                else:
                    print(f"  User exists: {username}")
        
        db.session.commit()
        
        print(f"\n[create_users] Complete: {created} created, {updated} updated")
        print("Demo accounts:")
        print("  - demouser / DemoPass123!")
        print("  - devuser / DevPass123!")
        print("  - produser / ProdPass123!")


def reset_user_password(username: str, new_password: str):
    """Reset a user's password."""
    from app import app
    
    with app.app_context():
        user = User.query.filter_by(username=username).first()
        if user:
            user.password_hash = generate_password_hash(new_password)
            db.session.commit()
            print(f"Password reset for user: {username}")
        else:
            print(f"User not found: {username}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--reset":
        # Usage: python create_users.py --reset username newpassword
        if len(sys.argv) >= 4:
            reset_user_password(sys.argv[2], sys.argv[3])
        else:
            print("Usage: python create_users.py --reset <username> <newpassword>")
    else:
        create_users()
