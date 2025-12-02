"""
SageAlpha.ai Database Migration Utility
Safe SQLite schema migration without data loss

Run this automatically on app startup to add missing columns
to existing tables without dropping or recreating them.
"""

import sqlite3
from datetime import datetime, timezone
from typing import Dict, List, Optional, Set, Tuple


def get_existing_columns(conn: sqlite3.Connection, table_name: str) -> Set[str]:
    """Get set of existing column names for a table."""
    cursor = conn.execute(f"PRAGMA table_info({table_name})")
    return {row[1] for row in cursor.fetchall()}


def table_exists(conn: sqlite3.Connection, table_name: str) -> bool:
    """Check if a table exists in the database."""
    cursor = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
        (table_name,)
    )
    return cursor.fetchone() is not None


def add_column_if_missing(
    conn: sqlite3.Connection,
    table_name: str,
    column_name: str,
    column_type: str,
    default_value: Optional[str] = None,
) -> bool:
    """
    Add a column to a table if it doesn't exist.
    
    Args:
        conn: SQLite connection
        table_name: Name of the table
        column_name: Name of the column to add
        column_type: SQLite column type (TEXT, INTEGER, BOOLEAN, DATETIME, etc.)
        default_value: Optional default value (as SQL expression)
    
    Returns:
        True if column was added, False if it already existed
    """
    existing = get_existing_columns(conn, table_name)
    
    if column_name in existing:
        return False
    
    # Build ALTER TABLE statement
    sql = f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_type}"
    if default_value is not None:
        sql += f" DEFAULT {default_value}"
    
    print(f"[migrate] Adding column: {table_name}.{column_name} ({column_type})")
    conn.execute(sql)
    return True


def migrate_users_table(conn: sqlite3.Connection) -> int:
    """
    Migrate the users table to add missing columns.
    
    Returns:
        Number of columns added
    """
    if not table_exists(conn, "users"):
        print("[migrate] users table does not exist, will be created by SQLAlchemy")
        return 0
    
    columns_added = 0
    
    # List of columns to add: (name, type, default)
    columns_to_add: List[Tuple[str, str, Optional[str]]] = [
        ("email", "VARCHAR(120)", "NULL"),
        ("created_at", "DATETIME", f"'{datetime.now(timezone.utc).isoformat()}'"),
        ("updated_at", "DATETIME", f"'{datetime.now(timezone.utc).isoformat()}'"),
        ("is_active", "BOOLEAN", "1"),  # SQLite uses 1 for True
    ]
    
    for col_name, col_type, default in columns_to_add:
        if add_column_if_missing(conn, "users", col_name, col_type, default):
            columns_added += 1
    
    return columns_added


def migrate_messages_table(conn: sqlite3.Connection) -> int:
    """
    Migrate the messages table to add missing columns.
    
    Returns:
        Number of columns added
    """
    if not table_exists(conn, "messages"):
        print("[migrate] messages table does not exist, will be created by SQLAlchemy")
        return 0
    
    columns_added = 0
    
    # Check for session_id column (was added in v3)
    columns_to_add: List[Tuple[str, str, Optional[str]]] = [
        ("session_id", "VARCHAR(36)", "NULL"),
        ("meta_json", "TEXT", "NULL"),
    ]
    
    for col_name, col_type, default in columns_to_add:
        if add_column_if_missing(conn, "messages", col_name, col_type, default):
            columns_added += 1
    
    return columns_added


def run_migrations(db_path: str) -> Dict[str, int]:
    """
    Run all database migrations.
    
    Args:
        db_path: Path to the SQLite database file
    
    Returns:
        Dictionary with table names and number of columns added
    """
    print(f"[migrate] Starting database migration for: {db_path}")
    
    results: Dict[str, int] = {}
    
    try:
        conn = sqlite3.connect(db_path)
        
        # Disable foreign key checks during migration
        conn.execute("PRAGMA foreign_keys=OFF")
        
        # Run migrations
        results["users"] = migrate_users_table(conn)
        results["messages"] = migrate_messages_table(conn)
        
        # Re-enable foreign key checks
        conn.execute("PRAGMA foreign_keys=ON")
        
        # Commit all changes
        conn.commit()
        conn.close()
        
        total_added = sum(results.values())
        if total_added > 0:
            print(f"[migrate] Migration complete: {total_added} column(s) added")
        else:
            print("[migrate] No migration needed, schema is up to date")
        
        return results
        
    except Exception as e:
        print(f"[migrate][ERROR] Migration failed: {e!r}")
        raise


def check_schema_compatibility(db_path: str) -> Tuple[bool, List[str]]:
    """
    Check if the database schema is compatible with the current models.
    
    Args:
        db_path: Path to the SQLite database file
    
    Returns:
        Tuple of (is_compatible, list_of_missing_columns)
    """
    import os
    
    if not os.path.exists(db_path):
        return True, []
    
    missing = []
    
    try:
        conn = sqlite3.connect(db_path)
        
        # Check users table
        if table_exists(conn, "users"):
            existing = get_existing_columns(conn, "users")
            required = {"id", "username", "display_name", "password_hash", 
                       "email", "created_at", "updated_at", "is_active"}
            missing_users = required - existing
            for col in missing_users:
                missing.append(f"users.{col}")
        
        # Check messages table  
        if table_exists(conn, "messages"):
            existing = get_existing_columns(conn, "messages")
            required = {"id", "user_id", "session_id", "role", "content", 
                       "timestamp", "meta_json"}
            missing_messages = required - existing
            for col in missing_messages:
                missing.append(f"messages.{col}")
        
        conn.close()
        
        return len(missing) == 0, missing
        
    except Exception as e:
        print(f"[migrate] Schema check failed: {e!r}")
        return False, [f"error: {e!s}"]


if __name__ == "__main__":
    # Allow running directly for manual migration
    import os
    import sys
    
    # Determine database path
    if len(sys.argv) > 1:
        db_path = sys.argv[1]
    else:
        # Default paths to check
        possible_paths = [
            "sagealpha.db",
            "instance/sagealpha.db",
            os.path.join(os.path.dirname(__file__), "sagealpha.db"),
        ]
        db_path = None
        for path in possible_paths:
            if os.path.exists(path):
                db_path = path
                break
        
        if db_path is None:
            print("No database found. Specify path as argument.")
            sys.exit(1)
    
    print(f"Database path: {db_path}")
    
    # Check compatibility first
    is_compatible, missing = check_schema_compatibility(db_path)
    if not is_compatible:
        print(f"Missing columns: {missing}")
        print("Running migration...")
        run_migrations(db_path)
    else:
        print("Schema is already up to date.")

