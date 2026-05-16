from genericpath import isdir
import os

from rag.src.config import USERS_DIR

class UserManager:
    """A simple user manager."""
    
    @staticmethod
    def get_users():
        """Get all users."""
        if not os.path.exists(USERS_DIR):
            return []
        
        users = []
        for item in os.listdir(USERS_DIR):
            user_path = os.path.join(USERS_DIR, item)
            if os.path.isdir(user_path):
                users.append(item)
        return sorted(users)
    
    @staticmethod
    def exists_user(user_id: str):
        """Check if a user exists by ID."""
        user_path = os.path.join(USERS_DIR, user_id)
        return os.path.exists(user_path)
    
    @staticmethod
    def create_user(user_id: str):
        """Create a new user."""
        try:
            user_path = os.path.join(USERS_DIR, user_id)
            os.makedirs(user_path)
            return True
        except FileExistsError:
            return False
    