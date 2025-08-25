import unittest
import os
from dotenv import load_dotenv
from app import auth  # Import your Chainlit handler
import chainlit as cl
# Load environment variables
load_dotenv()

class TestClass(unittest.TestCase):
    def test_authentication_valid_credentials(self):
        # Test with valid credentials - should return a User object
        user = auth("admin", os.getenv("PASSWORD"))
        assert isinstance(user, cl.User)
         
    def test_authentication_invalid_credentials(self):
        # Test with invalid credentials - should return None
        user = auth("admin", "wrong_password")
        assert not isinstance(user, cl.User)

    def test_authentication_invalid_username(self):
        # Test with invalid username - should return None
        user = auth("wrong_user", os.getenv("PASSWORD"))
        assert not isinstance(user, cl.User)
        