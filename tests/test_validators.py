"""
Tests for validation utilities
"""

import pytest
from airline_service.utils.validators import (
    validate_pnr,
    validate_flight_number,
    validate_email,
    validate_phone,
    validate_airport_code,
    extract_pnr_from_text,
    extract_flight_number_from_text,
    extract_email_from_text,
    extract_phone_from_text
)


class TestPNRValidation:
    """Test PNR validation"""
    
    def test_valid_pnr(self):
        """Test valid PNR formats"""
        assert validate_pnr("ABC123")
        assert validate_pnr("XYZ789")
        assert validate_pnr("123ABC")
        assert validate_pnr("abc123")  # Should work with lowercase
    
    def test_invalid_pnr(self):
        """Test invalid PNR formats"""
        assert not validate_pnr("")
        assert not validate_pnr("ABC12")  # Too short
        assert not validate_pnr("ABC1234")  # Too long
        assert not validate_pnr("ABC-123")  # Contains special character
        assert not validate_pnr("ABC 123")  # Contains space
        assert not validate_pnr(None)


class TestFlightNumberValidation:
    """Test flight number validation"""
    
    def test_valid_flight_number(self):
        """Test valid flight number formats"""
        assert validate_flight_number("AA100")
        assert validate_flight_number("UA2000")
        assert validate_flight_number("DL1")
        assert validate_flight_number("JBU123")
        assert validate_flight_number("aa100")  # Should work with lowercase
    
    def test_invalid_flight_number(self):
        """Test invalid flight number formats"""
        assert not validate_flight_number("")
        assert not validate_flight_number("A100")  # Too few letters
        assert not validate_flight_number("ABCD100")  # Too many letters
        assert not validate_flight_number("AA")  # No numbers
        assert not validate_flight_number("AA12345")  # Too many numbers
        assert not validate_flight_number("AA-100")  # Contains special character


class TestEmailValidation:
    """Test email validation"""
    
    def test_valid_email(self):
        """Test valid email formats"""
        assert validate_email("test@example.com")
        assert validate_email("user.name@domain.co.uk")
        assert validate_email("user+tag@example.org")
    
    def test_invalid_email(self):
        """Test invalid email formats"""
        assert not validate_email("")
        assert not validate_email("invalid-email")
        assert not validate_email("@example.com")
        assert not validate_email("test@")
        assert not validate_email("test.example.com")


class TestPhoneValidation:
    """Test phone validation"""
    
    def test_valid_phone(self):
        """Test valid phone formats"""
        assert validate_phone("1234567890")
        assert validate_phone("+11234567890")
        assert validate_phone("(123) 456-7890")
        assert validate_phone("123-456-7890")
        assert validate_phone("+44 20 7946 0958")  # International
    
    def test_invalid_phone(self):
        """Test invalid phone formats"""
        assert not validate_phone("")
        assert not validate_phone("123")  # Too short
        assert not validate_phone("12345678901234567")  # Too long
        assert not validate_phone("abc-def-ghij")  # No digits


class TestAirportCodeValidation:
    """Test airport code validation"""
    
    def test_valid_airport_code(self):
        """Test valid airport codes"""
        assert validate_airport_code("JFK")
        assert validate_airport_code("LAX")
        assert validate_airport_code("ORD")
        assert validate_airport_code("jfk")  # Should work with lowercase
    
    def test_invalid_airport_code(self):
        """Test invalid airport codes"""
        assert not validate_airport_code("")
        assert not validate_airport_code("JF")  # Too short
        assert not validate_airport_code("JFKL")  # Too long
        assert not validate_airport_code("JF1")  # Contains number
        assert not validate_airport_code("JF-K")  # Contains special character


class TestTextExtraction:
    """Test text extraction functions"""
    
    def test_extract_pnr_from_text(self):
        """Test PNR extraction from text"""
        assert extract_pnr_from_text("My PNR is ABC123") == "ABC123"
        assert extract_pnr_from_text("Booking reference: XYZ789") == "XYZ789"
        assert extract_pnr_from_text("Confirmation ABC123 for my flight") == "ABC123"
        assert extract_pnr_from_text("No PNR here") is None
    
    def test_extract_flight_number_from_text(self):
        """Test flight number extraction from text"""
        assert extract_flight_number_from_text("Flight AA100 is delayed") == "AA100"
        assert extract_flight_number_from_text("I'm on UA2000 tomorrow") == "UA2000"
        assert extract_flight_number_from_text("No flight number here") is None
    
    def test_extract_email_from_text(self):
        """Test email extraction from text"""
        assert extract_email_from_text("My email is test@example.com") == "test@example.com"
        assert extract_email_from_text("Contact user.name@domain.co.uk") == "user.name@domain.co.uk"
        assert extract_email_from_text("No email here") is None
    
    def test_extract_phone_from_text(self):
        """Test phone extraction from text"""
        assert extract_phone_from_text("Call me at 123-456-7890") == "1234567890"
        assert extract_phone_from_text("My number is (123) 456-7890") == "1234567890"
        assert extract_phone_from_text("No phone here") is None