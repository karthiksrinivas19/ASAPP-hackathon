"""
Validation utilities
"""

import re
from typing import Optional


def validate_pnr(pnr: str) -> bool:
    """
    Validate PNR format
    PNR should be 6 alphanumeric characters
    """
    if not pnr:
        return False
    
    # Remove whitespace and convert to uppercase
    pnr = pnr.strip().upper()
    
    # Check if it's 6 alphanumeric characters
    pattern = r'^[A-Z0-9]{6}$'
    return bool(re.match(pattern, pnr))


def validate_flight_number(flight_number: str) -> bool:
    """
    Validate flight number format
    Flight number should be 2-3 letters followed by 1-4 digits
    """
    if not flight_number:
        return False
    
    # Remove whitespace and convert to uppercase
    flight_number = flight_number.strip().upper()
    
    # Check format: 2-3 letters + 1-4 digits
    pattern = r'^[A-Z]{2,3}[0-9]{1,4}$'
    return bool(re.match(pattern, flight_number))


def validate_email(email: str) -> bool:
    """
    Validate email format
    """
    if not email:
        return False
    
    # Basic email validation pattern
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email.strip()))


def validate_phone(phone: str) -> bool:
    """
    Validate phone number format
    Accepts various formats: +1234567890, (123) 456-7890, 123-456-7890, etc.
    """
    if not phone:
        return False
    
    # Remove all non-digit characters except +
    cleaned = re.sub(r'[^\d+]', '', phone.strip())
    
    # Check if it's a valid length (10-15 digits, optionally starting with +)
    if cleaned.startswith('+'):
        # International format: +1234567890 (11-16 characters total)
        return len(cleaned) >= 11 and len(cleaned) <= 16
    else:
        # Domestic format: 1234567890 (10-15 digits)
        return len(cleaned) >= 10 and len(cleaned) <= 15


def validate_airport_code(code: str) -> bool:
    """
    Validate airport code format (IATA 3-letter codes)
    """
    if not code:
        return False
    
    # Remove whitespace and convert to uppercase
    code = code.strip().upper()
    
    # Check if it's exactly 3 letters
    pattern = r'^[A-Z]{3}$'
    return bool(re.match(pattern, code))


def extract_pnr_from_text(text: str) -> Optional[str]:
    """
    Extract PNR from text using regex patterns
    """
    if not text:
        return None
    
    # Look for 6 alphanumeric characters that could be a PNR
    # Common patterns: "PNR ABC123", "booking ABC123", "confirmation ABC123"
    patterns = [
        r'\b(?:PNR|booking|confirmation|reference)\s*:?\s*([A-Z0-9]{6})\b',
        r'\b([A-Z0-9]{6})\b',  # Any 6 alphanumeric characters
    ]
    
    text_upper = text.upper()
    
    for pattern in patterns:
        matches = re.findall(pattern, text_upper)
        for match in matches:
            if validate_pnr(match):
                return match
    
    return None


def extract_flight_number_from_text(text: str) -> Optional[str]:
    """
    Extract flight number from text using regex patterns
    """
    if not text:
        return None
    
    # Look for flight number patterns
    patterns = [
        r'\b(?:flight|flt)\s*:?\s*([A-Z]{2,3}[0-9]{1,4})\b',
        r'\b([A-Z]{2,3}[0-9]{1,4})\b',  # Any airline code + numbers
    ]
    
    text_upper = text.upper()
    
    for pattern in patterns:
        matches = re.findall(pattern, text_upper)
        for match in matches:
            if validate_flight_number(match):
                return match
    
    return None


def extract_email_from_text(text: str) -> Optional[str]:
    """
    Extract email address from text
    """
    if not text:
        return None
    
    pattern = r'\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b'
    matches = re.findall(pattern, text)
    
    for match in matches:
        if validate_email(match):
            return match.lower()
    
    return None


def extract_phone_from_text(text: str) -> Optional[str]:
    """
    Extract phone number from text
    """
    if not text:
        return None
    
    # Look for phone number patterns
    patterns = [
        r'\+?1?[-.\s]?\(?([0-9]{3})\)?[-.\s]?([0-9]{3})[-.\s]?([0-9]{4})',
        r'\+?([0-9]{1,4})[-.\s]?([0-9]{3,4})[-.\s]?([0-9]{3,4})[-.\s]?([0-9]{3,4})',
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, text)
        for match in matches:
            # Reconstruct phone number
            if isinstance(match, tuple):
                phone = ''.join(match)
            else:
                phone = match
            
            if validate_phone(phone):
                return phone
    
    return None