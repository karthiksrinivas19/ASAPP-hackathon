"""
Automated booking selection logic with intelligent priority rules
"""

from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass
import re

from ..types import BookingDetails, CustomerSearchInfo, FlightDetails


class BookingPriority(str, Enum):
    """Priority levels for booking selection"""
    CRITICAL = "critical"      # Departing within 24 hours
    HIGH = "high"             # Departing within 7 days
    MEDIUM = "medium"         # Departing within 30 days
    LOW = "low"              # Future bookings
    PAST = "past"            # Past flights


class SelectionReason(str, Enum):
    """Reasons for booking selection"""
    SINGLE_BOOKING = "single_booking"
    MOST_RECENT_UPCOMING = "most_recent_upcoming"
    DEPARTING_SOON = "departing_soon"
    REFUNDABLE_FARE = "refundable_fare"
    HIGHEST_VALUE = "highest_value"
    CUSTOMER_PREFERENCE = "customer_preference"
    FALLBACK_SELECTION = "fallback_selection"


@dataclass
class BookingScore:
    """Scoring information for booking selection"""
    booking: BookingDetails
    total_score: float
    priority: BookingPriority
    reason: SelectionReason
    factors: Dict[str, float]
    is_refundable: bool
    time_to_departure: timedelta
    estimated_value: float


class FareTypeClassifier:
    """Classifies fare types and determines refundability"""
    
    # Fare type patterns and their characteristics
    FARE_PATTERNS = {
        'blue_basic': {
            'patterns': ['basic', 'blue basic', 'economy basic'],
            'refundable': False,
            'changeable': False,
            'priority_score': 0.1
        },
        'blue': {
            'patterns': ['blue', 'economy', 'standard'],
            'refundable': True,
            'changeable': True,
            'priority_score': 0.3
        },
        'blue_plus': {
            'patterns': ['blue plus', 'plus', 'premium economy'],
            'refundable': True,
            'changeable': True,
            'priority_score': 0.6
        },
        'blue_extra': {
            'patterns': ['blue extra', 'extra', 'flexible'],
            'refundable': True,
            'changeable': True,
            'priority_score': 0.8
        },
        'mint': {
            'patterns': ['mint', 'business', 'first'],
            'refundable': True,
            'changeable': True,
            'priority_score': 1.0
        }
    }
    
    @classmethod
    def classify_fare_type(cls, booking: BookingDetails) -> Dict[str, Any]:
        """Classify fare type from booking information"""
        # Try to extract fare type from various sources
        fare_indicators = []
        
        # Check if booking has fare type information
        if hasattr(booking, 'fare_type') and booking.fare_type:
            fare_indicators.append(booking.fare_type.lower())
        
        # Check seat assignment for class indicators
        if booking.assigned_seat:
            seat = booking.assigned_seat.lower()
            if any(char in seat for char in ['f', 'j', 'c']):  # First/Business class seats
                fare_indicators.append('business')
            elif 'premium' in seat or 'plus' in seat:
                fare_indicators.append('plus')
        
        # Default classification logic based on patterns
        for fare_type, config in cls.FARE_PATTERNS.items():
            for pattern in config['patterns']:
                if any(pattern in indicator for indicator in fare_indicators):
                    return {
                        'fare_type': fare_type,
                        'refundable': config['refundable'],
                        'changeable': config['changeable'],
                        'priority_score': config['priority_score']
                    }
        
        # Default to standard Blue fare if no match
        return {
            'fare_type': 'blue',
            'refundable': True,
            'changeable': True,
            'priority_score': 0.3
        }


class BookingValueEstimator:
    """Estimates booking value for selection prioritization"""
    
    # Base values by route type (in USD)
    ROUTE_VALUES = {
        'domestic_short': 150,      # < 3 hours
        'domestic_medium': 250,     # 3-6 hours
        'domestic_long': 400,       # > 6 hours
        'international_short': 500, # < 6 hours
        'international_long': 800   # > 6 hours
    }
    
    # Fare type multipliers
    FARE_MULTIPLIERS = {
        'blue_basic': 0.8,
        'blue': 1.0,
        'blue_plus': 1.3,
        'blue_extra': 1.6,
        'mint': 3.0
    }
    
    @classmethod
    def estimate_booking_value(cls, booking: BookingDetails, fare_info: Dict[str, Any]) -> float:
        """Estimate the monetary value of a booking"""
        # Determine route type
        route_type = cls._classify_route(booking)
        base_value = cls.ROUTE_VALUES.get(route_type, 250)
        
        # Apply fare type multiplier
        fare_type = fare_info.get('fare_type', 'blue')
        multiplier = cls.FARE_MULTIPLIERS.get(fare_type, 1.0)
        
        # Calculate flight duration factor
        duration = booking.scheduled_arrival - booking.scheduled_departure
        duration_hours = duration.total_seconds() / 3600
        
        # Longer flights generally more expensive
        duration_factor = min(1.5, 1.0 + (duration_hours - 2) * 0.1)
        
        estimated_value = base_value * multiplier * duration_factor
        
        return max(50, estimated_value)  # Minimum value of $50
    
    @classmethod
    def _classify_route(cls, booking: BookingDetails) -> str:
        """Classify route type based on airports"""
        source = booking.source_airport_code
        dest = booking.destination_airport_code
        
        # Simple international detection (different first letters)
        is_international = source[0] != dest[0] if len(source) >= 1 and len(dest) >= 1 else False
        
        # Calculate flight duration
        duration = booking.scheduled_arrival - booking.scheduled_departure
        duration_hours = duration.total_seconds() / 3600
        
        if is_international:
            return 'international_long' if duration_hours > 6 else 'international_short'
        else:
            if duration_hours < 3:
                return 'domestic_short'
            elif duration_hours < 6:
                return 'domestic_medium'
            else:
                return 'domestic_long'


class BookingSelector:
    """Intelligent booking selector with priority rules and business logic"""
    
    def __init__(self):
        self.fare_classifier = FareTypeClassifier()
        self.value_estimator = BookingValueEstimator()
    
    def select_best_booking(
        self, 
        bookings: List[BookingDetails], 
        context: Optional[Dict[str, Any]] = None,
        customer_preferences: Optional[Dict[str, Any]] = None
    ) -> Optional[BookingScore]:
        """Select the best booking from multiple options"""
        
        if not bookings:
            return None
        
        if len(bookings) == 1:
            return self._score_booking(
                bookings[0], 
                SelectionReason.SINGLE_BOOKING,
                context,
                customer_preferences
            )
        
        # Score all bookings
        scored_bookings = []
        for booking in bookings:
            score = self._score_booking(booking, None, context, customer_preferences)
            scored_bookings.append(score)
        
        # Sort by total score (descending)
        scored_bookings.sort(key=lambda x: x.total_score, reverse=True)
        
        # Determine selection reason for the best booking
        best_booking = scored_bookings[0]
        best_booking.reason = self._determine_selection_reason(best_booking, scored_bookings)
        
        return best_booking
    
    def _score_booking(
        self, 
        booking: BookingDetails, 
        reason: Optional[SelectionReason] = None,
        context: Optional[Dict[str, Any]] = None,
        customer_preferences: Optional[Dict[str, Any]] = None
    ) -> BookingScore:
        """Calculate comprehensive score for a booking"""
        
        now = datetime.now()
        time_to_departure = booking.scheduled_departure - now
        
        # Classify fare type and get characteristics
        fare_info = self.fare_classifier.classify_fare_type(booking)
        
        # Estimate booking value
        estimated_value = self.value_estimator.estimate_booking_value(booking, fare_info)
        
        # Determine priority based on departure time
        priority = self._calculate_priority(time_to_departure)
        
        # Calculate individual scoring factors
        factors = {}
        
        # 1. Time urgency score (0-1)
        factors['urgency'] = self._calculate_urgency_score(time_to_departure)
        
        # 2. Refundability score (0-1)
        factors['refundability'] = 1.0 if fare_info['refundable'] else 0.2
        
        # 3. Fare type priority score (0-1)
        factors['fare_priority'] = fare_info['priority_score']
        
        # 4. Value score (0-1, normalized)
        factors['value'] = min(1.0, estimated_value / 1000)
        
        # 5. Status score (0-1)
        factors['status'] = self._calculate_status_score(booking)
        
        # 6. Customer preference score (0-1)
        factors['preference'] = self._calculate_preference_score(
            booking, customer_preferences
        )
        
        # 7. Context relevance score (0-1)
        factors['context'] = self._calculate_context_score(booking, context)
        
        # Calculate weighted total score
        weights = {
            'urgency': 0.25,
            'refundability': 0.20,
            'fare_priority': 0.15,
            'value': 0.15,
            'status': 0.10,
            'preference': 0.10,
            'context': 0.05
        }
        
        total_score = sum(factors[factor] * weights[factor] for factor in factors)
        
        return BookingScore(
            booking=booking,
            total_score=total_score,
            priority=priority,
            reason=reason or SelectionReason.FALLBACK_SELECTION,
            factors=factors,
            is_refundable=fare_info['refundable'],
            time_to_departure=time_to_departure,
            estimated_value=estimated_value
        )
    
    def _calculate_priority(self, time_to_departure: timedelta) -> BookingPriority:
        """Calculate priority based on time to departure"""
        hours = time_to_departure.total_seconds() / 3600
        
        if hours < 0:
            return BookingPriority.PAST
        elif hours < 24:
            return BookingPriority.CRITICAL
        elif hours < 168:  # 7 days
            return BookingPriority.HIGH
        elif hours < 720:  # 30 days
            return BookingPriority.MEDIUM
        else:
            return BookingPriority.LOW
    
    def _calculate_urgency_score(self, time_to_departure: timedelta) -> float:
        """Calculate urgency score based on departure time"""
        hours = time_to_departure.total_seconds() / 3600
        
        if hours < 0:
            return 0.0  # Past flights have no urgency
        elif hours < 2:
            return 1.0  # Extremely urgent
        elif hours < 24:
            return 0.8  # Very urgent
        elif hours < 168:  # 7 days
            return 0.6  # Moderately urgent
        elif hours < 720:  # 30 days
            return 0.4  # Somewhat urgent
        else:
            return 0.2  # Low urgency
    
    def _calculate_status_score(self, booking: BookingDetails) -> float:
        """Calculate score based on flight status"""
        status = booking.current_status.lower()
        
        if 'cancelled' in status:
            return 0.0
        elif 'delayed' in status:
            return 0.7
        elif 'on time' in status or 'scheduled' in status:
            return 1.0
        else:
            return 0.8  # Unknown status
    
    def _calculate_preference_score(
        self, 
        booking: BookingDetails, 
        preferences: Optional[Dict[str, Any]]
    ) -> float:
        """Calculate score based on customer preferences"""
        if not preferences:
            return 0.5  # Neutral score
        
        score = 0.5
        
        # Preferred routes
        preferred_routes = preferences.get('preferred_routes', [])
        route = f"{booking.source_airport_code}-{booking.destination_airport_code}"
        if route in preferred_routes:
            score += 0.3
        
        # Preferred times
        preferred_times = preferences.get('preferred_departure_times', [])
        departure_hour = booking.scheduled_departure.hour
        if any(abs(departure_hour - pref_hour) <= 2 for pref_hour in preferred_times):
            score += 0.2
        
        # Seat preferences
        preferred_seats = preferences.get('preferred_seat_types', [])
        if booking.assigned_seat and any(pref in booking.assigned_seat.lower() for pref in preferred_seats):
            score += 0.1
        
        return min(1.0, score)
    
    def _calculate_context_score(
        self, 
        booking: BookingDetails, 
        context: Optional[Dict[str, Any]]
    ) -> float:
        """Calculate score based on request context"""
        if not context:
            return 0.5
        
        score = 0.5
        
        # If specific PNR mentioned in context
        mentioned_pnr = context.get('mentioned_pnr')
        if mentioned_pnr and mentioned_pnr.upper() == booking.pnr.upper():
            score = 1.0
        
        # If specific flight number mentioned
        mentioned_flight = context.get('mentioned_flight_number')
        if mentioned_flight and hasattr(booking, 'flight_number'):
            if mentioned_flight.upper() in str(booking.flight_id).upper():
                score += 0.3
        
        # If specific route mentioned
        mentioned_route = context.get('mentioned_route')
        if mentioned_route:
            booking_route = f"{booking.source_airport_code}-{booking.destination_airport_code}"
            if mentioned_route.upper() in booking_route.upper():
                score += 0.2
        
        return min(1.0, score)
    
    def _determine_selection_reason(
        self, 
        best_booking: BookingScore, 
        all_bookings: List[BookingScore]
    ) -> SelectionReason:
        """Determine the primary reason for selecting this booking"""
        
        # Check if it's the only upcoming flight
        upcoming_bookings = [b for b in all_bookings if b.time_to_departure.total_seconds() > 0]
        if len(upcoming_bookings) == 1 and best_booking in upcoming_bookings:
            return SelectionReason.MOST_RECENT_UPCOMING
        
        # Check if departing very soon (critical priority)
        if best_booking.priority == BookingPriority.CRITICAL:
            return SelectionReason.DEPARTING_SOON
        
        # Check if selected primarily for refundability
        if best_booking.is_refundable and best_booking.factors['refundability'] > 0.8:
            non_refundable_count = sum(1 for b in all_bookings if not b.is_refundable)
            if non_refundable_count > 0:
                return SelectionReason.REFUNDABLE_FARE
        
        # Check if selected for high value
        if best_booking.estimated_value > 500 and best_booking.factors['value'] > 0.7:
            return SelectionReason.HIGHEST_VALUE
        
        # Check if customer preference was a major factor
        if best_booking.factors['preference'] > 0.8:
            return SelectionReason.CUSTOMER_PREFERENCE
        
        return SelectionReason.FALLBACK_SELECTION
    
    def should_auto_cancel(
        self, 
        booking_score: BookingScore, 
        context: Optional[Dict[str, Any]] = None
    ) -> Tuple[bool, str]:
        """Determine if booking should be automatically cancelled"""
        
        booking = booking_score.booking
        
        # Never auto-cancel past flights
        if booking_score.time_to_departure.total_seconds() < 0:
            return False, "Cannot cancel past flights"
        
        # Never auto-cancel flights departing within 2 hours
        if booking_score.time_to_departure.total_seconds() < 7200:  # 2 hours
            return False, "Too close to departure time for automatic cancellation"
        
        # Check if flight is already cancelled
        if 'cancelled' in booking.current_status.lower():
            return False, "Flight is already cancelled"
        
        # Auto-cancel conditions
        auto_cancel_reasons = []
        
        # 1. High confidence selection with refundable fare
        if booking_score.total_score > 0.8 and booking_score.is_refundable:
            auto_cancel_reasons.append("High confidence match with refundable fare")
        
        # 2. Explicit PNR match in context
        if context and context.get('mentioned_pnr') == booking.pnr:
            auto_cancel_reasons.append("Explicit PNR match in customer request")
        
        # 3. Single upcoming booking scenario
        if booking_score.reason == SelectionReason.MOST_RECENT_UPCOMING:
            auto_cancel_reasons.append("Only upcoming booking for customer")
        
        # 4. Customer has history of cancellations (if available in context)
        customer_history = context.get('customer_history', {}) if context else {}
        if customer_history.get('frequent_canceller', False):
            auto_cancel_reasons.append("Customer history indicates frequent cancellations")
        
        # Require at least one strong reason for auto-cancellation
        if auto_cancel_reasons:
            return True, "; ".join(auto_cancel_reasons)
        
        return False, "Insufficient confidence for automatic cancellation"
    
    def get_selection_explanation(self, booking_score: BookingScore) -> str:
        """Generate human-readable explanation for booking selection"""
        booking = booking_score.booking
        
        explanation_parts = []
        
        # Primary reason
        reason_explanations = {
            SelectionReason.SINGLE_BOOKING: "This is your only booking",
            SelectionReason.MOST_RECENT_UPCOMING: "This is your most recent upcoming flight",
            SelectionReason.DEPARTING_SOON: f"This flight departs soon ({booking.scheduled_departure.strftime('%Y-%m-%d %H:%M')})",
            SelectionReason.REFUNDABLE_FARE: "This booking has a refundable fare type",
            SelectionReason.HIGHEST_VALUE: f"This is your highest value booking (estimated ${booking_score.estimated_value:.0f})",
            SelectionReason.CUSTOMER_PREFERENCE: "This booking matches your preferences",
            SelectionReason.FALLBACK_SELECTION: "This booking was selected based on multiple factors"
        }
        
        explanation_parts.append(reason_explanations.get(
            booking_score.reason, 
            "This booking was automatically selected"
        ))
        
        # Add key factors
        key_factors = []
        if booking_score.factors['urgency'] > 0.7:
            key_factors.append("departing soon")
        if booking_score.is_refundable:
            key_factors.append("refundable fare")
        if booking_score.factors['value'] > 0.7:
            key_factors.append("high value")
        
        if key_factors:
            explanation_parts.append(f"Key factors: {', '.join(key_factors)}")
        
        # Add booking details
        route = f"{booking.source_airport_code} to {booking.destination_airport_code}"
        departure = booking.scheduled_departure.strftime('%Y-%m-%d %H:%M')
        explanation_parts.append(f"Flight: {route} on {departure} (PNR: {booking.pnr})")
        
        return ". ".join(explanation_parts) + "."


# Global booking selector instance
booking_selector = BookingSelector()