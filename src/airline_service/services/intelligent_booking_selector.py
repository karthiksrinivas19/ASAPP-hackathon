"""
Intelligent booking selector for automatic data retrieval
"""

from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum

from ..types import BookingDetails, RequestType, CustomerProfile


class SelectionCriteria(str, Enum):
    """Criteria for booking selection"""
    MOST_RECENT_UPCOMING = "most_recent_upcoming"
    SOONEST_DEPARTURE = "soonest_departure"
    MOST_RECENT_BOOKING = "most_recent_booking"
    HIGHEST_VALUE = "highest_value"
    REFUNDABLE_FIRST = "refundable_first"
    CUSTOMER_PREFERENCE = "customer_preference"


@dataclass
class BookingScore:
    """Scoring result for a booking"""
    booking: BookingDetails
    score: float
    reasons: List[str]
    selection_criteria: List[SelectionCriteria]


@dataclass
class SelectionContext:
    """Context for booking selection"""
    request_type: RequestType
    customer_preferences: Optional[Dict[str, Any]] = None
    urgency_level: str = "normal"  # low, normal, high
    time_sensitivity: bool = False
    prefer_upcoming: bool = True
    prefer_refundable: bool = False


class IntelligentBookingSelector:
    """Intelligent booking selector with multiple selection strategies"""
    
    def __init__(self):
        self.selection_weights = {
            SelectionCriteria.MOST_RECENT_UPCOMING: 1.0,
            SelectionCriteria.SOONEST_DEPARTURE: 0.8,
            SelectionCriteria.MOST_RECENT_BOOKING: 0.6,
            SelectionCriteria.HIGHEST_VALUE: 0.4,
            SelectionCriteria.REFUNDABLE_FIRST: 0.7,
            SelectionCriteria.CUSTOMER_PREFERENCE: 1.2
        }
    
    def select_best_booking(
        self, 
        bookings: List[BookingDetails], 
        context: SelectionContext
    ) -> Optional[BookingScore]:
        """Select the best booking from a list based on context"""
        
        if not bookings:
            return None
        
        if len(bookings) == 1:
            return BookingScore(
                booking=bookings[0],
                score=1.0,
                reasons=["Only booking available"],
                selection_criteria=[SelectionCriteria.MOST_RECENT_BOOKING]
            )
        
        # Score all bookings
        scored_bookings = []
        for booking in bookings:
            score_result = self._score_booking(booking, context, bookings)
            scored_bookings.append(score_result)
        
        # Sort by score (highest first)
        scored_bookings.sort(key=lambda x: x.score, reverse=True)
        
        return scored_bookings[0]
    
    def select_booking_for_cancellation(
        self, 
        bookings: List[BookingDetails],
        customer_preferences: Optional[Dict[str, Any]] = None
    ) -> Optional[BookingScore]:
        """Select booking specifically for cancellation requests"""
        
        context = SelectionContext(
            request_type=RequestType.CANCEL_TRIP,
            customer_preferences=customer_preferences,
            prefer_upcoming=True,
            prefer_refundable=True,
            time_sensitivity=True
        )
        
        return self.select_best_booking(bookings, context)
    
    def select_booking_for_status_check(
        self, 
        bookings: List[BookingDetails],
        customer_preferences: Optional[Dict[str, Any]] = None
    ) -> Optional[BookingScore]:
        """Select booking specifically for status check requests"""
        
        context = SelectionContext(
            request_type=RequestType.FLIGHT_STATUS,
            customer_preferences=customer_preferences,
            prefer_upcoming=True,
            time_sensitivity=True
        )
        
        return self.select_best_booking(bookings, context)
    
    def select_booking_for_seat_availability(
        self, 
        bookings: List[BookingDetails],
        customer_preferences: Optional[Dict[str, Any]] = None
    ) -> Optional[BookingScore]:
        """Select booking specifically for seat availability requests"""
        
        context = SelectionContext(
            request_type=RequestType.SEAT_AVAILABILITY,
            customer_preferences=customer_preferences,
            prefer_upcoming=True,
            time_sensitivity=True
        )
        
        return self.select_best_booking(bookings, context)
    
    def _score_booking(
        self, 
        booking: BookingDetails, 
        context: SelectionContext,
        all_bookings: List[BookingDetails]
    ) -> BookingScore:
        """Score a booking based on selection context"""
        
        score = 0.0
        reasons = []
        criteria_used = []
        
        now = datetime.now()
        
        # 1. Time-based scoring
        time_score, time_reasons, time_criteria = self._score_time_factors(booking, context, now)
        score += time_score
        reasons.extend(time_reasons)
        criteria_used.extend(time_criteria)
        
        # 2. Request type specific scoring
        request_score, request_reasons, request_criteria = self._score_request_type_factors(
            booking, context, now
        )
        score += request_score
        reasons.extend(request_reasons)
        criteria_used.extend(request_criteria)
        
        # 3. Customer preference scoring
        pref_score, pref_reasons, pref_criteria = self._score_customer_preferences(
            booking, context
        )
        score += pref_score
        reasons.extend(pref_reasons)
        criteria_used.extend(pref_criteria)
        
        # 4. Relative scoring (compared to other bookings)
        rel_score, rel_reasons, rel_criteria = self._score_relative_factors(
            booking, all_bookings, context
        )
        score += rel_score
        reasons.extend(rel_reasons)
        criteria_used.extend(rel_criteria)
        
        return BookingScore(
            booking=booking,
            score=score,
            reasons=reasons,
            selection_criteria=list(set(criteria_used))
        )
    
    def _score_time_factors(
        self, 
        booking: BookingDetails, 
        context: SelectionContext,
        now: datetime
    ) -> Tuple[float, List[str], List[SelectionCriteria]]:
        """Score booking based on time factors"""
        
        score = 0.0
        reasons = []
        criteria = []
        
        time_until_departure = booking.scheduled_departure - now
        
        # Upcoming vs past flights
        if booking.scheduled_departure > now:
            if context.prefer_upcoming:
                score += 2.0
                reasons.append("Upcoming flight preferred")
                criteria.append(SelectionCriteria.MOST_RECENT_UPCOMING)
            
            # Score based on how soon the flight is
            if time_until_departure.days <= 1:
                score += 1.5
                reasons.append("Flight departing within 24 hours")
                criteria.append(SelectionCriteria.SOONEST_DEPARTURE)
            elif time_until_departure.days <= 7:
                score += 1.0
                reasons.append("Flight departing within a week")
                criteria.append(SelectionCriteria.SOONEST_DEPARTURE)
            elif time_until_departure.days <= 30:
                score += 0.5
                reasons.append("Flight departing within a month")
        else:
            # Past flight - lower score but still relevant for some requests
            score += 0.2
            reasons.append("Past flight")
        
        # Time sensitivity bonus
        if context.time_sensitivity and booking.scheduled_departure > now:
            if time_until_departure.days <= 3:
                score += 1.0
                reasons.append("Time-sensitive request with near-term flight")
        
        return score, reasons, criteria
    
    def _score_request_type_factors(
        self, 
        booking: BookingDetails, 
        context: SelectionContext,
        now: datetime
    ) -> Tuple[float, List[str], List[SelectionCriteria]]:
        """Score booking based on request type specific factors"""
        
        score = 0.0
        reasons = []
        criteria = []
        
        if context.request_type == RequestType.CANCEL_TRIP:
            # For cancellations, prefer upcoming flights
            if booking.scheduled_departure > now:
                score += 2.0
                reasons.append("Upcoming flight suitable for cancellation")
                criteria.append(SelectionCriteria.MOST_RECENT_UPCOMING)
            
            # Prefer flights that can be cancelled (not departed)
            if booking.current_status not in ["Departed", "Arrived", "Completed"]:
                score += 1.5
                reasons.append("Flight can still be cancelled")
            
            # Prefer refundable bookings if context suggests it
            if context.prefer_refundable:
                # In a real system, we'd check fare rules
                # For now, assume newer bookings are more likely refundable
                booking_age = now - booking.scheduled_departure
                if booking_age.days < 30:
                    score += 1.0
                    reasons.append("Recent booking likely refundable")
                    criteria.append(SelectionCriteria.REFUNDABLE_FIRST)
        
        elif context.request_type == RequestType.FLIGHT_STATUS:
            # For status checks, prefer upcoming flights
            if booking.scheduled_departure > now:
                time_until = booking.scheduled_departure - now
                if time_until.days <= 1:
                    score += 2.0
                    reasons.append("Flight status most relevant for near-term departure")
                    criteria.append(SelectionCriteria.SOONEST_DEPARTURE)
                elif time_until.days <= 7:
                    score += 1.0
                    reasons.append("Flight status relevant for upcoming departure")
        
        elif context.request_type == RequestType.SEAT_AVAILABILITY:
            # For seat changes, only upcoming flights are relevant
            if booking.scheduled_departure > now:
                if booking.current_status in ["Confirmed", "On Time", "Delayed"]:
                    score += 2.0
                    reasons.append("Flight suitable for seat changes")
                    criteria.append(SelectionCriteria.MOST_RECENT_UPCOMING)
            else:
                score -= 1.0  # Past flights can't have seat changes
                reasons.append("Past flight not suitable for seat changes")
        
        return score, reasons, criteria
    
    def _score_customer_preferences(
        self, 
        booking: BookingDetails, 
        context: SelectionContext
    ) -> Tuple[float, List[str], List[SelectionCriteria]]:
        """Score booking based on customer preferences"""
        
        score = 0.0
        reasons = []
        criteria = []
        
        if not context.customer_preferences:
            return score, reasons, criteria
        
        prefs = context.customer_preferences
        
        # Preferred routes
        if "preferred_routes" in prefs:
            route = f"{booking.source_airport_code}-{booking.destination_airport_code}"
            if route in prefs["preferred_routes"]:
                score += 1.0
                reasons.append("Matches preferred route")
                criteria.append(SelectionCriteria.CUSTOMER_PREFERENCE)
        
        # Preferred airports
        if "preferred_airports" in prefs:
            if (booking.source_airport_code in prefs["preferred_airports"] or 
                booking.destination_airport_code in prefs["preferred_airports"]):
                score += 0.5
                reasons.append("Involves preferred airport")
                criteria.append(SelectionCriteria.CUSTOMER_PREFERENCE)
        
        # Seat preferences (for seat availability requests)
        if context.request_type == RequestType.SEAT_AVAILABILITY:
            if "seat_preference" in prefs:
                # This would be enhanced with actual seat data
                score += 0.3
                reasons.append("Booking suitable for seat preference")
                criteria.append(SelectionCriteria.CUSTOMER_PREFERENCE)
        
        return score, reasons, criteria
    
    def _score_relative_factors(
        self, 
        booking: BookingDetails, 
        all_bookings: List[BookingDetails],
        context: SelectionContext
    ) -> Tuple[float, List[str], List[SelectionCriteria]]:
        """Score booking relative to other bookings"""
        
        score = 0.0
        reasons = []
        criteria = []
        
        if len(all_bookings) <= 1:
            return score, reasons, criteria
        
        now = datetime.now()
        
        # Find the most recent booking
        most_recent = max(all_bookings, key=lambda b: b.scheduled_departure)
        if booking == most_recent:
            score += 0.5
            reasons.append("Most recent booking")
            criteria.append(SelectionCriteria.MOST_RECENT_BOOKING)
        
        # Find the soonest upcoming departure
        upcoming_bookings = [b for b in all_bookings if b.scheduled_departure > now]
        if upcoming_bookings:
            soonest_upcoming = min(upcoming_bookings, key=lambda b: b.scheduled_departure)
            if booking == soonest_upcoming:
                score += 1.0
                reasons.append("Soonest upcoming departure")
                criteria.append(SelectionCriteria.SOONEST_DEPARTURE)
        
        # Estimate value (simplified - in real system would use fare data)
        # For now, use flight distance as proxy
        booking_distance = self._estimate_flight_distance(booking)
        max_distance = max(self._estimate_flight_distance(b) for b in all_bookings)
        
        if max_distance > 0 and booking_distance == max_distance:
            score += 0.3
            reasons.append("Highest value booking (estimated)")
            criteria.append(SelectionCriteria.HIGHEST_VALUE)
        
        return score, reasons, criteria
    
    def _estimate_flight_distance(self, booking: BookingDetails) -> float:
        """Estimate flight distance (simplified)"""
        # In a real system, this would use airport coordinates
        # For now, use a simple heuristic based on airport codes
        
        source = booking.source_airport_code
        dest = booking.destination_airport_code
        
        # Simple distance estimation based on common routes
        long_haul_routes = [
            ("JFK", "LAX"), ("LAX", "JFK"),
            ("ORD", "LAX"), ("LAX", "ORD"),
            ("JFK", "SFO"), ("SFO", "JFK")
        ]
        
        if (source, dest) in long_haul_routes or (dest, source) in long_haul_routes:
            return 2500.0  # Long haul
        
        # Check for cross-country patterns
        east_coast = ["JFK", "BOS", "DCA", "ATL", "MIA"]
        west_coast = ["LAX", "SFO", "SEA", "PDX"]
        
        if ((source in east_coast and dest in west_coast) or 
            (source in west_coast and dest in east_coast)):
            return 2000.0  # Cross country
        
        return 1000.0  # Default medium distance
    
    def get_selection_explanation(self, score_result: BookingScore) -> str:
        """Get human-readable explanation for booking selection"""
        
        if not score_result:
            return "No booking selected"
        
        booking = score_result.booking
        reasons = score_result.reasons
        
        explanation = f"Selected booking {booking.pnr} "
        explanation += f"({booking.source_airport_code} to {booking.destination_airport_code}) "
        explanation += f"departing {booking.scheduled_departure.strftime('%Y-%m-%d %H:%M')}.\n\n"
        
        explanation += "Selection reasons:\n"
        for i, reason in enumerate(reasons, 1):
            explanation += f"{i}. {reason}\n"
        
        explanation += f"\nSelection score: {score_result.score:.2f}"
        
        return explanation
    
    def update_selection_weights(self, weights: Dict[SelectionCriteria, float]) -> None:
        """Update selection criteria weights"""
        self.selection_weights.update(weights)
    
    def get_selection_weights(self) -> Dict[SelectionCriteria, float]:
        """Get current selection criteria weights"""
        return self.selection_weights.copy()


# Factory function for easy instantiation
def create_booking_selector() -> IntelligentBookingSelector:
    """Create an intelligent booking selector instance"""
    return IntelligentBookingSelector()