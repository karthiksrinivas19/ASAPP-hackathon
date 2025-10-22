"""
Main application entry point
"""

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from contextlib import asynccontextmanager
import structlog
from datetime import datetime

from .config import config
from .types import CustomerRequest, APIResponse


# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer() if config.logging.log_format == "json" else structlog.dev.ConsoleRenderer(),
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management"""
    # Startup
    logger.info("Starting Airline Customer Service API", version="1.0.0")
    
    # TODO: Initialize ML models, database connections, etc.
    # await initialize_ml_models()
    # await initialize_database()
    # await initialize_cache()
    
    yield
    
    # Shutdown
    logger.info("Shutting down Airline Customer Service API")
    
    # TODO: Cleanup resources
    # await cleanup_resources()


# Create FastAPI application
app = FastAPI(
    title="Airline Customer Service API",
    description="Automated airline customer service system with ML-based request classification",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs" if config.server.debug else None,
    redoc_url="/redoc" if config.server.debug else None,
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if config.is_development else ["https://yourdomain.com"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["*"] if config.is_development else ["yourdomain.com", "*.yourdomain.com"]
)


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0",
        "environment": config.server.environment
    }


@app.post("/api/v1/customer-service/query", response_model=APIResponse)
async def process_customer_query(request: CustomerRequest):
    """
    Process customer query and return automated response
    
    This endpoint handles all customer service requests including:
    - Flight cancellations
    - Flight status inquiries  
    - Seat availability checks
    - Policy information requests
    - Pet travel inquiries
    """
    try:
        logger.info("Processing customer query", utterance=request.utterance)
        
        # Initialize classifier (in production, this would be a singleton)
        from .services.request_classifier_service import ClassifierFactory
        classifier = ClassifierFactory.create_classifier()
        
        if not classifier.is_loaded():
            # Use mock classifier if model not available
            classifier = ClassifierFactory.create_mock_classifier()
            logger.warning("Using mock classifier - model not found")
        
        # 1. Classify request using ML model
        classification_result = await classifier.classify_request(request.utterance)
        
        # 2. Extract entities (already done in classifier)
        entities = classification_result.extracted_entities
        
        # 3. Create response based on classification
        intent = classification_result.request_type
        confidence = classification_result.confidence
        
        # Generate appropriate response based on intent
        if intent == "cancel_trip":
            message = "I can help you cancel your flight. To proceed with the cancellation, I'll need your booking reference (PNR) or flight details."
            if entities:
                pnr_entities = [e for e in entities if e.type == "pnr"]
                if pnr_entities:
                    pnr = pnr_entities[0].value
                    message = f"I found your booking reference {pnr}. I can help you cancel this flight. Please note that cancellation fees may apply based on your fare type."
        
        elif intent == "flight_status":
            message = "I can check your flight status. Let me look up the current information for your flight."
            if entities:
                flight_entities = [e for e in entities if e.type == "flight_number"]
                pnr_entities = [e for e in entities if e.type == "pnr"]
                if flight_entities:
                    flight = flight_entities[0].value
                    message = f"Checking status for flight {flight}. Your flight is currently on time with no delays reported."
                elif pnr_entities:
                    pnr = pnr_entities[0].value
                    message = f"Checking status for booking {pnr}. Your flight is currently on time with no delays reported."
        
        elif intent == "seat_availability":
            message = "I can show you available seats on your flight. Let me check the current seat map."
            if entities:
                class_entities = [e for e in entities if e.type == "class"]
                seat_type_entities = [e for e in entities if e.type == "seat_type"]
                if class_entities:
                    seat_class = class_entities[0].value
                    message = f"Checking available seats in {seat_class} class. I found several options for you."
                elif seat_type_entities:
                    seat_type = seat_type_entities[0].value
                    message = f"Looking for {seat_type} seats. I found several {seat_type} seats available."
        
        elif intent == "cancellation_policy":
            message = "Our cancellation policy varies by fare type. Generally, you can cancel flights up to 24 hours before departure. Fees may apply depending on your ticket type."
        
        elif intent == "pet_travel":
            message = "I can help you with pet travel information. Small pets can travel in the cabin in approved carriers, while larger pets may need to travel as cargo. Service animals are always welcome."
            if entities:
                pet_entities = [e for e in entities if e.type == "pet_type"]
                if pet_entities:
                    pet_type = pet_entities[0].value
                    message = f"For {pet_type} travel, specific requirements apply. Small {pet_type}s can travel in cabin with proper carriers."
        
        else:
            message = "I understand you need assistance. Could you please provide more details about what you'd like to help with?"
        
        # 4. Return formatted response
        return APIResponse(
            status="completed",
            message=message,
            data={
                "intent": intent.value,
                "confidence": confidence,
                "entities": [
                    {
                        "type": entity.type.value,
                        "value": entity.value,
                        "confidence": entity.confidence
                    } for entity in entities
                ],
                "alternatives": [
                    {
                        "intent": alt["type"].value,
                        "confidence": alt["confidence"]
                    } for alt in classification_result.alternative_intents
                ]
            }
        )
        
    except Exception as e:
        logger.error("Error processing customer query", error=str(e), utterance=request.utterance)
        raise HTTPException(
            status_code=500,
            detail={
                "status": "error",
                "message": "Internal server error occurred while processing your request",
                "error_code": "INTERNAL_ERROR",
                "timestamp": datetime.now().isoformat()
            }
        )


@app.exception_handler(404)
async def not_found_handler(request, exc):
    """Handle 404 errors"""
    return {
        "status": "error",
        "message": "Endpoint not found",
        "timestamp": datetime.now().isoformat()
    }


@app.exception_handler(500)
async def internal_error_handler(request, exc):
    """Handle 500 errors"""
    logger.error("Internal server error", error=str(exc))
    return {
        "status": "error", 
        "message": "Internal server error",
        "timestamp": datetime.now().isoformat()
    }


def main():
    """Main entry point"""
    uvicorn.run(
        "airline_service.main:app",
        host=config.server.host,
        port=config.server.port,
        reload=config.is_development,
        log_level=config.logging.level.lower(),
        access_log=config.logging.enable_audit,
    )


if __name__ == "__main__":
    main()