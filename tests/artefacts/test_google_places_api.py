import os
import pytest
import httpx
import json
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

API_KEY = os.getenv('GOOGLE_PLACES_API_KEY')
BASE_URL = 'https://places.googleapis.com/v1'

# Mock response data for Places API v1
MOCK_RESPONSES = {
    "search": {
        "places": [
            {
                "name": "places/ChIJN1t_tDeuEmsRUsoyG83frY4",
                "displayName": {
                    "text": "Eiffel Tower",
                    "languageCode": "en"
                },
                "location": {
                    "latitude": 48.858372,
                    "longitude": 2.294481
                },
                "rating": 4.5,
                "userRatingCount": 25000,
                "primaryType": "tourist_attraction",
                "primaryTypeDisplayName": {
                    "text": "Tourist Attraction",
                    "languageCode": "en"
                }
            }
        ]
    },
    "place_details": {
        "name": "places/ChIJN1t_tDeuEmsRUsoyG83frY4",
        "id": "ChIJN1t_tDeuEmsRUsoyG83frY4",
        "types": ["tourist_attraction", "point_of_interest"],
        "nationalPhoneNumber": "+33 892 70 12 39",
        "internationalPhoneNumber": "+33 892 70 12 39",
        "formattedAddress": "Champ de Mars, 5 Avenue Anatole France, 75007 Paris, France",
        "addressComponents": [
            {
                "longText": "Champ de Mars",
                "shortText": "Champ de Mars",
                "types": ["route"],
                "languageCode": "en"
            }
        ],
        "location": {
            "latitude": 48.858372,
            "longitude": 2.294481
        },
        "rating": 4.5,
        "googleMapsUri": "https://maps.google.com/?cid=1234567890"
    }
}

@pytest.fixture
def api_headers():
    """Set up API headers for Google Places API v1"""
    api_key = os.getenv('GOOGLE_PLACES_API_KEY')
    logger.info(f"API Key length: {len(api_key) if api_key else 'None'}")
    logger.info(f"API Key format: {api_key[:5]}...{api_key[-5:] if api_key else ''}")
    
    return {
        'Content-Type': 'application/json',
        'X-Goog-Api-Key': api_key,
        'X-Goog-FieldMask': '*'  # Request all fields
    }

@pytest.fixture
def mock_client():
    """Create a mock client that returns predefined responses"""
    class MockResponse:
        def __init__(self, data, status_code=200):
            self._data = data
            self.status_code = status_code

        async def json(self):
            return self._data

    class MockClient:
        async def post(self, url, **kwargs):
            if ":searchNearby" in url:
                return MockResponse(MOCK_RESPONSES["search"])
            return MockResponse({"error": "Not found"}, 404)

        async def get(self, url, **kwargs):
            if "/places/" in url:
                return MockResponse(MOCK_RESPONSES["place_details"])
            return MockResponse({"error": "Not found"}, 404)

    return MockClient()

@pytest.mark.mock
async def test_nearby_search_mock(mock_client):
    """Test nearby search with mock data"""
    payload = {
        "locationRestriction": {
            "circle": {
                "center": {
                    "latitude": 48.858372,
                    "longitude": 2.294481
                },
                "radius": 5000.0
            }
        },
        "includedTypes": ["tourist_attraction"]
    }
    
    response = await mock_client.post(f'{BASE_URL}/places:searchNearby', json=payload)
    assert response.status_code == 200
    
    data = await response.json()
    assert "places" in data
    assert len(data["places"]) > 0
    
    place = data["places"][0]
    assert all(key in place for key in [
        "name", "displayName", "location", "rating"
    ])

@pytest.mark.mock
async def test_place_details_mock(mock_client):
    """Test place details with mock data"""
    place_name = "places/ChIJN1t_tDeuEmsRUsoyG83frY4"
    response = await mock_client.get(f'{BASE_URL}/{place_name}')
    assert response.status_code == 200
    
    data = await response.json()
    assert data["name"] == place_name
    assert "formattedAddress" in data
    assert "location" in data

@pytest.mark.real
@pytest.mark.skipif(
    not os.getenv('GOOGLE_PLACES_API_KEY'),
    reason="Google Places API key not configured"
)
async def test_nearby_search(api_headers):
    """Test nearby search endpoint with real API"""
    async with httpx.AsyncClient() as client:
        api_key = os.getenv('GOOGLE_PLACES_API_KEY')
        if not api_key:
            pytest.skip("GOOGLE_PLACES_API_KEY not found in environment")
            
        payload = {
            "includedTypes": ["tourist_attraction"],
            "locationRestriction": {
                "circle": {
                    "center": {
                        "latitude": 48.858372,
                        "longitude": 2.294481
                    },
                    "radius": 5000.0
                }
            },
            "maxResultCount": 10,
            "languageCode": "en"
        }
        
        url = f'{BASE_URL}/places:searchNearby'
        
        response = await client.post(
            url,
            headers=api_headers,
            json=payload
        )
        
        assert response.status_code == 200
        
        try:
            data = response.json()  # No need to await here
            
            # Verify response structure
            assert "places" in data, "Response should contain 'places' array"
            assert isinstance(data["places"], list), "'places' should be an array"
            assert len(data["places"]) > 0, "Should return at least one place"
            
            # Verify first place structure
            place = data["places"][0]
            required_fields = {
                "name": str,
                "displayName": dict,
                "location": dict,
                "types": list
            }
            
            for field, expected_type in required_fields.items():
                assert field in place, f"Place should contain '{field}'"
                assert isinstance(place[field], expected_type), f"'{field}' should be {expected_type.__name__}"
            
            # Verify location structure
            location = place["location"]
            assert "latitude" in location, "Location should contain latitude"
            assert "longitude" in location, "Location should contain longitude"
            assert isinstance(location["latitude"], (int, float)), "Latitude should be numeric"
            assert isinstance(location["longitude"], (int, float)), "Longitude should be numeric"
            
            # Verify displayName structure
            display_name = place["displayName"]
            assert "text" in display_name, "displayName should contain text"
            assert "languageCode" in display_name, "displayName should contain languageCode"
            assert display_name["languageCode"] == "en", "Language code should match request"
            
        except Exception as e:
            logger.error(f"Error parsing response: {str(e)}")
            logger.error(f"Response text: {response.text}")
            raise

@pytest.mark.real
@pytest.mark.skipif(
    not os.getenv('GOOGLE_PLACES_API_KEY'),
    reason="Google Places API key not configured"
)
async def test_place_details(api_headers):
    """Test place details endpoint with real API"""
    async with httpx.AsyncClient() as client:
        api_key = os.getenv('GOOGLE_PLACES_API_KEY')
        if not api_key:
            pytest.skip("GOOGLE_PLACES_API_KEY not found in environment")
            
        # Eiffel Tower
        place_id = "places/ChIJN1t_tDeuEmsRUsoyG83frY4"
        
        url = f'{BASE_URL}/{place_id}'
        
        response = await client.get(
            url,
            headers=api_headers
        )
        
        assert response.status_code == 200
        
        try:
            data = response.json()  # No need to await here
            
            # Verify required fields
            required_fields = {
                "name": str,
                "id": str,
                "location": dict,
                "types": list,
                "displayName": dict,
                "primaryType": str,
                "formattedAddress": str
            }
            
            for field, expected_type in required_fields.items():
                assert field in data, f"Response should contain '{field}'"
                assert isinstance(data[field], expected_type), f"'{field}' should be {expected_type.__name__}"
            
            # Verify location structure
            location = data["location"]
            assert "latitude" in location, "Location should contain latitude"
            assert "longitude" in location, "Location should contain longitude"
            assert isinstance(location["latitude"], (int, float)), "Latitude should be numeric"
            assert isinstance(location["longitude"], (int, float)), "Longitude should be numeric"
            
            # Verify display name
            display_name = data["displayName"]
            assert "text" in display_name, "displayName should contain text"
            assert "languageCode" in display_name, "displayName should contain languageCode"
            
            # Optional but common fields
            optional_fields = [
                "rating",
                "userRatingCount",
                "googleMapsUri",
                "internationalPhoneNumber",
                "nationalPhoneNumber",
                "websiteUri",
                "regularOpeningHours"
            ]
            
            found_optional = [field for field in optional_fields if field in data]
            logger.info(f"Found optional fields: {found_optional}")
            
            # If rating exists, verify its structure
            if "rating" in data:
                assert isinstance(data["rating"], (int, float)), "Rating should be numeric"
                assert 1 <= data["rating"] <= 5, "Rating should be between 1 and 5"
                
        except Exception as e:
            logger.error(f"Error parsing response: {str(e)}")
            logger.error(f"Response text: {response.text}")
            raise 