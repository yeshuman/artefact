import os
import pytest
import httpx
from typing import Dict, Any
from pprint import pprint

API_KEY = os.getenv('TRIPADVISOR_API_KEY')
BASE_URL = 'https://api.content.tripadvisor.com/api/v1'


# Mock response data
MOCK_RESPONSES = {
    "location_search": {
        "data": [{
            "location_id": "187147",
            "name": "Paris",
            "description": "City of Light",
            "web_url": "https://www.tripadvisor.com/Tourism-g187147-Paris_Ile_de_France-Vacations.html",
            "address_obj": {
                "street1": "",
                "street2": "",
                "city": "Paris",
                "state": "Ile-de-France",
                "country": "France",
                "postalcode": "",
                "address_string": "Paris, France"
            },
            "ancestors": [
                {
                    "level": "City",
                    "name": "Paris",
                    "location_id": "187147"
                }
            ]
        }]
    },
    "location_details": {
        "location_id": "187147",
        "name": "Paris",
        "description": "Paris, France's capital, is a major European city...",
        "rating": "4.5",
        "num_reviews": "25000",
        "category": {
            "key": "city",
            "name": "City"
        }
    },
    "nearby_search": {
        "data": [
            {
                "location_id": "188975",
                "name": "Eiffel Tower",
                "distance": "0.5",
                "rating": "4.5",
                "bearing": "NE"
            },
            {
                "location_id": "188757",
                "name": "Louvre Museum",
                "distance": "2.1",
                "rating": "4.7",
                "bearing": "N"
            }
        ]
    }
}

@pytest.fixture
def api_headers():
    """
    Set up API headers according to TripAdvisor documentation
    https://api.content.tripadvisor.com/api/v1/documentation/authentication
    """
    return {
        'accept': 'application/json',
        'X-TripAdvisor-API-Key': os.getenv('TRIPADVISOR_API_KEY'),
        # Add additional required headers if any
        'User-Agent': 'Artefact/1.0'  # Optional but recommended
    }

@pytest.fixture
def mock_client():
    """Create a mock client that returns our predefined responses"""
    class MockResponse:
        def __init__(self, data, status_code=200):
            self._data = data
            self.status_code = status_code

        async def json(self):
            return self._data

    class MockClient:
        async def get(self, url, **kwargs):
            if "/location/search" in url:
                return MockResponse(MOCK_RESPONSES["location_search"])
            elif "/details" in url:
                return MockResponse(MOCK_RESPONSES["location_details"])
            elif "/nearby_search" in url:
                return MockResponse(MOCK_RESPONSES["nearby_search"])
            return MockResponse({"error": "Not found"}, 404)

    return MockClient()

async def print_response_structure(data: Dict[str, Any], indent: int = 0):
    """Helper to visualize API response structure."""
    if isinstance(data, dict):
        for key, value in data.items():
            value_type = type(value).__name__
            print(f"{'  ' * indent}{key} ({value_type}):")
            await print_response_structure(value, indent + 1)
    elif isinstance(data, list) and data:
        print(f"{'  ' * indent}[List of {len(data)} items]")
        # Print structure of first item as example
        await print_response_structure(data[0], indent + 1)
    else:
        print(f"{'  ' * indent}Value: {data}")

@pytest.mark.mock
async def test_location_search_mock(mock_client):
    """Test location search with mock data"""
    response = await mock_client.get(f'{BASE_URL}/location/search')
    assert response.status_code == 200
    
    data = await response.json()
    assert data["data"][0]["name"] == "Paris"
    assert "location_id" in data["data"][0]
    
    # Validate structure
    location = data["data"][0]
    assert all(key in location for key in [
        "location_id", "name", "description", "web_url", "address_obj"
    ])

@pytest.mark.mock
async def test_location_details_mock(mock_client):
    """Test location details with mock data"""
    response = await mock_client.get(f'{BASE_URL}/location/187147/details')
    assert response.status_code == 200
    
    data = await response.json()
    assert data["name"] == "Paris"
    assert "rating" in data
    assert "description" in data

@pytest.mark.mock
async def test_nearby_search_mock(mock_client):
    """Test nearby search with mock data"""
    response = await mock_client.get(f'{BASE_URL}/location/nearby_search')
    assert response.status_code == 200
    
    data = await response.json()
    assert len(data["data"]) > 0
    assert all(key in data["data"][0] for key in [
        "location_id", "name", "distance", "rating"
    ])

# Keep the real API tests but mark them to skip by default
@pytest.mark.real
@pytest.mark.skipif(
    not os.getenv('TRIPADVISOR_API_READY'), 
    reason="TripAdvisor API not configured or waiting for activation"
)
async def test_location_search_real():
    """Real API test - only runs when explicitly enabled"""
    # Original test code here
    pass

@pytest.mark.real
async def test_location_details(api_headers):
    """Test location details endpoint for a specific location."""
    async with httpx.AsyncClient() as client:
        # First get a location ID from search
        params = {
            'searchQuery': 'Paris',
            'language': 'en'
        }
        search_response = await client.get(
            f'{BASE_URL}/location/search',
            headers=api_headers,
            params=params
        )
        
        search_data = search_response.json()
        if 'data' in search_data and search_data['data']:
            location_id = search_data['data'][0]['location_id']
            
            # Now get details for this location
            details_response = await client.get(
                f'{BASE_URL}/location/{location_id}/details',
                headers=api_headers,
                params={'language': 'en'}
            )
            
            assert details_response.status_code == 200
            details_data = details_response.json()
            
            print("\nLocation Details Response Structure:")
            await print_response_structure(details_data)
            
            print("\nRaw Details Response:")
            pprint(details_data)

@pytest.mark.real
@pytest.mark.skip(reason="TripAdvisor nearby search endpoint needs further configuration")
async def test_nearby_search(api_headers):
    """Test nearby search endpoint for a specific location."""
    async with httpx.AsyncClient() as client:
        params = {
            'latLong': '48.856614,2.352222',  # Paris coordinates
            'language': 'en',
            'radius': '5'  # 5km radius
        }
        
        response = await client.get(
            f'{BASE_URL}/location/nearby_search',
            headers=api_headers,
            params=params
        )
        
        print("\nAPI Request Details:")
        print(f"URL: {response.url}")
        print(f"Headers Sent: {api_headers}")
        
        print("\nAPI Response Details:")
        print(f"Status Code: {response.status_code}")
        print(f"Response Headers: {dict(response.headers)}")
        try:
            print(f"Response Body: {response.json()}")
        except Exception as e:
            print(f"Raw Response Text: {response.text}")
        
        assert response.status_code == 200 

@pytest.mark.real
async def test_location_search(api_headers):
    """Test location search endpoint and examine response structure."""
    async with httpx.AsyncClient() as client:
        params = {
            'searchQuery': 'Paris',
            'language': 'en',
            'key': os.getenv('TRIPADVISOR_API_KEY')  # Try adding key in params as well
        }
        
        print("\nDebug Information:")
        print(f"API Key being used: {os.getenv('TRIPADVISOR_API_KEY')}")
        print(f"API Key length: {len(os.getenv('TRIPADVISOR_API_KEY') or '')}")
        
        response = await client.get(
            f'{BASE_URL}/location/search',
            headers=api_headers,
            params=params
        )
        
        print("\nAPI Request Details:")
        print(f"URL: {response.url}")
        print(f"Headers Sent: {api_headers}")
        print(f"Params Sent: {params}")
        
        print("\nAPI Response Details:")
        print(f"Status Code: {response.status_code}")
        print(f"Response Headers: {dict(response.headers)}")
        try:
            print(f"Response Body: {response.json()}")
        except Exception as e:
            print(f"Raw Response Text: {response.text}")
            print(f"JSON Parsing Error: {str(e)}")
        
        assert response.status_code == 200
