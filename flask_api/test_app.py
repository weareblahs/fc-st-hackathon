import sys
sys.path.insert(0, r'c:\Users\tanyu\Documents\school stuff\fc-st-hackathon\flask_api')

from app import app

def test_home_endpoint():
    """Test the root endpoint."""
    with app.test_client() as client:
        response = client.get('/')
        data = response.get_json()
        
        print(f"Status Code: {response.status_code}")
        print(f"Response Data: {data}")
        
        assert response.status_code == 200, f"Expected 200, got {response.status_code}"
        assert data['message'] == "Hello from Flask API!", f"Unexpected message: {data['message']}"
        assert data['status'] == "success", f"Unexpected status: {data['status']}"
        
        print("✓ All tests passed!")

if __name__ == '__main__':
    test_home_endpoint()
