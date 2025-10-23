# Flask API

A simple Flask API with a single root endpoint.

## Setup

1. Install dependencies:

```powershell
pip install -r requirements.txt
```

2. Run the application:

```powershell
python app.py
```

3. The API will be available at `http://localhost:5000/`

## Endpoint

- **GET /** - Returns a welcome message in JSON format

## Example Response

```json
{
  "message": "Hello from Flask API!",
  "status": "success"
}
```
