import requests
import argparse
import time
import json

def send_prediction_request(url, image_path):
    """
    Send an image to the prediction API
    
    Args:
        url: API endpoint URL
        image_path: Path to the local image file
        
    Returns:
        API response as JSON
    """
    start_time = time.time()
    
    with open(image_path, 'rb') as f:
        files = {'file': (image_path, f, 'image/jpeg')}
        response = requests.post(f"{url}/predict", files=files)
    
    elapsed_time = time.time() - start_time
    
    print(f"Request took {elapsed_time:.2f} seconds")
    
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error: {response.status_code}")
        print(response.text)
        return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test the CV MLOps API")
    parser.add_argument("--url", default="http://localhost:8000", help="API URL")
    parser.add_argument("--image", required=True, help="Path to image file")
    
    args = parser.parse_args()
    
    result = send_prediction_request(args.url, args.image)
    
    if result:
        print(json.dumps(result, indent=2))
        
        # Print a summary
        predictions = result.get('predictions', [])
        print(f"\nDetected {len(predictions)} objects:")
        
        for pred in predictions:
            print(f"- {pred['class']} (confidence: {pred['confidence']:.2f})")