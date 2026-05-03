import json
from mangum import Mangum
from app.api.endpoints import app

# Create handler for AWS Lambda
handler = Mangum(app)