import requests
import json

# URL for the web service, should be similar to:
# 'http://8530a665-66f3-49c8-a953-b82a2d312917.eastus.azurecontainer.io/score'
scoring_uri = "http://224563f7-bd3f-4b01-a721-c698e57386ac.eastus2.azurecontainer.io/score"

# If the service is authenticated, set the key or token
key = "aTQuU1JruHFu7B0Mw27BxG9kI17r1R2o"

# Two sets of data to score, so we get two results back
data = {
    "data": [
        {
            "instant": 1,
            "date": "2013-01-01 00:00:00,000000",
            "season": 1,
            "yr": 0,
            "mnth": 1,
            "weekday": 10,
            "weathersit": 2,
            "temp": 0.344167,
            "atemp": 0.123625,
            "hum": 0.405833,
            "windspeed": 0.160446,
            "casual": 331,
            "registered": 654,
        },
    ]
}
# Convert to JSON string
input_data = json.dumps(data)
with open("data.json", "w") as _f:
    _f.write(input_data)

# Set the content type
headers = {"Content-Type": "application/json"}
# If authentication is enabled, set the authorization header
headers["Authorization"] = f"Bearer {key}"

# Make the request and display the response
resp = requests.post(scoring_uri, input_data, headers=headers)
print(resp.json())
