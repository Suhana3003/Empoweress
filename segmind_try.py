import requests
import base64

api_key = "SEGMIND_API_KEY"
url = "https://api.segmind.com/v1/ssd-1b"

# Request payload
data = {
    "prompt": "A beatiful girl wearing red hat",
    "negative_prompt": "scary, cartoon",
    "samples": 1,
    "scheduler": "UniPC",
    "num_inference_steps": 25,
    "guidance_scale": "9",
    "seed": "36446545871",
    "img_width": "1024",
    "img_height": "1024",
    "base64": False
}

response = requests.post(url, json=data, headers={'x-api-key': api_key})

# Check if the request was successful (status code 200)
if response.status_code == 200:
    # Get the content of the response (the generated image)
    image_content = response.content

    # Save the image to a file
    with open("generated_image.jpg", "wb") as f:
        f.write(image_content)

    print("Image saved successfully.")
else:
    print("Error:", response.status_code, response.text)
