from PIL import Image
import urllib.request
import io

def test_skin():
    # Fetch a dummy image (e.g. a red apple and a person)
    url_apple = "https://upload.wikimedia.org/wikipedia/commons/1/15/Red_Apple.jpg"
    url_face = "https://upload.wikimedia.org/wikipedia/commons/e/e0/Placeholder_female_superhero.png"
    
    for url in [url_apple, url_face]:
        req = urllib.request.urlopen(url)
        img = Image.open(io.BytesIO(req.read())).convert('HSV')
        pixels = img.getdata()
        
        skin_pixels = 0
        step = 4
        sampled_pixels = len(pixels) // step
        
        for i in range(0, len(pixels), step):
            h, s, v = pixels[i]
            if (h <= 30 or h >= 240) and s > 20 and v > 30:
                skin_pixels += 1
                
        ratio = skin_pixels / sampled_pixels
        print(f"URL: {url.split('/')[-1]}, Skin Ratio: {ratio*100:.2f}%")

test_skin()
