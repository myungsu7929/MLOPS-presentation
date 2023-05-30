import requests
import msgpack

test_img_bytes = open('./000.png', 'rb').read()

payload = msgpack.packb({
    "img" : test_img_bytes
})

response = requests.post("http://example-load-balancer-1062794341.us-east-1.elb.amazonaws.com:80/gender_filter", payload)
print(response)
result = msgpack.unpackb(response.content)
print(result['result'])