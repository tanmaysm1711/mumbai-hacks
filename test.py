import requests
r = requests.get('https://docs.google.com/spreadsheet/ccc?key=1h7Sgev9_HCTv7cs9E51icDK95aRut2p1ogP5NjecZjs&output=csv')
data = r.content
print(data)
