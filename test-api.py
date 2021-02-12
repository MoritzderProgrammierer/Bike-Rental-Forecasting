import requests, json

url = '[http://127.0.0.1:5000/api/makecalc/](http://127.0.0.1:5000/api/makecalc/)'

text = json.dumps({"0":{"instant":1,"dteday":"2011-01-01T00:00:00.000Z","season":1,"yr":0,"mnth":1,"holiday":0,"weekday":6,"workingday":0,"weathersit":2,"temp":0.344167,"atemp":0.363625,"hum":0.805833,"windspeed":0.160446},

               	"1":{"instant":2,"dteday":"2011-01-02T00:00:00.000Z","season":1,"yr":0,"mnth":1,"holiday":0,"weekday":3,"workingday":0,"weathersit":2,"temp":0.363478,"atemp":0.353739,"hum":0.696087,"windspeed":0.248539},

               	"2":{"instant":3,"dteday":"2011-01-03T00:00:00.000Z","season":1,"yr":0,"mnth":1,"holiday":0,"weekday":1,"workingday":1,"weathersit":1,"temp":0.196364,"atemp":0.189405,"hum":0.437273,"windspeed":0.248309}})

headers = {'content-type': 'application/json', 'Accept-Charset': 'UTF-8'}
r = requests.post(url, data=text, headers=headers)
print(r,r.text)