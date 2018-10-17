import json
import requests

OMDB_URL = "http://www.omdbapi.com/?i=tt%s&plot=full&r=json"

fdata = open("tagged_plots.json", 'w')
flink = open("links.csv", 'r')
for line in flink:
    if line.startswith('movieId'):
        continue
    mid, imdb_id, _ = line.strip().split(",")
    resp = requests.get(OMDB_URL % (imdb_id))
    resp_json = json.loads(resp.text)
    json.dump(resp_json, fdata)
    fdata.write("\n")
flink.close()
fdata.close()


