from flask import Flask, request, render_template, jsonify
import requests
import math

app = Flask(__name__)

HEADERS = {"User-Agent": "RelaxationFinder/1.0 (contact: example@example.com)"}

CATEGORY_MAP = {
    ("amenity", "place_of_worship"): "Meditation / Worship",
    ("leisure", "spa"): "Spas",
    ("leisure", "park"): "Parks",
    ("leisure", "garden"): "Parks",
    ("amenity", "community_centre"): "Community Centres",
    ("amenity", "theatre"): "Arts & Theatre",
    ("amenity", "arts_centre"): "Arts & Theatre",
    ("amenity", "cinema"): "Cinema",
    ("amenity", "music_venue"): "Music Venues",
    ("tourism", "museum"): "Museums & Galleries",
    ("tourism", "gallery"): "Museums & Galleries",
}

CATEGORY_EMOJI = {
    "Meditation / Worship": "🧘",
    "Spas": "💆",
    "Parks": "🌳",
    "Community Centres": "🏛️",
    "Arts & Theatre": "🎭",
    "Cinema": "🎬",
    "Music Venues": "🎵",
    "Museums & Galleries": "🖼️",
    "Other": "⭐",
}

# India bounding box (lat_min, lat_max, lon_min, lon_max)
INDIA_BBOX = (6.5546079, 35.6745457, 68.1113787, 97.395561)

def is_in_india(lat, lng):
    return (INDIA_BBOX[0] <= lat <= INDIA_BBOX[1] and INDIA_BBOX[2] <= lng <= INDIA_BBOX[3])

def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    dlat = math.radians(float(lat2) - float(lat1))
    dlon = math.radians(float(lon2) - float(lon1))
    a = (math.sin(dlat / 2) ** 2 +
         math.cos(math.radians(float(lat1))) * math.cos(math.radians(float(lat2))) *
         math.sin(dlon / 2) ** 2)
    c = 2 * math.asin(math.sqrt(a))
    return R * c

def overpass_query(lat, lng, radius_m=5000):
    q = f"""
    [out:json][timeout:25];
    (
      node["amenity"="place_of_worship"](around:{radius_m},{lat},{lng});
      node["leisure"="spa"](around:{radius_m},{lat},{lng});
      node["leisure"="park"](around:{radius_m},{lat},{lng});
      node["leisure"="garden"](around:{radius_m},{lat},{lng});
      node["amenity"="community_centre"](around:{radius_m},{lat},{lng});
      node["amenity"="theatre"](around:{radius_m},{lat},{lng});
      node["amenity"="arts_centre"](around:{radius_m},{lat},{lng});
      node["amenity"="cinema"](around:{radius_m},{lat},{lng});
      node["amenity"="music_venue"](around:{radius_m},{lat},{lng});
      node["tourism"="museum"](around:{radius_m},{lat},{lng});
      node["tourism"="gallery"](around:{radius_m},{lat},{lng});
    );
    out center tags;
    """
    url = "https://overpass-api.de/api/interpreter"
    r = requests.post(url, data={"data": q}, headers=HEADERS, timeout=40)
    r.raise_for_status()
    return r.json()

def detect_category(tags):
    for (k, v), label in CATEGORY_MAP.items():
        if tags.get(k) == v:
            return label
    if "amenity" in tags:
        return tags["amenity"].replace("_", " ").title()
    if "leisure" in tags:
        return tags["leisure"].replace("_", " ").title()
    if "tourism" in tags:
        return tags["tourism"].replace("_", " ").title()
    return "Other"

@app.route("/")
def index():
    return render_template("index_osm.html", category_emoji=CATEGORY_EMOJI)

@app.route("/nearby", methods=["POST"])
def nearby():
    data = request.get_json()
    lat = float(data.get("lat"))
    lng = float(data.get("lng"))
    radius_m = int(data.get("radius_m", 5000))

    raw = overpass_query(lat, lng, radius_m)
    elements = raw.get("elements", [])

    flat_results = []
    for el in elements:
        tags = el.get("tags", {})
        name = tags.get("name", "").strip()

        # Skip if name is empty or "Unnamed Place"
        if not name or name.lower() == "unnamed place":
            continue

        lat2 = el.get("lat")
        lng2 = el.get("lon")

        # Skip if not in India
        if not is_in_india(lat2, lng2):
            continue

        cat = detect_category(tags)
        dist_km = round(haversine(lat, lng, lat2, lng2), 2)

        flat_results.append({
            "name": name,
            "category": cat,
            "emoji": CATEGORY_EMOJI.get(cat, CATEGORY_EMOJI["Other"]),
            "lat": lat2,
            "lng": lng2,
            "distance_km": dist_km,
            "osm_link": f"https://www.openstreetmap.org/?mlat={lat2}&mlon={lng2}#map=18/{lat2}/{lng2}"
        })

    flat_results.sort(key=lambda x: x["distance_km"])

    categories = {}
    for item in flat_results:
        categories.setdefault(item["category"], []).append(item)

    return jsonify({
        "center": {"lat": lat, "lng": lng},
        "categories": categories,
        "all": flat_results
    })

if __name__ == "__main__":
    app.run(debug=True)
