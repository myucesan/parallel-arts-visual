import pandas as pd
import plotly.express as px
import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
from pymongo import MongoClient

if __name__ == '__main__':
    print('This code will serve as a comparison for Dask-ML and Spark-Mlib')
    client = MongoClient("localhost:27017")
    db = client.deds
    result = db.reviews.find({}, { "Hotel_Name": 1, "Hotel_Address": 1, "lat": 1, "lng": 1})
    df = pd.DataFrame(list(result))


# Map with hotels on latitude and longitude (zoom
# Filter on tags
# List will all hotels (overeiw)
# Select hotel and get more information (details)

#average score per hotel
# Bar chart of Reviewer nationality

px.set_mapbox_access_token(
    "pk.eyJ1IjoiYmQzYnRlYW0xIiwiYSI6ImNraTY0Zm5haDIwbTcycW1zc3RxcTU1eW4ifQ.Ic3rinQogDpTHOOhOYNDIQ")

mapLocations = df.drop_duplicates(["lat", "lng"])
mapLocations = mapLocations[mapLocations["lat"] != "NA"]
mapLocations = mapLocations[mapLocations["lng"] != "NA"]
mapLocations["lat"] = pd.to_numeric(mapLocations["lat"])
mapLocations["lng"] = pd.to_numeric(mapLocations["lng"])
print(mapLocations.info())


fig = px.scatter_mapbox(mapLocations,
                        lat=mapLocations.lat,
                        lon=mapLocations.lng,
                        hover_data=[mapLocations.Hotel_Name, mapLocations.Hotel_Address],
                        width=1500,
                        height=800,
                        zoom=4
                        )

fig.update_layout(
    mapbox_style="mapbox://styles/bd3bteam1/ckix6zvnw5i0619rpu1i4isl2",
)

app = dash.Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])
app.layout = html.Div([
    dcc.Graph(figure=fig)
])


if __name__ == '__main__':
    app.run_server(
        port=8090,
        host='0.0.0.0',
        use_reloader=True
    )

