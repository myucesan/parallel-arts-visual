import pandas as pd
import plotly.express as px
import numpy as np
import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
from pymongo import MongoClient
import re
from dash.dependencies import Input, Output

if __name__ == '__main__':
    print('This code will serve as a comparison for Dask-ML and Spark-Mlib')
    client = MongoClient("localhost:27017")
    db = client.deds
    mapLocations = pd.DataFrame(
        list(db.reviews.find({}, {"Hotel_Name": 1, "Hotel_Address": 1, "Tags": 1, "lat": 1, "lng": 1})))
    uniqueTags = pd.DataFrame(list(db.reviews.find({}, {"Hotel_Name": 1, "Tags": 1})))
    uniqueTags = uniqueTags.drop_duplicates(["Hotel_Name"])["Tags"]
    allTags = list()
    for tags in uniqueTags:
        tags = re.sub(r"[\[\]']", "", tags).split(",")
        for tag in tags:
            tag = tag.strip()
            if tag not in allTags:
                allTags.append({'label': tag, 'value': tag})

# Map with hotels on latitude and longitude (zoom)
# Filter on tags
# List will all hotels (overview)
# Select hotel and get more information (details)

# average score per hotel
# Bar chart of Reviewer nationality

px.set_mapbox_access_token(
    "pk.eyJ1IjoiYmQzYnRlYW0xIiwiYSI6ImNraTY0Zm5haDIwbTcycW1zc3RxcTU1eW4ifQ.Ic3rinQogDpTHOOhOYNDIQ")

mapLocations = mapLocations.drop_duplicates(["lat", "lng"])
mapLocations = mapLocations[mapLocations["lat"] != "NA"]
mapLocations = mapLocations[mapLocations["lng"] != "NA"]
mapLocations["lat"] = pd.to_numeric(mapLocations["lat"])
mapLocations["lng"] = pd.to_numeric(mapLocations["lng"])

# mapLocations["Tags"] = mapLocations["Tags"].apply(
# #     lambda x: re.sub(r"[\[\]']", "", x).split(","))  # converts string to list
# # mapLocations["Tags"] = mapLocations["Tags"].apply(
# #     lambda x: [i.strip() for i in x])  # takes every item in list and removes the whitespace left and right
mapLocations["Tags"] = mapLocations["Tags"].apply(str)
# mapLocations["Tags"] = mapLocations["Tags"].apply(lambda x: x.strip())


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


@app.callback(Output(component_id="hotel-plot", component_property="figure"),
              Input(component_id="tag-select", component_property="value"))
def updator(tags):
    mapLocations["Tags"] = mapLocations["Tags"].apply(lambda x: str(x).__contains__(tags))
    updated = mapLocations[mapLocations["Tags"] == True]

    updatedFigure = px.scatter_mapbox(updated,
                                      lat=mapLocations.lat,
                                      lon=mapLocations.lng,
                                      hover_data=[mapLocations.Hotel_Name, mapLocations.Hotel_Address],
                                      width=1500,
                                      height=800,
                                      zoom=4
                                      )

    return updatedFigure


app.layout = html.Div([
    dcc.Graph(id="hotel-plot", figure=fig),
    dcc.Dropdown(
        id='tag-select',
        options=allTags,
        value='Leisure trip')
])

if __name__ == '__main__':
    app.run_server(
        port=8090,
        host='0.0.0.0',
        use_reloader=True
    )
