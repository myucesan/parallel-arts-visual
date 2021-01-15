import pandas as pd
import numpy as np
import plotly.express as px
import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html

from dash.dependencies import Input, Output
import dashboard_data as dd

px.set_mapbox_access_token(
    "pk.eyJ1IjoiYmQzYnRlYW0xIiwiYSI6ImNraTY0Zm5haDIwbTcycW1zc3RxcTU1eW4ifQ.Ic3rinQogDpTHOOhOYNDIQ")

fig = px.scatter_mapbox(dd.mapLocations,
                        lat=dd.mapLocations.lat,
                        lon=dd.mapLocations.lng,
                        hover_data=[dd.mapLocations.Hotel_Name, dd.mapLocations.Hotel_Address],
                        width=1500,
                        height=800,
                        zoom=4,
                        title="Hotel Locations of the reviews"
                        )

fig.update_layout(
    mapbox_style="mapbox://styles/bd3bteam1/ckix6zvnw5i0619rpu1i4isl2",
)
nationalityFig = px.pie(dd.reviewerNationalityCountAggregate,
                        values="COUNT(Reviewer_Nationality)",
                        names="Reviewer_Nationality")

additionalScoringFig = px.treemap(dd.additionalNumberDF, path=["_id"], values='value')


app = dash.Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])


@app.callback(Output(component_id="hotel-plot", component_property="figure"),
              Input(component_id="tag-select", component_property="value"))
def updator(tags):
    dd.mapLocations["Tags"] = dd.mapLocations["Tags"].apply(lambda x: str(x).__contains__(tags))
    updated = dd.mapLocations[dd.mapLocations["Tags"] == True]

    updatedFigure = px.scatter_mapbox(updated,
                                      lat=dd.mapLocations.lat,
                                      lon=dd.mapLocations.lng,
                                      hover_data=[dd.mapLocations.Hotel_Name, dd.mapLocations.Hotel_Address],
                                      width=1500,
                                      height=800,
                                      zoom=4
                                      )

    return updatedFigure

app.layout = html.Div([
    dcc.Tabs(id='tabs', value='tab-1', children=[
        dcc.Tab(label='Overview | Zoom | Filter | Details on Demand', value='tab-1'),
        dcc.Tab(label='Graph of Aggregate Data', value='tab-2'),
        dcc.Tab(label='Graph of of Map-Reduce Data', value='tab-3'),
    ]),
    html.Div(id='tabs-example-content')
])

@app.callback(Output('tabs-example-content', 'children'),
              Input('tabs', 'value'))
def render_content(tab):
    if tab == 'tab-1':
        return html.Div([
            html.H3('Tab content 1'),
            dcc.Graph(id="hotel-plot", figure=fig),
            dcc.Dropdown(
                    id='tag-select',
                    options=dd.allTags,
                    value='Leisure trip')
        ])
    elif tab == 'tab-2':
        return html.Div([
            html.H3('Tab content 2'),
            dcc.Graph(id="piechart-nationality", figure=nationalityFig)
        ])
    elif tab == 'tab-3':
        return html.Div([
            html.H3('Tab content 3'),
            dcc.Graph(id="additional-scoring", figure=additionalScoringFig)

        ])



if __name__ == '__main__':
    app.run_server(
        port=8090,
        host='0.0.0.0',
        use_reloader=True
    )
