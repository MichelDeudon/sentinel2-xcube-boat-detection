import dash
import dash_core_components as dcc
import dash_html_components as html

from dash.dependencies import Input, Output
from plotly import graph_objs as go
from plotly.graph_objs import *
import json
import datetime
import glob
import os
from src.utils import get_monthly_aggregation, get_selection

app = dash.Dash(
    __name__, meta_tags=[{"name": "viewport", "content": "width=device-width"}]
)
server = app.server

mapbox_access_token = os.getenv("mapbox_access_token")

list_of_locations = {
    "Corfu": {"lat": 39.68, "lon": 19.92},
    "Dover": {"lat": 51.0, "lon": 1.43},
    "St_Tropez": {"lat": 43.28, "lon": 6.635}
}

global g_selected_location
g_selected_location = None


def get_map_graph():
    zoom = 3
    latInitial = 46.39
    lonInitial = 13.22
    bearing = 0

    return go.Figure(
        data=[
            # Data for all rides based on date and time
            # Plot of all locations on the map
            Scattermapbox(
                lat=[list_of_locations[i]["lat"] for i in list_of_locations.keys()],
                lon=[list_of_locations[i]["lon"] for i in list_of_locations.keys()],
                mode="markers",
                hoverinfo="text",
                text=[i for i in list_of_locations],
                marker=dict(size=10, color="#ffa0a0"),
            ),
        ],
        layout=Layout(
            autosize=True,
            margin=go.layout.Margin(l=0, r=35, t=0, b=0),
            showlegend=False,
            clickmode='event+select',
            # clear_on_unhover=True,
            mapbox=dict(
                accesstoken=mapbox_access_token,
                center=dict(lat=latInitial, lon=lonInitial),  # 40.7272  # -73.991251
                style="dark",
                bearing=bearing,
                zoom=zoom,
            ),
            updatemenus=[
                dict(
                    buttons=(
                        [
                            dict(
                                args=[
                                    {
                                        "mapbox.zoom": zoom,
                                        "mapbox.center.lon": str(lonInitial),
                                        "mapbox.center.lat": str(latInitial),
                                        "mapbox.bearing": 0,
                                        "mapbox.style": "dark",
                                    }
                                ],
                                label="Reset Zoom",
                                method="relayout",
                            )
                        ]
                    ),
                    direction="left",
                    pad={"r": 0, "t": 0, "b": 0, "l": 0},
                    showactive=False,
                    type="buttons",
                    x=0.45,
                    y=0.02,
                    xanchor="left",
                    yanchor="bottom",
                    bgcolor="#323130",
                    borderwidth=1,
                    bordercolor="#6d6d6d",
                    font=dict(color="#FFFFFF"),
                )
            ],
        ),
    )

app.layout = html.Div(
    children=[
        html.Div(
            className="row",
            children=[
                # Column for user controls
                html.Div(
                    className="four columns div-user-controls",
                    children=[
                        html.Img(
                            className="logo", src=app.get_asset_url("dash-logo-new.png")
                        ),
                        html.H2("DASH APP"),
                        html.P(
                            """Select different locations or select
                            different years."""
                        ),
                        # Change to side-by-side for mobile layout
                        html.Div(
                            className="row",
                            children=[
                                html.Div(
                                    className="div-for-dropdown",
                                    children=[
                                        # Dropdown for locations on map
                                        dcc.Dropdown(
                                            id="location-dropdown",
                                            options=[
                                                {"label": i, "value": i}
                                                for i in list_of_locations
                                            ],
                                            placeholder="Select a location",
                                        )
                                    ],
                                ),
                                html.Div(
                                    className="div-for-dropdown",
                                    children=[
                                        # Dropdown to select times
                                        dcc.Dropdown(
                                            id="bar-selector",
                                            options=[
                                                {
                                                    "label": str(y),
                                                    "value": str(y),
                                                }
                                                for y in ["2019", "2020"]
                                            ],
                                            multi=True,
                                            placeholder="Select years",
                                        )
                                    ],
                                ),
                                html.P(
                                    """Mean count of all locations overtime."""
                                ),
                                dcc.Graph(id="histogram"),
                            ],
                        ),
                    ],
                ),
                # Column for app graphs and plots
                html.Div(
                    className="eight columns div-for-charts bg-grey",
                    children=[
                        dcc.Graph(id="map-graph", figure=get_map_graph()),
                        html.Div(
                            className="text-padding",
                            children=[
                                "Select any of the bars on the histogram to section data by time."
                            ],
                        ),
                    ],
                ),
            ],
        )
    ]
)


# Update Histogram Figure based on Month, Day and Times Chosen
@app.callback(
    Output("histogram", "figure"),
    [Input("bar-selector", "value"), Input("location-dropdown", "value"), Input('map-graph', 'clickData')]
)
def update_histogram(barSelection, locationSelection, clickData):
    global g_selected_location
    print(f"bar selection: {barSelection}, locationSelection: {locationSelection}, clickData: {clickData}")
    print(g_selected_location)
    if clickData:
        selected_location = clickData["points"][0]["text"]
        if selected_location != g_selected_location:
            g_selected_location = selected_location
            xVal, yVal, colorVal = get_selection(barSelection, selected_location)
        # print(f"clickData get {clickData}")
    elif locationSelection and g_selected_location != locationSelection:
        g_selected_location = locationSelection
        xVal, yVal, colorVal = get_selection(barSelection, locationSelection)

    if clickData is None and locationSelection is None:
        g_selected_location = None

    if g_selected_location is None:
        xVal, yVal, colorVal = get_monthly_aggregation()

    layout = go.Layout(
        bargap=0.01,
        bargroupgap=0,
        barmode="group",
        margin=go.layout.Margin(l=10, r=0, t=0, b=50),
        showlegend=False,
        plot_bgcolor="#323130",
        paper_bgcolor="#323130",
        dragmode="select",
        font=dict(color="white"),
        xaxis=dict(
            range=[min(xVal), max(xVal)],
            showgrid=False,
            nticks=25,
            fixedrange=True,
            ticksuffix=":00",
        ),
        yaxis=dict(
            range=[0, max(yVal)*1.25],
            showticklabels=False,
            showgrid=False,
            fixedrange=True,
            rangemode="nonnegative",
            zeroline=False,
        ),
        annotations=[
            dict(
                x=xi,
                y=yi,
                text=str(yi),
                xanchor="center",
                yanchor="bottom",
                showarrow=False,
                font=dict(color="white"),
            )
            for xi, yi in zip(xVal, yVal)
        ],
    )

    return go.Figure(
        data=[
            go.Bar(x=xVal, y=yVal, marker=dict(color=colorVal), hoverinfo="x"),
            go.Scatter(
                opacity=0,
                x=xVal,
                y=yVal,
                hoverinfo="none",
                mode="markers",
                marker=dict(color="rgb(66, 134, 244, 0)", symbol="square", size=40),
                visible=True,
            ),
        ],
        layout=layout,
    )


@app.callback(
    Output("location-dropdown", "value"),
    [Input('map-graph', 'clickData')]
)

def update_location_selection_dropdown(clickData):
    # print(f"clickData {clickData}")
    if clickData:
        selected_location = clickData["points"][0]["text"]
        return selected_location
    else:
        return None



if __name__ == "__main__":
    app.run_server(debug=True)
