import plotly.graph_objects as go
import plotly.io as pio

pio.templates["patterndetectives"] = go.layout.Template(
    layout=dict(
    colorscale_sequential = ["#6d45ae", "#9540b2", "#ba36b1", "#de24ab", "#ff0085"]
    colorway = ["#ff0085", "#de24ab", "#ba36b1", "#9540b2", "#6d45ae", "#705589", "#392323", "#968478"],
    paper_bgcolor = "#f9f4e3",
    plot_bgcolor = "#e9dcc8"
    )
)

pio.templates.default = "plotly+patterndetectives"